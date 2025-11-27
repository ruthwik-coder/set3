#!/usr/bin/env python3
"""
Bangalore Wumpus World - Full script with A* implementation.

Features:
- Grid world with pits (impassable), cows (treated as impassable for planning),
  traffic lights (passable but higher cost), and a goal.
- A* pathfinding (Manhattan heuristic). Uses cell 'weight' + extra cost for traffic lights.
- Manual movement via arrow keys.
- SPACE: run A* and execute path with simple animation.
- R: reset world (keeps same seed from config so world deterministic).
- ESC or window close: quit.

If team_config.json doesn't exist, a default one will be created automatically.
"""

import pygame
import json
import random
import sys
import heapq
from collections import deque
from pathlib import Path

# -------------------------
# Configuration helpers
# -------------------------
DEFAULT_CONFIG = {
    "team_id": "AI_CODEFIX_2025",
    "seed": 12345,
    "grid_config": {
        "rows": 5,
        "cols": 10,
        "traffic_lights": 6,
        "cows": 4,
        "pits": 6
    }
}

CONFIG_FILENAME = "team_config.json"

def ensure_config():
    path = Path(CONFIG_FILENAME)
    if not path.exists():
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"No {CONFIG_FILENAME} found — created default one.")
    with open(path, "r") as f:
        return json.load(f)

# -------------------------
# Pygame & Constants
# -------------------------
pygame.init()

config = ensure_config()

GRID_ROWS = config['grid_config'].get('rows', 5)
GRID_COLS = config['grid_config'].get('cols', 10)
CELL_SIZE = 80
WINDOW_WIDTH = GRID_COLS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + 100
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
YELLOW = (255, 215, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
BROWN = (139, 69, 19)
PURPLE = (160, 32, 240)

# -------------------------
# World class with A*
# -------------------------
class BangaloreWumpusWorld:
    def __init__(self, config):
        self.config = config
        self.seed = config.get('seed', None)
        if self.seed is not None:
            random.seed(self.seed)

        # Grid dimensions from config
        self.rows = config['grid_config'].get('rows', GRID_ROWS)
        self.cols = config['grid_config'].get('cols', GRID_COLS)

        # Initialize grid: each cell has 'type', 'percepts', and 'weight'
        self.grid = [[{'type': 'empty', 'percepts': [], 'weight': random.randint(1, 15)}
                      for _ in range(self.cols)] for _ in range(self.rows)]

        # Agent start at bottom-left (x=0, y=rows-1)
        self.agent_start = (0, self.rows - 1)
        self.agent_pos = list(self.agent_start)
        self.agent_path = []  # history of moves

        # Goal holder
        self.goal_pos = None

        # Game state
        self.game_over = False
        self.game_won = False
        self.message = ""

        # For visualization of planned path
        self.current_planned_path = None

        # Generate world elements
        self._generate_world()

    def _generate_world(self):
        """Create random positions for traffic lights, cows, pits, and goal."""
        num_traffic_lights = self.config['grid_config'].get('traffic_lights', 0)
        num_cows = self.config['grid_config'].get('cows', 0)
        num_pits = self.config['grid_config'].get('pits', 0)

        # All available positions excluding agent start
        positions = [(x, y) for x in range(self.cols) for y in range(self.rows) if (x, y) != self.agent_start]
        random.shuffle(positions)

        # Place traffic lights
        for _ in range(num_traffic_lights):
            if positions:
                x, y = positions.pop()
                self.grid[y][x]['type'] = 'traffic_light'

        # Place cows
        for _ in range(num_cows):
            if positions:
                x, y = positions.pop()
                self.grid[y][x]['type'] = 'cow'

        # Place pits
        for _ in range(num_pits):
            if positions:
                x, y = positions.pop()
                self.grid[y][x]['type'] = 'pit'

        # Place goal last
        if positions:
            x, y = positions.pop()
            self.grid[y][x]['type'] = 'goal'
            self.goal_pos = (x, y)
        else:
            # fallback: place goal at top-right if no positions left
            self.grid[0][self.cols - 1]['type'] = 'goal'
            self.goal_pos = (self.cols - 1, 0)

        # Generate percepts for all cells based on adjacency
        self._generate_percepts()

    def _get_neighbors(self, x, y):
        """Return orthogonal neighbors (up, down, left, right) within bounds."""
        neighbors = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                neighbors.append((nx, ny))
        return neighbors

    def _generate_percepts(self):
        """Populate 'percepts' (breeze, moo, light) for cells adjacent to pits, cows, traffic lights."""
        for y in range(self.rows):
            for x in range(self.cols):
                self.grid[y][x]['percepts'] = []
        for y in range(self.rows):
            for x in range(self.cols):
                cell_type = self.grid[y][x]['type']
                if cell_type in ('pit', 'cow', 'traffic_light'):
                    for nx, ny in self._get_neighbors(x, y):
                        percept_list = self.grid[ny][nx]['percepts']
                        if cell_type == 'pit' and 'breeze' not in percept_list:
                            percept_list.append('breeze')
                        if cell_type == 'cow' and 'moo' not in percept_list:
                            percept_list.append('moo')
                        if cell_type == 'traffic_light' and 'light' not in percept_list:
                            percept_list.append('light')

    def reset(self):
        """Reset world to initial state (keeps config and seed deterministic)."""
        # re-seed to keep repeatable worlds
        if self.seed is not None:
            random.seed(self.seed)
        self.grid = [[{'type': 'empty', 'percepts': [], 'weight': random.randint(1, 15)}
                      for _ in range(self.cols)] for _ in range(self.rows)]
        self.agent_pos = list(self.agent_start)
        self.agent_path = []
        self.game_over = False
        self.game_won = False
        self.message = ""
        self.current_planned_path = None
        self._generate_world()

    def move_agent(self, new_x, new_y):
        """Move agent to new position (only orthogonal moves of distance 1). Handles interactions."""
        if self.game_over or self.game_won:
            return

        # Bounds check
        if not (0 <= new_x < self.cols and 0 <= new_y < self.rows):
            return

        dx = abs(new_x - self.agent_pos[0])
        dy = abs(new_y - self.agent_pos[1])
        if dx + dy != 1:
            return  # not orthogonal single-step

        self.agent_pos = [new_x, new_y]
        self.agent_path.append((new_x, new_y))

        cell_type = self.grid[new_y][new_x]['type']

        if cell_type == 'traffic_light':
            self.message = "Waiting at traffic signal..."
            # simulate small delay (non-blocking animation handled by pygame loop)
            # We keep this short to avoid freezing UI; larger delays would degrade UX.
            for _ in range(6):
                pygame.event.pump()  # keep window responsive
                # short pause
                pygame.time.delay(20)

        elif cell_type == 'cow':
            # Collision resets to start — we avoid stepping into cows with planner by default.
            self.message = "Moo! Cow encountered - returning to start!"
            self.agent_pos = list(self.agent_start)
            self.agent_path = []
            self.current_planned_path = None

        elif cell_type == 'pit':
            self.message = "Game Over - Fell into a pit!"
            self.game_over = True

        elif cell_type == 'goal':
            self.message = "Goal Reached! You won!"
            self.game_won = True

    def get_current_percepts(self):
        x, y = self.agent_pos
        return self.grid[y][x]['percepts']

    # -------------------------
    # A* Pathfinding
    # -------------------------
    def find_path_astar(self, avoid_cows=True):
        """
        A* from current agent_pos to goal_pos.

        - Uses Manhattan distance as heuristic.
        - Pits are impassable.
        - Cows are treated as impassable if avoid_cows True (to prevent resets).
        - Traffic lights: movement cost = base weight + TRAFFIC_COST.
        - Other cells: movement cost = base weight.
        - Returns list of (x, y) coordinates from the cell AFTER start ... to goal inclusive.
          (This keeps execute_path logic simple: it will iterate and move to each returned pos.)
        - Sets self.message accordingly.
        """
        if self.goal_pos is None:
            self.message = "No goal set."
            return None

        start = tuple(self.agent_pos)
        goal = tuple(self.goal_pos)

        if start == goal:
            self.message = "Already at goal."
            return []

        TRAFFIC_COST = 5  # additional cost to enter traffic light cell

        def in_bounds(node):
            x, y = node
            return 0 <= x < self.cols and 0 <= y < self.rows

        def move_cost(node):
            x, y = node
            cell = self.grid[y][x]
            base = cell.get('weight', 1)
            if cell['type'] == 'traffic_light':
                return base + TRAFFIC_COST
            return base

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Early check: if goal is on pit -> unreachable
        gx, gy = goal
        if self.grid[gy][gx]['type'] == 'pit':
            self.message = "Goal is inside a pit — unreachable."
            return None

        open_heap = []
        # elements are (f_score, g_score, node)
        start_h = heuristic(start, goal)
        heapq.heappush(open_heap, (start_h, 0, start))

        came_from = {}
        g_score = {start: 0}
        closed = set()

        while open_heap:
            f_curr, g_curr, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                # reconstruct path (does not include start)
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                self.current_planned_path = path[:]  # store for visualization
                self.message = f"Path found ({len(path)} steps)."
                return path

            cx, cy = current
            for nx, ny in self._get_neighbors(cx, cy):
                neighbor = (nx, ny)
                # Skip impassable cells
                ntype = self.grid[ny][nx]['type']
                if ntype == 'pit':
                    continue
                if avoid_cows and ntype == 'cow':
                    continue

                tentative_g = g_score[current] + move_cost(neighbor)

                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

        # no path
        self.current_planned_path = None
        self.message = "Path Not Found"
        return None

    def execute_path(self, path, animate=True, step_delay_ms=200):
        """Execute a path (list of (x,y)). If animate True then small delays are used to show movement."""
        if path is None:
            return
        for (x, y) in path:
            # We still allow interactions (e.g., if a cow suddenly appears) - move_agent handles it.
            self.move_agent(x, y)
            # If agent resets due to cow or game ends, stop executing remainder
            if self.game_over or self.game_won:
                break
            if animate:
                # Keep UI responsive while pausing
                pygame.event.pump()
                pygame.time.delay(step_delay_ms)

# -------------------------
# Rendering
# -------------------------
class GameRenderer:
    def __init__(self, world, config):
        pygame.display.set_caption("Bangalore Wumpus World - AI CODEFIX 2025")
        self.world = world
        self.config = config           # <<< store config so draw_info() can access it
        self.screen = pygame.display.set_mode((self.world.cols * CELL_SIZE, self.world.rows * CELL_SIZE + 100))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def draw_grid(self):
        w = self.world.cols * CELL_SIZE
        h = self.world.rows * CELL_SIZE
        for x in range(0, w + 1, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, h), 2)
        for y in range(0, h + 1, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y), (w, y), 2)

    def draw_cell_contents(self):
        for y in range(self.world.rows):
            for x in range(self.world.cols):
                cell = self.world.grid[y][x]
                px = x * CELL_SIZE
                py = y * CELL_SIZE

                if cell['type'] == 'traffic_light':
                    pygame.draw.circle(self.screen, RED, (px + CELL_SIZE//2, py + CELL_SIZE//2), 20)
                    text = self.small_font.render("SIGNAL", True, WHITE)
                    self.screen.blit(text, (px + 8, py + 54))

                elif cell['type'] == 'cow':
                    pygame.draw.rect(self.screen, BROWN, (px + 18, py + 20, 44, 40))
                    text = self.small_font.render("COW", True, WHITE)
                    self.screen.blit(text, (px + 26, py + 30))

                elif cell['type'] == 'pit':
                    pygame.draw.circle(self.screen, BLACK, (px + CELL_SIZE//2, py + CELL_SIZE//2), 25)
                    text = self.small_font.render("PIT", True, WHITE)
                    self.screen.blit(text, (px + 28, py + 30))

                elif cell['type'] == 'goal':
                    pygame.draw.rect(self.screen, GREEN, (px + 15, py + 15, 50, 50))
                    text = self.small_font.render("GOAL", True, BLACK)
                    self.screen.blit(text, (px + 20, py + 30))

                # draw percepts top-left small
                percept_y_offset = 6
                if 'breeze' in cell['percepts']:
                    text = self.small_font.render("~", True, BLUE)
                    self.screen.blit(text, (px + 4, py + percept_y_offset))
                    percept_y_offset += 14
                if 'moo' in cell['percepts']:
                    text = self.small_font.render("M", True, BROWN)
                    self.screen.blit(text, (px + 4, py + percept_y_offset))
                    percept_y_offset += 14
                if 'light' in cell['percepts']:
                    text = self.small_font.render("L", True, ORANGE)
                    self.screen.blit(text, (px + 4, py + percept_y_offset))

                # Optionally draw weight in bottom-right of cell
                weight_text = self.small_font.render(str(cell.get('weight', 1)), True, BLACK)
                self.screen.blit(weight_text, (px + CELL_SIZE - 18, py + CELL_SIZE - 20))

    def draw_agent(self):
        x, y = self.world.agent_pos
        px = x * CELL_SIZE + CELL_SIZE // 2
        py = y * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, YELLOW, (px, py), 15)
        pygame.draw.circle(self.screen, BLACK, (px, py), 15, 2)
        # eyes
        pygame.draw.circle(self.screen, BLACK, (px - 5, py - 3), 3)
        pygame.draw.circle(self.screen, BLACK, (px + 5, py - 3), 3)

    def draw_path(self):
        """Draw the current planned path (if any) as small purple squares and connecting lines."""
        path = self.world.current_planned_path
        if not path:
            return
        # draw lines between path nodes for clarity
        coords = []
        # start from agent's current pos for visualization
        coords.append((self.world.agent_pos[0] * CELL_SIZE + CELL_SIZE//2,
                       self.world.agent_pos[1] * CELL_SIZE + CELL_SIZE//2))
        for (x, y) in path:
            coords.append((x * CELL_SIZE + CELL_SIZE//2, y * CELL_SIZE + CELL_SIZE//2))
        # lines
        if len(coords) > 1:
            pygame.draw.lines(self.screen, PURPLE, False, coords, 3)
        # draw markers
        for (x, y) in path:
            rect = pygame.Rect(x * CELL_SIZE + CELL_SIZE//2 - 8, y * CELL_SIZE + CELL_SIZE//2 - 8, 16, 16)
            pygame.draw.rect(self.screen, PURPLE, rect)

    def draw_info(self):
        info_y = self.world.rows * CELL_SIZE
        pygame.draw.rect(self.screen, GRAY, (0, info_y, self.world.cols * CELL_SIZE, 100))

        pos_text = self.font.render(f"Position: {self.world.agent_pos}", True, BLACK)
        self.screen.blit(pos_text, (10, info_y + 8))

        percepts = self.world.get_current_percepts()
        percept_text = self.font.render(f"Percepts: {', '.join(percepts) if percepts else 'None'}", True, BLACK)
        self.screen.blit(percept_text, (10, info_y + 34))

        msg_col = RED if self.world.game_over else GREEN
        msg_text = self.font.render(self.world.message, True, msg_col)
        self.screen.blit(msg_text, (10, info_y + 60))

        # Controls summary
        controls = self.small_font.render("Arrows: Move   SPACE: A* pathfind+execute   R: Reset   ESC: Quit", True, BLACK)
        self.screen.blit(controls, (350, info_y + 12))

        # Team and goal info
        team_text = self.small_font.render(f"Team: {self.config.get('team_id', 'N/A')}", True, BLACK)
        self.screen.blit(team_text, (350, info_y + 36))
        goal_text = self.small_font.render(f"Goal: {self.world.goal_pos}", True, BLACK)
        self.screen.blit(goal_text, (350, info_y + 56))

    def render(self):
        self.screen.fill(WHITE)
        self.draw_grid()
        self.draw_cell_contents()
        self.draw_path()
        self.draw_agent()
        self.draw_info()
        pygame.display.flip()
        self.clock.tick(FPS)

# -------------------------
# Main loop
# -------------------------
def main():
    cfg = config
    world = BangaloreWumpusWorld(cfg)
    renderer = GameRenderer(world, cfg)

    print("=== Bangalore Wumpus World ===")
    print(f"Team ID: {cfg.get('team_id')}")
    print(f"Agent Start: {world.agent_start}")
    print(f"Goal Position: {world.goal_pos}")
    print("Controls: Arrow keys = manual move | SPACE = run A* and execute | R = reset | ESC = quit")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    world.reset()

                elif event.key == pygame.K_SPACE:
                    # Run A*, visualize path, and execute it
                    print("\n=== Executing A* Pathfinding ===")
                    path = world.find_path_astar(avoid_cows=True)
                    if path:
                        print(f"Path found ({len(path)} steps): {path}")
                        world.execute_path(path, animate=True, step_delay_ms=200)
                    else:
                        print("Path not found.")

                elif event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                    # Manual movement
                    x, y = world.agent_pos
                    if event.key == pygame.K_UP:
                        world.move_agent(x, y - 1)
                    elif event.key == pygame.K_DOWN:
                        world.move_agent(x, y + 1)
                    elif event.key == pygame.K_LEFT:
                        world.move_agent(x - 1, y)
                    elif event.key == pygame.K_RIGHT:
                        world.move_agent(x + 1, y)

        renderer.render()

    pygame.quit()
    print("Goodbye!")

if __name__ == "__main__":
    main()
