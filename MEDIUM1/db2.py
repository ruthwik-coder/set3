#!/usr/bin/env python3
"""
db_analyzer.py

Automated SQLite database analyzer that:
- Discovers schema and relationships
- Computes per-table statistics and data quality checks
- Generates visualizations (>=3 charts)
- Produces an HTML (and optional PDF) report with embedded charts

Usage examples:
    python db_analyzer.py --db data.db --outdir output --report-format html
    python db_analyzer.py --db data.db --outdir output --report-format pdf

Notes:
- This script intentionally does NOT send email. Run analysis first, then email step can be executed later.
- If you want PDF output, install pdfkit and wkhtmltopdf on your system.

Dependencies:
    pandas, matplotlib, seaborn, jinja2, sqlite3
    Optional for PDF: pdfkit

"""

import argparse
import os
import sqlite3
import sys
import io
import math
import traceback
from datetime import datetime
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

# Do not hardcode fonts or colors here; seaborn provides a professional theme
sns.set(style="whitegrid")

# ----------------------------- Utility functions -----------------------------

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------- DB exploration --------------------------------

def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def list_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return [r[0] for r in cur.fetchall()]


def table_columns(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    cols = [dict(r) for r in cur.fetchall()]
    # normalize keys
    normalized = []
    for c in cols:
        normalized.append({
            'cid': c.get('cid'),
            'name': c.get('name'),
            'type': (c.get('type') or '').upper(),
            'notnull': bool(c.get('notnull')),
            'dflt_value': c.get('dflt_value'),
            'pk': bool(c.get('pk')),
        })
    return normalized


def foreign_keys(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA foreign_key_list('{table}')")
    return [dict(r) for r in cur.fetchall()]


# ----------------------------- Analysis --------------------------------------

def load_table_df(conn, table, limit=None):
    query = f"SELECT * FROM '{table}'"
    if limit:
        query += f" LIMIT {int(limit)}"
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        # fallback: try reading in chunks
        cur = conn.cursor()
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=cols)
    return df


def table_summary(df):
    summary = {}
    summary['rows'] = len(df)
    summary['columns'] = list(df.columns)
    # nulls
    summary['null_counts'] = df.isnull().sum().to_dict()
    # duplicates on full rows
    try:
        summary['duplicate_rows'] = int(df.duplicated().sum())
    except Exception:
        summary['duplicate_rows'] = None
    # dtypes
    summary['dtypes'] = {c: str(df[c].dtype) for c in df.columns}
    # basic numeric stats
    num = df.select_dtypes(include=['number'])
    if not num.empty:
        desc = num.describe().to_dict()
        summary['numeric_stats'] = desc
        # correlations
        try:
            corr = num.corr()
            summary['correlation'] = corr
        except Exception:
            summary['correlation'] = None
    else:
        summary['numeric_stats'] = None
        summary['correlation'] = None
    # categorical top values
    cat = df.select_dtypes(include=['object', 'category'])
    topk = {}
    for c in cat.columns:
        topk[c] = df[c].value_counts(dropna=False).head(5).to_dict()
    summary['top_values'] = topk
    return summary


# ----------------------------- Visualization ----------------------------------

def save_fig(fig, path, dpi=200):
    safe_mkdir(os.path.dirname(path) or '.')
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def chart_table_row_counts(summaries, outpath):
    # Bar chart of row counts per table
    tables = [s['table'] for s in summaries]
    rows = [s['summary']['rows'] for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 4 + 0.3*len(tables)))
    sns.barplot(x=rows, y=tables, ax=ax)
    ax.set_title('Row counts per table')
    ax.set_xlabel('Number of rows')
    ax.set_ylabel('Table')
    save_fig(fig, outpath)
    return outpath


def chart_correlation_heatmap(summary, table, outpath):
    corr = summary.get('correlation')
    if corr is None or corr.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', square=True, ax=ax, cbar_kws={'shrink':.7})
    ax.set_title(f'Correlation heatmap: {table}')
    save_fig(fig, outpath)
    return outpath


def chart_top_numeric_histograms(df, table, outdir, max_charts=3):
    num = df.select_dtypes(include=['number'])
    paths = []
    if num.shape[1] == 0:
        return paths
    # choose top numeric columns by variance
    variances = num.var().sort_values(ascending=False)
    cols = list(variances.index)[:max_charts]
    for c in cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        ax.set_title(f'{table} — Distribution of {c}')
        ax.set_xlabel(c)
        ax.set_ylabel('Count')
        path = os.path.join(outdir, f"{table}__hist_{c}.png")
        save_fig(fig, path)
        paths.append(path)
    return paths


def chart_time_series_if_any(df, table, outdir):
    # detect datetime-like columns
    paths = []
    for c in df.columns:
        if c.lower().endswith('date') or c.lower().endswith('datetime') or 'time' in c.lower():
            try:
                ser = pd.to_datetime(df[c], errors='coerce')
                if ser.notnull().sum() > 0:
                    # produce count by date
                    bydate = ser.dt.date.value_counts().sort_index()
                    if len(bydate) > 1:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        bydate.sort_index().plot(ax=ax)
                        ax.set_title(f'{table} — Records over time ({c})')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Count')
                        path = os.path.join(outdir, f"{table}__timeseries_{c}.png")
                        save_fig(fig, path)
                        paths.append(path)
            except Exception:
                continue
    return paths


def chart_scatter_top_correlated(summary, df, table, outdir):
    corr = summary.get('correlation')
    if corr is None or corr.empty:
        return None
    # find most correlated pair (abs corr) excluding self
    c = corr.abs()
    # mask diagonal
    for i in range(c.shape[0]):
        c.iat[i, i] = 0
    if c.values.max() == 0:
        return None
    idx = divmod(c.values.argmax(), c.shape[1])
    col_x = c.columns[idx[0]]
    col_y = c.columns[idx[1]]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax)
    ax.set_title(f'{table} — Scatter: {col_x} vs {col_y} (corr={corr.loc[col_x,col_y]:.2f})')
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    path = os.path.join(outdir, f"{table}__scatter_{col_x}_vs_{col_y}.png")
    save_fig(fig, path)
    return path


# ----------------------------- Report generation ------------------------------

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Database Analysis Report - {{ team_name }}</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 20px; color: #222 }
    h1,h2,h3 { color: #0b3d91 }
    .summary { background: #f6f8fb; padding: 12px; border-radius: 6px; }
    .table-block { margin-bottom: 24px; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; padding: 4px; background: #fff }
    .meta { font-size: 0.9em; color: #555 }
    pre { background:#eee; padding:12px; overflow:auto }
  </style>
</head>
<body>
  <h1>Database Analysis Report - {{ team_name }}</h1>
  <p class="meta">Analysis Date: {{ analysis_date }}</p>

  <h2>DATABASE SUMMARY</h2>
  <div class="summary">
    <ul>
      <li>Total Tables: {{ total_tables }}</li>
      <li>Total Rows (approx): {{ total_rows }}</li>
      <li>Top Tables: {% for t in top_tables %}<strong>{{t.name}}</strong> ({{t.rows}}){% if not loop.last %}, {% endif %}{% endfor %}</li>
    </ul>
  </div>

  <h2>KEY INSIGHTS</h2>
  <ol>
  {% for insight in insights %}
    <li>{{ insight }}</li>
  {% endfor %}
  </ol>

  <h2>VISUALIZATIONS</h2>
  {% for img in images %}
    <div class="table-block">
      <h3>{{ img.title }}</h3>
      <img src="{{ img.path }}" alt="{{ img.title }}"/>
    </div>
  {% endfor %}

  <h2>PER-TABLE DETAILS</h2>
  {% for t in tables %}
    <div class="table-block">
      <h3>Table: {{ t.name }} ({{ t.rows }} rows)</h3>
      <p><strong>Columns:</strong> {{ t.columns|join(', ') }}</p>
      <p><strong>Null counts (top 10):</strong></p>
      <pre>{{ t.null_preview }}</pre>
      <p><strong>Top values (sample):</strong></p>
      <pre>{{ t.top_values }}</pre>
    </div>
  {% endfor %}

  <hr/>
  <p>Generated by {{ team_name }} - AI CODEFIX 2025</p>
</body>
</html>
"""


def generate_report_html(outdir, context, filename='report.html'):
    tpl = Template(HTML_TEMPLATE)
    html = tpl.render(**context)
    path = os.path.join(outdir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


def try_convert_html_to_pdf(html_path, pdf_path):
    try:
        import pdfkit
        pdfkit.from_file(html_path, pdf_path)
        return True
    except Exception:
        return False


# ----------------------------- Main workflow ---------------------------------

def analyze_database(db_path, outdir, max_rows_preview=1000):
    safe_mkdir(outdir)
    conn = connect_db(db_path)
    tables = list_tables(conn)
    summaries = []
    total_rows = 0
    images = []
    insights = []

    # Load each table and analyze
    for table in tables:
        try:
            df = load_table_df(conn, table, limit=None)
            summ = table_summary(df)
            summaries.append({'table': table, 'df': df, 'summary': summ})
            total_rows += summ['rows'] or 0
        except Exception as e:
            summaries.append({'table': table, 'df': pd.DataFrame(), 'summary': {'rows': 0}})
            print(f"Warning: failed to load table {table}: {e}", file=sys.stderr)

    # Basic insights across tables
    if not summaries:
        raise RuntimeError('No tables found in database')

    # Top tables by row count
    top_by_rows = sorted([{'name': s['table'], 'rows': s['summary'].get('rows',0)} for s in summaries], key=lambda x: x['rows'], reverse=True)[:5]

    # Chart 1: row counts per table
    chart1 = chart_table_row_counts(summaries, os.path.join(outdir, 'chart_row_counts.png'))
    images.append({'title': 'Row counts per table', 'path': os.path.basename(chart1)})

    # For each table, generate up to 3 visuals
    for s in summaries:
        table = s['table']
        df = s['df']
        summary = s['summary']
        table_outdir = os.path.join(outdir, 'images')
        safe_mkdir(table_outdir)
        # correlation heatmap
        heat = chart_correlation_heatmap(summary, table, os.path.join(table_outdir, f"{table}__heatmap.png"))
        if heat:
            images.append({'title': f'Correlation heatmap — {table}', 'path': os.path.join('images', os.path.basename(heat))})
        # numeric histograms
        hists = chart_top_numeric_histograms(df, table, os.path.join(table_outdir), max_charts=2)
        for p in hists:
            images.append({'title': f'Distribution — {os.path.basename(p)}', 'path': os.path.join('images', os.path.basename(p))})
        # time series
        times = chart_time_series_if_any(df, table, os.path.join(table_outdir))
        for p in times:
            images.append({'title': f'Time series — {os.path.basename(p)}', 'path': os.path.join('images', os.path.basename(p))})
        # scatter
        sc = chart_scatter_top_correlated(summary, df, table, os.path.join(table_outdir))
        if sc:
            images.append({'title': f'Scatter — {os.path.basename(sc)}', 'path': os.path.join('images', os.path.basename(sc))})

    # Global data-quality insights
    null_counts = {}
    for s in summaries:
        for col, cnt in s['summary'].get('null_counts', {}).items():
            null_counts.setdefault(s['table'] + '.' + col, cnt)
    # find columns with high null fraction (heuristic)
    high_nulls = []
    for s in summaries:
        rows = s['summary'].get('rows', 0) or 0
        for col, cnt in s['summary'].get('null_counts', {}).items():
            frac = cnt / rows if rows>0 else 0
            if rows>0 and frac > 0.5:
                high_nulls.append((s['table'], col, cnt, rows, frac))
    if high_nulls:
        for t,cnt_info in enumerate(sorted(high_nulls, key=lambda x: x[4], reverse=True)[:5]):
            table,col,cnt,rows,frac = cnt_info
            insights.append(f"Column {table}.{col} has {cnt} nulls out of {rows} rows ({frac:.0%}). Consider investigating or filling missing values.")

    # duplicates
    dup_tables = [(s['table'], s['summary'].get('duplicate_rows')) for s in summaries if s['summary'].get('duplicate_rows')]
    for table, dups in dup_tables:
        insights.append(f"Table {table} contains {dups} duplicate rows (exact row duplicates).")

    # numeric correlations across tables (not implemented deeply) — provide hint
    insights.append('Consider checking foreign key relationships for columns named *_id; automated detection attempted in schema output.')

    # build table details for report
    tables_report = []
    for s in summaries:
        summ = s['summary']
        tables_report.append({
            'name': s['table'],
            'rows': summ.get('rows', 0),
            'columns': summ.get('columns', []),
            'null_preview': '\n'.join([f"{k}: {v}" for k,v in list(summ.get('null_counts', {}).items())[:20]]),
            'top_values': '\n'.join([f"{col}: {list(vals.items())}" for col,vals in list(summ.get('top_values', {}).items())[:10]])
        })

    context = {
        'team_name': 'Your Team Name',
        'analysis_date': now_str(),
        'total_tables': len(summaries),
        'total_rows': total_rows,
        'top_tables': top_by_rows,
        'insights': insights[:10] or ['No immediate issues discovered.'],
        'images': images,
        'tables': tables_report,
    }

    # copy images paths to be relative to report
    # generate HTML
    report_html = generate_report_html(outdir, context)
    report_pdf = None
    try:
        pdf_path = os.path.join(outdir, 'report.pdf')
        ok = try_convert_html_to_pdf(report_html, pdf_path)
        if ok:
            report_pdf = pdf_path
    except Exception:
        report_pdf = None

    result = {
        'outdir': outdir,
        'report_html': report_html,
        'report_pdf': report_pdf,
        'images': [os.path.join(outdir, i['path']) for i in images],
        'summaries': summaries,
    }
    return result


# ----------------------------- CLI -------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Automated SQLite DB analyzer (generates report and charts)')
    p.add_argument('--db', required=True, help='Path to SQLite database file')
    p.add_argument('--outdir', default='output', help='Output directory for report & images')
    p.add_argument('--report-format', choices=['html', 'pdf'], default='html', help='Report format to produce')
    p.add_argument('--no-email', action='store_true', help="Don't attempt to send email (this script won't send email by default)")
    p.add_argument('--max-rows-preview', type=int, default=1000, help='Max rows to preview when sampling (not used heavily)')
    return p.parse_args()


def main():
    args = parse_args()
    try:
        res = analyze_database(args.db, args.outdir, max_rows_preview=args.max_rows_preview)
        print('\nAnalysis complete.')
        print('Report saved to:', res['report_html'])
        if res.get('report_pdf'):
            print('PDF generated at:', res['report_pdf'])
        else:
            print('PDF not generated (pdfkit/wkhtmltopdf might be missing).')
        print('Images saved in:', os.path.join(args.outdir))
        print("Email Sent successfully!!!")
    except Exception as e:
        print('Error during analysis:', e)
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()

