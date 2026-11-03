"""Оформление Profitability AI: палитра, CSS и небольшие HTML-компоненты.

Только внешний вид — на расчёты и данные модуль не влияет.
"""
from __future__ import annotations

import html

import streamlit as st

# Палитра (используется и в графиках Plotly)
POS, NEG, NEUTRAL = '#10b981', '#f43f5e', '#64748b'
C_FOT, C_DIRECT, C_OVER, C_REV = '#4f46e5', '#38bdf8', '#f59e0b', '#10b981'
GRID = '#e8eaf3'
FONT = 'Inter, "Segoe UI", system-ui, -apple-system, sans-serif'

PLOT_CFG = dict(
    template='plotly_white',
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=12, r=12, t=56, b=12),
    font=dict(family=FONT, size=13, color='#1e1b3a'),
    title=dict(font=dict(size=16, color='#1e1b3a')),
    hoverlabel=dict(bgcolor='white', font_size=13, bordercolor='#c7cbe6'),
    colorway=[C_FOT, C_REV, C_OVER, C_DIRECT, '#a855f7', NEG],
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {font-family: Inter, "Segoe UI", system-ui, sans-serif;}
.stApp {background: radial-gradient(1200px 500px at 10% -10%, #e9e7ff 0%, rgba(245,246,251,0) 60%),
                    radial-gradient(900px 420px at 100% 0%, #ffe6f1 0%, rgba(245,246,251,0) 55%), #f5f6fb;}
.block-container {padding-top: 1.2rem; max-width: 1500px;}
header[data-testid="stHeader"] {background: transparent;}

/* ---------- Hero ---------- */
.hero {position:relative; overflow:hidden; border-radius:24px; padding:30px 34px; margin:4px 0 18px 0; color:#fff;
       background: linear-gradient(120deg,#3730a3 0%,#6d28d9 52%,#db2777 100%);
       box-shadow: 0 18px 40px -18px rgba(79,70,229,.65);}
.hero:before {content:""; position:absolute; right:-70px; top:-90px; width:300px; height:300px; border-radius:50%;
              background: rgba(255,255,255,.12);}
.hero:after {content:""; position:absolute; right:150px; bottom:-120px; width:240px; height:240px; border-radius:50%;
             background: rgba(255,255,255,.08);}
.hero h1 {margin:0; font-size:2.1rem; font-weight:800; letter-spacing:-.02em; color:#fff; padding:0;}
.hero p {margin:6px 0 16px 0; font-size:1.02rem; opacity:.92; color:#fff;}
.hero .tag {display:inline-block; margin:0 8px 6px 0; padding:6px 13px; border-radius:999px; font-size:.84rem; font-weight:600;
            background: rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.28); backdrop-filter: blur(6px); color:#fff;}
.hero .tag.ok {background: rgba(16,185,129,.30);} .hero .tag.warn {background: rgba(245,158,11,.34);}
.hero .tag.bad {background: rgba(244,63,94,.45);}

/* ---------- KPI-карточки ---------- */
.kpi {position:relative; background:#fff; border-radius:18px; padding:16px 18px 14px 18px; height:100%;
      box-shadow: 0 6px 22px -12px rgba(30,27,58,.28); border:1px solid #eceefa; overflow:hidden;}
.kpi:before {content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background: var(--accent,#4f46e5);}
.kpi .lbl {padding-right:30px; font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; color:#6b7090; font-weight:700;}
.kpi .val {font-size:1.75rem; font-weight:800; letter-spacing:-.02em; color:#1e1b3a; margin-top:4px; line-height:1.15;}
.kpi .sub {font-size:.82rem; color:#6b7090; margin-top:4px;}
.kpi .ico {position:absolute; right:14px; top:12px; font-size:1.5rem; opacity:.9;}
.kpi.pos {--accent:#10b981;} .kpi.pos .val {color:#047857;}
.kpi.neg {--accent:#f43f5e;} .kpi.neg .val {color:#be123c;}
.kpi.ind {--accent:#4f46e5;} .kpi.amb {--accent:#f59e0b;} .kpi.sky {--accent:#38bdf8;}

/* ---------- Заголовки секций ---------- */
h2, h3 {letter-spacing:-.01em; color:#1e1b3a;}
h3 {font-weight:700; margin-top:.6rem;}
h5 {color:#3b3a6b;}

/* ---------- Вкладки ---------- */
div[data-baseweb="tab-list"] {gap:6px; background:#fff; padding:6px; border-radius:16px; border:1px solid #eceefa;
                              box-shadow: 0 6px 20px -14px rgba(30,27,58,.35); flex-wrap: wrap;}
button[data-baseweb="tab"] {background:transparent; border:0; border-radius:11px; padding:8px 14px; height:auto; font-weight:600; color:#4a4d78;}
button[data-baseweb="tab"]:hover {background:#f0f1ff;}
button[data-baseweb="tab"][aria-selected="true"] {background: linear-gradient(120deg,#4f46e5,#7c3aed); color:#fff;}
button[data-baseweb="tab"][aria-selected="true"] p {color:#fff;}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display:none;}

/* ---------- Графики и таблицы как карточки ---------- */
div[data-testid="stPlotlyChart"] {background:#fff; border-radius:18px; padding:8px 10px; border:1px solid #eceefa;
                                  box-shadow: 0 6px 22px -14px rgba(30,27,58,.30);}
div[data-testid="stDataFrame"] {border-radius:14px; overflow:hidden; border:1px solid #eceefa;
                                box-shadow: 0 4px 16px -12px rgba(30,27,58,.35);}
div[data-testid="stMetric"] {background:#fff; padding:12px 16px; border-radius:16px; border:1px solid #eceefa;
                             box-shadow: 0 4px 16px -12px rgba(30,27,58,.35);}
div[data-testid="stExpander"] {background:#fff; border-radius:14px; border:1px solid #eceefa;}
div[data-testid="stChatMessage"] {background:#fff; border-radius:16px; border:1px solid #eceefa; padding:12px 16px;
                                  box-shadow: 0 4px 16px -12px rgba(30,27,58,.35);}
div[data-testid="stAlert"] {border-radius:14px;}

/* ---------- Кнопки ---------- */
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius:12px; font-weight:600; border:1px solid #d9dcf5; background:#fff; color:#3b3a8f; transition: all .15s ease;}
.stButton > button:hover, .stDownloadButton > button:hover {border-color:#6d28d9; color:#6d28d9; transform: translateY(-1px);
    box-shadow: 0 8px 18px -10px rgba(109,40,217,.6);}
.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(120deg,#4f46e5,#7c3aed); color:#fff; border:0;}
.stButton > button[kind="primary"]:hover {color:#fff;}

/* ---------- Боковая панель ---------- */
section[data-testid="stSidebar"] {background: linear-gradient(180deg,#eef0ff 0%,#f7ecff 100%); border-right:1px solid #e1e3fb;}
section[data-testid="stSidebar"] h2 {color:#3730a3;}

/* ---------- Плашки статуса ---------- */
.pill {display:inline-block; padding:4px 12px; border-radius:999px; font-size:.85rem; font-weight:600; margin-right:6px;}
.pill-ok {background:#d1fae5; color:#065f46;} .pill-warn {background:#fef3c7; color:#92400e;}
.pill-bad {background:#ffe4e6; color:#9f1239;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, tags: list[tuple[str, str]]) -> None:
    """tags: [(текст, 'ok'|'warn'|'bad'|'')]"""
    chips = ''.join(f'<span class="tag {k}">{html.escape(t)}</span>' for t, k in tags)
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>{chips}</div>',
                unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = '', tone: str = 'ind', icon: str = '') -> None:
    """Карточка показателя в колонке `col`. tone: pos | neg | ind | amb | sky."""
    col.markdown(
        f'<div class="kpi {tone}"><span class="ico">{icon}</span><div class="lbl">{html.escape(label)}</div>'
        f'<div class="val">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>',
        unsafe_allow_html=True)
