"""Profitability AI — интерфейс Streamlit.

Запуск:  streamlit run app.py
Все суммы считает engine.py; ИИ-аналитик (ai_analyst.py) только объясняет.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import engine as eng
from ai_analyst import EXAMPLES, Analyst, money, pct, tg
from theme import C_DIRECT, C_FOT, C_OVER, C_REV, NEG, NEUTRAL, PLOT_CFG, POS, hero, inject_css, kpi

st.set_page_config(page_title='Profitability AI', page_icon='📊', layout='wide')

DEMO_FILE = Path(__file__).parent / 'data' / 'Profitability_AI_Test_Data.xlsx'
inject_css()


# --------------------------------------------------------------------------
# Вспомогательные функции (совместимость со старыми/новыми версиями Streamlit)
# --------------------------------------------------------------------------
def show_df(df, **kw):
    try:
        st.dataframe(df, width='stretch', hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def show_plot(fig, key=None):
    fig.update_layout(**PLOT_CFG)
    fig.update_xaxes(gridcolor='#e8eaf3', zerolinecolor='#c9cde6')
    fig.update_yaxes(gridcolor='#e8eaf3', zerolinecolor='#c9cde6')
    try:
        st.plotly_chart(fig, width='stretch', key=key)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key)


def spaced(x, digits=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'н/д'
    return f'{x:,.{digits}f}'.replace(',', ' ').replace('-', '−')


def style_money(df, money_cols=(), pct_cols=(), digits=2):
    fm = {c: (lambda v: spaced(v, digits)) for c in money_cols if c in df.columns}
    fm.update({c: (lambda v: 'н/д' if pd.isna(v) else f'{v:.2f}%'.replace('-', '−')) for c in pct_cols if c in df.columns})
    return df.style.format(fm, na_rep='н/д')


def pill(text, kind='ok'):
    return f'<span class="pill pill-{kind}">{text}</span>'


def to_excel_bytes(result, controls, anomalies, warnings_df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xw:
        result.to_excel(xw, sheet_name='P&L по ОП', index=False)
        controls.to_excel(xw, sheet_name='Контрольные суммы', index=False)
        anomalies.drop(columns=['код'], errors='ignore').to_excel(xw, sheet_name='Аномалии', index=False)
        warnings_df.to_excel(xw, sheet_name='Проверки данных', index=False)
    return buf.getvalue()


@st.cache_data(show_spinner='Читаю Excel и считаю P&L…')
def process(file_bytes: bytes):
    data = eng.load_workbook(io.BytesIO(file_bytes))
    result, details = eng.calculate_details(data)
    controls = eng.control_checks(data, result, details)
    anomalies = eng.detect_signal_anomalies(data)
    warnings_df = eng.detect_core_warnings(data, result)
    # Проверка изоляции: P&L без листа «Сырые_сигналы» должен быть идентичен.
    stripped = dict(data)
    stripped['Сырые_сигналы'] = data['Сырые_сигналы'].iloc[0:0]
    r2 = eng.calculate_financials(stripped)
    isolation_diff = float((r2['финансовый_результат'] - result['финансовый_результат']).abs().max())
    return data, result, details, controls, anomalies, warnings_df, isolation_diff


# --------------------------------------------------------------------------
# Боковая панель: данные, фильтры, настройки ИИ
# --------------------------------------------------------------------------
with st.sidebar:
    st.header('📊 Profitability AI')
    st.caption('Анализ рентабельности образовательных программ · 2025–2026')
    uploaded = st.file_uploader('Excel с данными (.xlsx)', type=['xlsx'])
    use_demo = False
    if DEMO_FILE.exists():
        use_demo = st.checkbox('Использовать встроенный демо-файл', value=uploaded is None,
                               help='Тот же Profitability_AI_Test_Data.xlsx, без ручной правки.')

if uploaded is not None:
    file_bytes, source_name = uploaded.getvalue(), uploaded.name
elif use_demo and DEMO_FILE.exists():
    file_bytes, source_name = DEMO_FILE.read_bytes(), DEMO_FILE.name
else:
    st.title('📊 Profitability AI')
    st.info('Загрузите Profitability_AI_Test_Data.xlsx в боковой панели, чтобы начать анализ.')
    st.stop()

try:
    data, result, details, controls, anomalies, core_warn, isolation_diff = process(file_bytes)
except eng.DataError as exc:
    st.error(f'Файл не прошёл проверку структуры: {exc}')
    st.stop()
except Exception as exc:  # неожиданный формат
    st.error(f'Не удалось обработать файл: {exc}')
    st.stop()

with st.sidebar:
    st.success(f'Загружено: {source_name}')
    st.markdown('**Фильтры**')
    faculties = sorted(result['факультет'].dropna().unique().tolist())
    sel_fac = st.multiselect('Факультет', faculties, default=faculties)
    profit_filter = st.radio('Рентабельность', ['Все', 'Прибыльные', 'Убыточные'], horizontal=True)
    with st.expander('🤖 Настройки ИИ-аналитика'):
        st.caption('Без ключа работает локальный режим (формулы + шаблоны). С ключом Claude лишь '
                   'переформулирует ответ, а каждое число проверяется кодом.')
        api_key = st.text_input('API key: Anthropic (sk-ant-…) или OpenRouter (sk-or-…), необязательно', type='password',
                                value=os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('OPENROUTER_API_KEY', ''))
        use_llm = st.checkbox('Использовать LLM для формулировок', value=False, disabled=not api_key)
    st.download_button('⬇ Скачать результаты (Excel)', to_excel_bytes(result, controls, anomalies, core_warn),
                       file_name='profitability_results.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

view = result[result['факультет'].isin(sel_fac)]
if profit_filter == 'Прибыльные':
    view = view[view['финансовый_результат'] >= 0]
elif profit_filter == 'Убыточные':
    view = view[view['финансовый_результат'] < 0]
view = view.sort_values('финансовый_результат', ascending=False)

analyst = Analyst(data, result, details, anomalies, controls)
T = eng.totals(result)
names = result.set_index('op_id')['название_ОП'].to_dict()

# --------------------------------------------------------------------------
# Шапка
# --------------------------------------------------------------------------
hero('Profitability AI', 'Интеллектуальный анализ рентабельности образовательных программ · учебный год 2025–2026',
     [('✔ Контрольные суммы сходятся', 'ok') if controls['OK'].all() else ('✖ Контрольные суммы НЕ сходятся', 'bad'),
      (f'⚠ {len(anomalies)} аномалий в «Сырые_сигналы» — не влияют на P&L', 'warn') if len(anomalies)
      else ('✔ Аномалий в сигналах нет', 'ok'),
      (f'{len(result)} образовательных программ', ''), (f'Файл: {source_name}', '')])

tab_dash, tab_op, tab_dq, tab_ai, tab_wi, tab_fc, tab_doc = st.tabs(
    ['📈 Дашборд', '🎓 Образовательные программы', '🧪 Data Quality', '🤖 AI-аналитик',
     '🔮 What-If', '📅 Прогноз набора', '📘 Методика'])

# ==========================================================================
# 1. Дашборд
# ==========================================================================
with tab_dash:
    if view.empty:
        st.warning('По выбранным фильтрам нет программ.')
    else:
        vt = eng.totals(view)
        scope = 'по всем ОП' if len(view) == len(result) else f'по выбранным ОП ({len(view)} из {len(result)})'
        st.caption(f'Итоги {scope}. Накладные всегда распределяются между всеми 12 ОП — фильтр меняет только отображение.')
        c1, c2, c3, c4, c5 = st.columns(5)
        _m = lambda v: f"{v / 1e6:,.1f} млн ₸".replace(',', ' ').replace('-', '−')
        kpi(c1, 'Доход', _m(vt['доход']), f"{vt['студентов']:,.0f} студентов".replace(',', ' '), 'ind', '💰')
        kpi(c2, 'Общие затраты', _m(vt['затраты']), f"ФОТ {vt['фот'] / 1e6:,.0f} · прям. {vt['прямые'] / 1e6:,.0f} · накл. {vt['накладные'] / 1e6:,.0f} млн", 'amb', '🧾')
        kpi(c3, 'Финансовый результат', _m(vt['результат']), 'доход − затраты', 'pos' if vt['результат'] >= 0 else 'neg', '📈' if vt['результат'] >= 0 else '📉')
        kpi(c4, 'Рентабельность', pct(vt['рентабельность_pct']), 'результат / доход', 'pos' if vt['рентабельность_pct'] >= 0 else 'neg', '🎯')
        kpi(c5, 'Убыточных ОП', f"{vt['убыточных']} из {len(view)}", 'финансовый результат < 0', 'neg' if vt['убыточных'] else 'pos', '⚠️' if vt['убыточных'] else '✅')
        st.write('')

        left, right = st.columns(2)
        with left:
            d = view.sort_values('финансовый_результат')
            fig = go.Figure(go.Bar(
                x=d['финансовый_результат'] / 1e6, y=d['название_ОП'], orientation='h',
                marker_color=[POS if v >= 0 else NEG for v in d['финансовый_результат']],
                text=[f'{v / 1e6:,.1f}'.replace(',', ' ') for v in d['финансовый_результат']], textposition='outside',
                hovertemplate='%{y}<br>Результат: %{x:,.2f} млн ₸<extra></extra>'))
            fig.update_layout(title='Финансовый результат по ОП, млн ₸', height=460, xaxis_title=None, yaxis_title=None)
            show_plot(fig, 'dash_profit')
        with right:
            d = view.sort_values('рентабельность_pct')
            fig = go.Figure(go.Bar(
                x=d['рентабельность_pct'], y=d['название_ОП'], orientation='h',
                marker_color=[POS if v >= 0 else NEG for v in d['рентабельность_pct']],
                text=[f'{v:.1f}%' for v in d['рентабельность_pct']], textposition='outside',
                hovertemplate='%{y}<br>Рентабельность: %{x:.2f}%<extra></extra>'))
            fig.update_layout(title='Рентабельность по ОП, %', height=460, xaxis_title=None, yaxis_title=None)
            show_plot(fig, 'dash_margin')

        left, right = st.columns(2)
        with left:
            d = view.sort_values('итого_затраты', ascending=False)
            fig = go.Figure()
            for col, name, colr in (('фот_оп', 'ФОТ', C_FOT), ('прямые_затраты', 'Прямые', C_DIRECT), ('накладные', 'Накладные', C_OVER)):
                fig.add_bar(x=d['название_ОП'], y=d[col] / 1e6, name=name, marker_color=colr)
            fig.add_scatter(x=d['название_ОП'], y=d['доход'] / 1e6, mode='markers', name='Доход',
                            marker=dict(symbol='diamond', size=11, color=C_REV, line=dict(width=1, color='white')))
            fig.update_layout(barmode='stack', title='Структура затрат и доход, млн ₸', height=430,
                              xaxis_tickangle=-35, legend=dict(orientation='h', y=1.12))
            show_plot(fig, 'dash_struct')
        with right:
            fig = go.Figure(go.Waterfall(
                x=['Доход', 'ФОТ', 'Прямые затраты', 'Накладные', 'Результат'],
                measure=['absolute', 'relative', 'relative', 'relative', 'total'],
                y=[vt['доход'] / 1e6, -vt['фот'] / 1e6, -vt['прямые'] / 1e6, -vt['накладные'] / 1e6, 0],
                increasing=dict(marker=dict(color=POS)), decreasing=dict(marker=dict(color=NEG)), totals=dict(marker=dict(color=C_FOT)),
                texttemplate='%{y:,.1f}', connector=dict(line=dict(color='#aab')), ))
            fig.update_layout(title='Как доход превращается в результат, млн ₸', height=430)
            show_plot(fig, 'dash_waterfall')

        left, right = st.columns(2)
        with left:
            fig = px.scatter(view, x='студентов_активных', y='себестоимость_на_студента', size='доход',
                             color=np.where(view['финансовый_результат'] >= 0, 'Прибыльная', 'Убыточная'),
                             color_discrete_map={'Прибыльная': POS, 'Убыточная': NEG}, hover_name='название_ОП',
                             labels={'студентов_активных': 'Студентов', 'себестоимость_на_студента': 'Себестоимость студента, ₸',
                                     'color': ''}, text='op_id')
            fig.update_traces(textposition='top center')
            fig.update_layout(title='Масштаб vs себестоимость (размер = доход)', height=430)
            show_plot(fig, 'dash_scatter')
        with right:
            fg = (view.groupby('факультет', as_index=False)
                  .agg(доход=('доход', 'sum'), затраты=('итого_затраты', 'sum'), результат=('финансовый_результат', 'sum'),
                       ОП=('op_id', 'count')))
            fg['рентабельность_pct'] = fg['результат'] / fg['доход'] * 100
            fig = go.Figure()
            fig.add_bar(x=fg['факультет'], y=fg['доход'] / 1e6, name='Доход', marker_color=C_REV)
            fig.add_bar(x=fg['факультет'], y=fg['затраты'] / 1e6, name='Затраты', marker_color=C_FOT)
            fig.add_bar(x=fg['факультет'], y=fg['результат'] / 1e6, name='Результат',
                        marker_color=[POS if v >= 0 else NEG for v in fg['результат']])
            fig.update_layout(barmode='group', title='Сравнение факультетов, млн ₸', height=430, xaxis_tickangle=-20,
                              legend=dict(orientation='h', y=1.12))
            show_plot(fig, 'dash_fac')

        st.subheader('Результаты по образовательным программам')
        cols = ['op_id', 'название_ОП', 'факультет', 'студентов_активных', 'доход', 'фот_оп', 'прямые_затраты', 'накладные',
                'итого_затраты', 'финансовый_результат', 'рентабельность_pct', 'себестоимость_на_студента']
        tbl = view[cols].rename(columns={'студентов_активных': 'студентов'}).reset_index(drop=True)
        tbl.loc[len(tbl)] = ['', 'ИТОГО', '', vt['студентов'], vt['доход'], vt['фот'], vt['прямые'], vt['накладные'],
                             vt['затраты'], vt['результат'], vt['рентабельность_pct'], vt['себестоимость_на_студента']]
        show_df(style_money(tbl, ['доход', 'фот_оп', 'прямые_затраты', 'накладные', 'итого_затраты', 'финансовый_результат',
                                  'себестоимость_на_студента'], ['рентабельность_pct']))
        st.subheader('Факультеты')
        show_df(style_money(fg, ['доход', 'затраты', 'результат'], ['рентабельность_pct']))

# ==========================================================================
# 2. Образовательные программы
# ==========================================================================
with tab_op:
    op_id = st.selectbox('Образовательная программа', list(names), format_func=lambda o: f'{o} · {names[o]}', key='op_sel')
    x = result.set_index('op_id').loc[op_id]
    st.markdown(f"**{x['название_ОП']}** · {x['факультет']} · {x.get('уровень', '')} · срок {x.get('срок_обучения_лет', '')} г.")
    m1, m2, m3, m4, m5 = st.columns(5)
    _m = lambda v: f"{v / 1e6:,.2f} млн ₸".replace(',', ' ').replace('-', '−')
    kpi(m1, 'Доход', _m(x['доход']), f"{x['студентов_активных']:.0f} студентов", 'ind', '💰')
    kpi(m2, 'Затраты', _m(x['итого_затраты']), 'ФОТ + прямые + накладные', 'amb', '🧾')
    kpi(m3, 'Результат', _m(x['финансовый_результат']), 'доход − затраты', 'pos' if x['финансовый_результат'] >= 0 else 'neg',
        '📈' if x['финансовый_результат'] >= 0 else '📉')
    kpi(m4, 'Рентабельность', pct(x['рентабельность_pct']), 'результат / доход', 'pos' if x['рентабельность_pct'] >= 0 else 'neg', '🎯')
    kpi(m5, 'Себестоимость студента', tg(x['себестоимость_на_студента']), 'затраты / студенты', 'sky', '🎓')
    st.write('')

    left, right = st.columns([1, 1])
    with left:
        den = x['доход'] if x['доход'] else np.nan
        pl = pd.DataFrame({
            'Статья': ['Доход', '− ФОТ', '− Прямые затраты', '− Накладные', '= Итого затраты', '= Финансовый результат'],
            'Сумма, ₸': [x['доход'], -x['фот_оп'], -x['прямые_затраты'], -x['накладные'], -x['итого_затраты'], x['финансовый_результат']],
            'Доля дохода, %': [100.0, x['фот_оп'] / den * 100, x['прямые_затраты'] / den * 100, x['накладные'] / den * 100,
                               x['итого_затраты'] / den * 100, x['рентабельность_pct']]})
        st.markdown('##### P&L программы')
        show_df(style_money(pl, ['Сумма, ₸'], ['Доля дохода, %']))
        fig = go.Figure(go.Waterfall(
            x=['Доход', 'ФОТ', 'Прямые', 'Накладные', 'Результат'], measure=['absolute', 'relative', 'relative', 'relative', 'total'],
            y=[x['доход'] / 1e6, -x['фот_оп'] / 1e6, -x['прямые_затраты'] / 1e6, -x['накладные'] / 1e6, 0],
            increasing=dict(marker=dict(color=POS)), decreasing=dict(marker=dict(color=NEG)), totals=dict(marker=dict(color=C_FOT)),
            texttemplate='%{y:,.1f}'))
        fig.update_layout(title='Доход → результат, млн ₸', height=330)
        show_plot(fig, 'op_wf')
    with right:
        T_ = T
        cmp_df = pd.DataFrame({
            'Показатель на студента, ₸': ['Доход', 'ФОТ', 'Прямые затраты', 'Накладные', 'Себестоимость'],
            'ОП': [x['доход'] / x['студентов_активных'], x['фот_оп'] / x['студентов_активных'],
                   x['прямые_затраты'] / x['студентов_активных'], x['накладные'] / x['студентов_активных'],
                   x['себестоимость_на_студента']],
            'Среднее по вузу': [T_['доход'] / T_['студентов'], T_['фот'] / T_['студентов'], T_['прямые'] / T_['студентов'],
                                T_['накладные'] / T_['студентов'], T_['себестоимость_на_студента']]})
        cmp_df['Отклонение, %'] = (cmp_df['ОП'] / cmp_df['Среднее по вузу'] - 1) * 100
        st.markdown('##### Сравнение со средним по вузу')
        show_df(style_money(cmp_df, ['ОП', 'Среднее по вузу'], ['Отклонение, %']))
        fig = go.Figure()
        fig.add_bar(x=cmp_df['Показатель на студента, ₸'], y=cmp_df['ОП'], name=x['название_ОП'], marker_color=C_FOT)
        fig.add_bar(x=cmp_df['Показатель на студента, ₸'], y=cmp_df['Среднее по вузу'], name='Среднее по вузу', marker_color='#b8c0cc')
        fig.update_layout(barmode='group', title='На одного студента, ₸', height=330, legend=dict(orientation='h', y=1.15))
        show_plot(fig, 'op_cmp')

    st.markdown('##### Детализация расчёта (проверяемые строки)')
    t1, t2, t3, t4 = st.tabs(['Доход по курсам', 'ФОТ по преподавателям', 'Накладные по центрам', 'Прямые затраты'])
    with t1:
        rv = details['revenue']
        rv = rv[rv['op_id'] == op_id][['курс', 'контракт_полный', 'контракт_со_скидкой', 'грант', 'тариф_контракт_тг',
                                      'скидка_доля', 'финансирование_грант_тг', 'доход_контракт_полный',
                                      'доход_контракт_скидка', 'доход_грант', 'доход']]
        show_df(style_money(rv, ['тариф_контракт_тг', 'финансирование_грант_тг', 'доход_контракт_полный', 'доход_контракт_скидка',
                                 'доход_грант', 'доход'], digits=0))
        st.caption(f'Σ дохода = {tg(rv["доход"].sum())}. Формула: полный × тариф + со скидкой × тариф × (1 − скидка) + грант × финансирование.')
    with t2:
        pw = details['payroll']
        pw = pw[pw['op_id'] == op_id][['teacher_id', 'часы_в_год', 'всего_часов_преподавателя', 'доля_часов', 'годовой_фот', 'фот_оп']]
        pw = pw.assign(доля_часов=pw['доля_часов'] * 100).rename(columns={'доля_часов': 'доля часов, %'})
        show_df(style_money(pw.sort_values('фот_оп', ascending=False), ['годовой_фот', 'фот_оп'], ['доля часов, %'], digits=0))
        st.caption(f'Σ ФОТ ОП = {tg(pw["фот_оп"].sum())}. ФОТ ОП = годовой ФОТ преподавателя × часы на ОП / все часы преподавателя.')
    with t3:
        oh = details['overhead']
        oh = oh[oh['op_id'] == op_id][['cost_center_id', 'статья', 'драйвер', 'сумма_центра', 'значение_драйвера',
                                       'сумма_драйверов', 'доля', 'накладные']]
        oh = oh.assign(доля=oh['доля'] * 100).rename(columns={'доля': 'доля ОП в драйвере, %'})
        show_df(style_money(oh, ['сумма_центра', 'значение_драйвера', 'сумма_драйверов', 'накладные'], ['доля ОП в драйвере, %'], digits=0))
        st.caption(f'Σ накладных ОП = {tg(oh["накладные"].sum())}. Накладные = сумма центра × драйвер ОП / Σ драйвера 12 ОП.')
    with t4:
        dd = details['direct']
        dd = dd[dd['op_id'] == op_id]
        show_df(style_money(dd, ['сумма_в_год_тг'], digits=0))
        st.caption(f'Σ прямых затрат = {tg(dd["сумма_в_год_тг"].sum())}.')

    if st.button('🤖 Объяснить ОП', type='primary', key='explain_btn'):
        ans = analyst.a_explain(op_id)
        st.markdown(ans.text)
        with st.expander('Доказательства: разложение, формулы, источники'):
            show_df(style_money(ans.evidence, ['ОП, ₸/студента', 'Среднее по вузу, ₸/студента', 'Вклад в результат, ₸'], digits=0))
            st.markdown('**Формулы**\n' + '\n'.join(f'- {f}' for f in ans.formulas))
            st.markdown('**Источники**\n' + '\n'.join(f'- {s}' for s in ans.sources))
            st.markdown('**Допущения**\n' + '\n'.join(f'- {a}' for a in ans.assumptions))

# ==========================================================================
# 3. Data Quality
# ==========================================================================
with tab_dq:
    st.subheader('Структура файла')
    st.success(f'Все {len(eng.REQUIRED_SHEETS)} обязательных листов и колонок найдены; числовые поля приведены к числам.')

    st.subheader('Аномалии в «Сырые_сигналы»')
    if anomalies.empty:
        st.success('Аномалий не найдено.')
    else:
        st.warning(f'Найдено **{len(anomalies)}** проблемных записей. Лист «Сырые_сигналы» полностью исключён из финансового расчёта.')
        show_df(anomalies.drop(columns=['код']))
        by_type = anomalies['Тип ошибки'].value_counts().reset_index()
        by_type.columns = ['Тип ошибки', 'Количество']
        show_df(by_type)
    if isolation_diff == 0:
        st.success('Проверка изоляции: P&L, посчитанный БЕЗ листа «Сырые_сигналы», идентичен основному (расхождение 0 ₸) — '
                   'проблемные записи не влияют на расчёт.')
    else:
        st.error(f'Расхождение P&L при исключении листа сигналов: {isolation_diff:,.2f} ₸')

    st.subheader('Контрольные суммы (приёмочные испытания)')
    ctl = controls.copy()
    ctl['OK'] = ctl['OK'].map({True: '✅', False: '❌'})
    show_df(style_money(ctl, ['Ожидается', 'Получено', 'Разница'], digits=2))
    st.caption(f'Допуск по суммам — {eng.TOLERANCE_TG:g} ₸. Все проверки пройдены: {"да" if controls["OK"].all() else "НЕТ"}.')

    st.subheader('Дополнительные проверки основных листов')
    show_df(core_warn)

# ==========================================================================
# 4. AI-аналитик
# ==========================================================================
with tab_ai:
    st.subheader('Спросите ИИ-аналитика')
    st.caption('Суммы считает код (engine.py), аналитик объясняет и показывает доказательства: формулу, лист-источник, допущения.')
    if 'chat' not in st.session_state:
        st.session_state.chat = []

    def render_answer(ans):
        st.markdown(ans.text)
        if ans.note:
            st.caption('ℹ ' + ans.note)
        if ans.evidence is not None or ans.formulas or ans.sources or ans.assumptions:
            with st.expander('📎 Доказательства: таблица, формулы, источники, допущения'):
                if ans.evidence is not None:
                    money_like = [c for c in ans.evidence.columns if ans.evidence[c].dtype.kind == 'f' and 'pct' not in c and '%' not in c and 'рентаб' not in c.lower()]
                    pct_like = [c for c in ans.evidence.columns if 'pct' in c or '%' in c or 'рентаб' in c.lower()]
                    show_df(style_money(ans.evidence, money_like, pct_like, digits=2))
                if ans.formulas:
                    st.markdown('**Формулы**\n' + '\n'.join(f'- {f}' for f in ans.formulas))
                if ans.sources:
                    st.markdown('**Источники данных**\n' + '\n'.join(f'- {s}' for s in ans.sources))
                if ans.assumptions:
                    st.markdown('**Допущения**\n' + '\n'.join(f'- {a}' for a in ans.assumptions))
        st.caption(f'Режим: {ans.mode}')

    cols = st.columns(4)
    pending = None
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 4].button(ex, key=f'ex_{i}'):
            pending = ex
    c_a, c_b = st.columns([3, 1])
    with c_a:
        explain_op = st.selectbox('Объяснить ОП', list(names), format_func=lambda o: f'{o} · {names[o]}', key='ai_op')
    with c_b:
        st.write('')
        st.write('')
        if st.button('Объяснить выбранную ОП', key='ai_explain'):
            pending = f'Объясни {names[explain_op]} ({explain_op})'

    for msg in st.session_state.chat:
        with st.chat_message(msg['role']):
            if msg['role'] == 'user':
                st.markdown(msg['content'])
            else:
                render_answer(msg['answer'])

    with st.form('ask_form', clear_on_submit=True):
        typed = st.text_input('Ваш вопрос', placeholder='Например: Почему у прикладной математики такой результат?')
        asked = st.form_submit_button('Спросить')
    question = pending or (typed.strip() if asked and typed.strip() else None)
    if question:
        st.session_state.chat.append({'role': 'user', 'content': question})
        with st.chat_message('user'):
            st.markdown(question)
        with st.chat_message('assistant'):
            ans = analyst.answer(question, llm=bool(use_llm and api_key), api_key=api_key or None)
            render_answer(ans)
        st.session_state.chat.append({'role': 'assistant', 'answer': ans})
    if st.session_state.chat and st.button('Очистить диалог'):
        st.session_state.chat = []
        st.rerun()

# ==========================================================================
# 5. What-If
# ==========================================================================
with tab_wi:
    st.subheader('Сценарий: изменение контингента')
    st.caption('Базовый сценарий кейса: доход пропорционален студентам; ФОТ и прямые затраты неизменны; общая сумма накладных '
               'неизменна, но центры с драйвером students перераспределяются по обновлённому контингенту.')
    default_ops = [o for o in result.sort_values('финансовый_результат')['op_id'].head(1)]
    wi_ops = st.multiselect('ОП для изменения', list(names), default=default_ops, format_func=lambda o: f'{o} · {names[o]}')
    st.session_state.setdefault('wi_pct', 10)

    def _set_pct(v):
        st.session_state['wi_pct'] = v

    b1, b2, b3, _ = st.columns([1, 1, 1, 5])
    b1.button('+10%', on_click=_set_pct, args=(10,))
    b2.button('−10%', on_click=_set_pct, args=(-10,))
    b3.button('0%', on_click=_set_pct, args=(0,))
    wi_pct = st.slider('Изменение числа студентов, %', -50, 50, key='wi_pct')
    with st.expander('⚙ Допущения (по умолчанию — базовый вариант кейса)'):
        grant_follows = st.checkbox('Грантовые студенты тоже меняются', value=True,
                                    help='Если выключить: грантовые места фиксированы госзаказом, меняются только платные.')
        pv = st.slider('Доля ФОТ, меняющаяся вместе с контингентом, %', 0, 100, 0)
        dv = st.slider('Доля прямых затрат, меняющаяся вместе с контингентом, %', 0, 100, 0)
        tariff = st.slider('Изменение тарифа контракта, %', -20, 30, 0)

    if not wi_ops:
        st.info('Выберите хотя бы одну ОП.')
    else:
        changes = {o: wi_pct / 100 for o in wi_ops}
        tchg = {o: tariff / 100 for o in wi_ops} if tariff else None
        sim = eng.simulate_enrollment(data, changes, grant_follows=grant_follows, payroll_var_share=pv / 100,
                                      direct_var_share=dv / 100, tariff_change=tchg)
        tb, ts = sim['totals_base'], sim['totals_scenario']
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Доход вуза', f"{ts['доход'] / 1e6:,.1f} млн ₸".replace(',', ' '),
                  f"{(ts['доход'] - tb['доход']) / 1e6:+,.2f} млн".replace(',', ' '))
        k2.metric('Затраты вуза', f"{ts['затраты'] / 1e6:,.1f} млн ₸".replace(',', ' '),
                  f"{(ts['затраты'] - tb['затраты']) / 1e6:+,.2f} млн".replace(',', ' '), delta_color='inverse')
        k3.metric('Результат вуза', f"{ts['результат'] / 1e6:,.1f} млн ₸".replace(',', ' '),
                  f"{(ts['результат'] - tb['результат']) / 1e6:+,.2f} млн".replace(',', ' '))
        k4.metric('Рентабельность вуза', pct(ts['рентабельность_pct']), f"{ts['рентабельность_pct'] - tb['рентабельность_pct']:+.2f} п.п.")

        dlt = sim['delta']
        fig = go.Figure()
        fig.add_bar(x=dlt['название_ОП'], y=dlt['финансовый_результат_до'] / 1e6, name='До', marker_color='#b8c0cc')
        fig.add_bar(x=dlt['название_ОП'], y=dlt['финансовый_результат_после'] / 1e6, name='После', marker_color=C_FOT)
        fig.update_layout(barmode='group', title='Финансовый результат до / после, млн ₸', height=420,
                          xaxis_tickangle=-35, legend=dict(orientation='h', y=1.12))
        show_plot(fig, 'wi_cmp')

        sel = dlt[dlt['op_id'].isin(wi_ops)]
        rows = []
        for _, rr in sel.iterrows():
            for label, col in (('Студентов', 'студентов_активных'), ('Доход', 'доход'), ('ФОТ', 'фот_оп'),
                               ('Прямые затраты', 'прямые_затраты'), ('Накладные', 'накладные'),
                               ('Итого затраты', 'итого_затраты'), ('Финансовый результат', 'финансовый_результат'),
                               ('Рентабельность, %', 'рентабельность_pct'), ('Себестоимость студента', 'себестоимость_на_студента')):
                rows.append({'ОП': rr['название_ОП'], 'Показатель': label, 'До': rr[f'{col}_до'],
                             'После': rr[f'{col}_после'], 'Δ': rr[f'{col}_Δ']})
        st.markdown('##### Изменённые ОП')
        show_df(style_money(pd.DataFrame(rows), ['До', 'После', 'Δ'], digits=2))

        with st.expander('Как перераспределились накладные по всем 12 ОП'):
            ov = dlt[['op_id', 'название_ОП', 'накладные_до', 'накладные_после', 'накладные_Δ']].copy()
            show_df(style_money(ov, ['накладные_до', 'накладные_после', 'накладные_Δ'], digits=0))
            st.caption(f'Σ накладных до = {tg(sim["base"]["накладные"].sum())}; после = {tg(sim["scenario"]["накладные"].sum())} '
                       '(общая сумма не меняется).')

        if len(wi_ops) == 1:
            be = eng.breakeven_growth(data, wi_ops[0], grant_follows=False)
            if be == 0:
                st.success('ОП уже безубыточна.')
            elif be is None:
                st.info('Безубыточность недостижима ростом платного контингента в модели (при росте до +1000%).')
            else:
                st.info(f'**Точка безубыточности:** при неизменных ФОТ и прямых затратах «{names[wi_ops[0]]}» выходит в ноль '
                        f'при росте платных студентов на ≈ **{be:.0%}** (грантовые места не меняются). '
                        'Практически это означает, что одним набором проблему не решить.')
        st.markdown('##### Допущения сценария')
        for a in eng.scenario_assumptions(changes, names, grant_follows, pv / 100, dv / 100, tchg):
            st.markdown(f'- {a}')

# ==========================================================================
# 6. Прогноз набора
# ==========================================================================
with tab_fc:
    st.subheader('Прогноз набора на 2026–2027 (бонус)')
    st.warning('Всего **3 наблюдения** на ОП. Это один год изменения плюс один — для тренда, но не для оценки случайного разброса. '
               'Прогноз — ориентир, а не обещание.')
    fc = eng.forecast_enrollment(data)
    pick = st.selectbox('ОП', fc['op_id'].tolist(), format_func=lambda o: f'{o} · {names.get(o, o)}', key='fc_op')
    row = fc[fc['op_id'] == pick].iloc[0]
    hist = data['История_набора']
    hist = hist[hist['op_id'] == pick].sort_values('учебный_год')
    xs = hist['учебный_год'].astype(str).tolist()
    fig = go.Figure()
    fig.add_scatter(x=xs, y=hist['зачислено'], mode='lines+markers', name='Зачислено', line=dict(color=C_FOT))
    fig.add_scatter(x=xs, y=hist['план_набора'], mode='lines+markers', name='План набора', line=dict(color='#b8c0cc', dash='dot'))
    nxt = '2026–2027'
    fig.add_scatter(x=[nxt], y=[row['прогноз (тренд)']], mode='markers', name='Прогноз (тренд)',
                    marker=dict(size=12, color=C_OVER),
                    error_y=dict(type='data', symmetric=False, array=[row['верхняя граница'] - row['прогноз (тренд)']],
                                 arrayminus=[row['прогноз (тренд)'] - row['нижняя граница']], color=C_OVER))
    fig.update_layout(title=f'{names.get(pick, pick)}: история и прогноз зачисления', height=400, legend=dict(orientation='h', y=1.12))
    show_plot(fig, 'fc_plot')
    c1, c2, c3 = st.columns(3)
    c1.metric('Прогноз (тренд)', f"{row['прогноз (тренд)']:.0f}")
    c2.metric('Диапазон', f"{row['нижняя граница']:.0f} – {row['верхняя граница']:.0f}")
    c3.metric('Ширина диапазона', f"{row['ширина диапазона, %']:.0f}% от прогноза")
    show_df(style_money(fc.drop(columns=['наблюдений']), [], ['ширина диапазона, %', 'прирост YoY, %', 'выполнение плана (посл. год), %'], digits=0)
            .format({'прогноз (тренд)': '{:.1f}', 'нижняя граница': '{:.1f}', 'верхняя граница': '{:.1f}'}))
    st.markdown("""
**Почему трёх наблюдений недостаточно для точного прогноза**
- Линейный тренд по трём точкам имеет всего **1 степень свободы**: два параметра (наклон и сдвиг) подгоняются под три числа, поэтому статистическая неопределённость огромна (t-критерий 80% при df = 1 равен 3,08).
- Одна аномальная кампания, изменение правил приёма или демографическая волна смещают тренд сильнее, чем он «подтверждается» данными.
- Нет сезонности, макрофакторов, конверсии заявок в зачисление по каналам.
- Диапазон здесь = от минимума до максимума четырёх простых методов (последнее значение, среднее, тренд, средний YoY), расширенный 80%-интервалом регрессии. Для планирования используйте его границы, а не точку.
- Для реального внедрения нужны 8–10 лет данных, демография выпускников, ценовая политика и конкуренты.""")

# ==========================================================================
# 7. Методика
# ==========================================================================
with tab_doc:
    st.subheader('Методика, допущения и границы решения')
    st.markdown(r"""
**Формулы (учебная модель, тенге, один учебный год)**
- Доход ОП = Σ по курсам (контракт_полный × тариф + контракт_со_скидкой × тариф × (1 − скидка) + грант × финансирование_гранта)
- Годовой ФОТ преподавателя = зарплата × месяцев × (1 + начисления). Доля на ОП = ФОТ × часы на ОП / все часы преподавателя
- Прямые затраты ОП = Σ строк «Прямые_затраты» по op_id
- Накладные ОП = Σ по центрам (сумма центра × драйвер ОП / Σ драйвера 12 ОП); students ↔ студентов_активных, area_m2 ↔ площадь_м², lab_hours ↔ лабораторные_часы
- Итого затраты = ФОТ + прямые + накладные; Результат = Доход − Итого затраты
- Рентабельность = Результат / Доход × 100%; Себестоимость студента = Итого затраты / активные студенты (при 0 — «н/д»)

**Принципы**
- Все суммы считает проверяемый код (`engine.py`); ИИ не вычисляет деньги, а объясняет посчитанное и показывает источник и формулу.
- «Сырые_сигналы» — внешний набор для проверки качества; в P&L не используется (проверяется автоматически на вкладке Data Quality).
- Округление только на экране, внутри расчёта — полная точность.

**Границы решения**
- Финансовый результат ≠ кассовый остаток; налог на прибыль, амортизация, стипендии, инвестиции, исследования не моделируются.
- Распределение накладных по драйверам — упрощение; в реальности модель утверждают финансово-экономические службы университета.
- What-If линейный: нет порогов (новые группы/потоки), эластичности спроса и ограничений мощностей.
- Финансовый результат не должен быть единственным основанием для закрытия ОП: учитываются академические, кадровые, стратегические и социальные факторы.

**Что нужно для реального внедрения**
Фактические платежи и дебиторка по студентам, ведомости штатного расписания и фактической нагрузки, учёт затрат по центрам ответственности, реальные площади и график занятости помещений, история набора за 8–10 лет, утверждённая методика распределения накладных, роли и доступы, аудит расчётов.

Данные синтетические; персональные данные и ИИН не используются.""")
