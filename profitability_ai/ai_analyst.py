"""ИИ-аналитик Profitability AI.

Принцип прозрачности: ВСЕ суммы вычисляет детерминированный код (engine.py).
Аналитик лишь (1) понимает вопрос, (2) достаёт нужные цифры из расчёта,
(3) формулирует объяснение и рекомендации и (4) показывает доказательства:
таблицу, формулу, листы-источники и допущения.

Два режима:
    * «локальный» — работает всегда, без интернета и ключей: маршрутизатор
      вопросов + шаблонные объяснения поверх посчитанных цифр;
    * «LLM» (необязательный) — если задан ANTHROPIC_API_KEY, модель Claude лишь
      переформулирует уже готовый локальный ответ. Затем каждое число в её
      тексте проверяется на наличие в посчитанных данных; при расхождении
      показывается локальный ответ (защита от выдуманных цифр).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import engine as eng

DEFAULT_MODEL = os.environ.get('PROFITABILITY_AI_MODEL', 'claude-sonnet-5-5')

EXAMPLES = [
    'Какие ОП убыточны?',
    'Почему у прикладной математики такой результат?',
    'Где самый высокий ФОТ на студента?',
    'Какая ОП самая прибыльная?',
    'Какая себестоимость студента на общей медицине?',
    'Что если контингент на сестринском деле вырастет на 10%?',
    'Какие аномалии в сырых сигналах?',
    'Как считается рентабельность?',
]

SRC_REVENUE = 'Лист «Контингент» (контракт_полный, контракт_со_скидкой, грант, тарифы, скидка_доля)'
SRC_PAYROLL = 'Листы «Штат_ППС» (зарплата, месяцы, начисления) и «Нагрузка_ППС» (часы по ОП)'
SRC_DIRECT = 'Лист «Прямые_затраты» (сумма_в_год_тг по op_id)'
SRC_OVERHEAD = 'Листы «Общие_затраты» (центры затрат) и «Драйверы» (студенты, м², лаб. часы)'
SRC_SIGNALS = 'Лист «Сырые_сигналы» (в P&L не используется)'

FORMULAS = {
    'revenue': 'Доход = Σ(контракт_полный × тариф + контракт_со_скидкой × тариф × (1 − скидка) + грант × финансирование_гранта) по всем курсам ОП',
    'payroll': 'Годовой ФОТ преподавателя = зарплата × месяцев × (1 + начисления); доля на ОП = ФОТ × часы на ОП / все часы преподавателя',
    'direct': 'Прямые затраты ОП = Σ строк «Прямые_затраты» с данным op_id',
    'overhead': 'Накладные ОП = Σ по центрам (сумма центра × драйвер ОП / Σ драйвера всех 12 ОП)',
    'total': 'Итого затраты = ФОТ + прямые + накладные',
    'profit': 'Финансовый результат = Доход − Итого затраты',
    'margin': 'Рентабельность, % = Финансовый результат / Доход × 100',
    'unit': 'Себестоимость студента = Итого затраты / активные студенты (при 0 — «н/д»)',
}


# --------------------------------------------------------------------------
# Форматирование
# --------------------------------------------------------------------------
def tg(x, sign=False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'н/д'
    s = f'{abs(x):,.0f}'.replace(',', ' ')
    pre = '−' if x < 0 else ('+' if sign and x > 0 else '')
    return f'{pre}{s} ₸'


def mln(x, sign=False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'н/д'
    pre = '−' if x < 0 else ('+' if sign and x > 0 else '')
    return f'{pre}{abs(x) / 1e6:,.2f}'.replace(',', ' ') + ' млн ₸'


def money(x, sign=False) -> str:
    """«−42 368 719 ₸ (−42,37 млн ₸)» — точная сумма + краткая запись."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'н/д'
    return f'{tg(x, sign)} ({mln(x, sign).replace(".", ",")})'


def pct(x, sign=False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'н/д'
    body = f'{abs(x):.2f}'.replace('.', ',')
    pre = '−' if x < 0 else ('+' if sign else '')
    return f'{pre}{body}%'


# --------------------------------------------------------------------------
# Ответ аналитика
# --------------------------------------------------------------------------
@dataclass
class Answer:
    text: str
    evidence: pd.DataFrame | None = None
    sources: list = field(default_factory=list)
    formulas: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    mode: str = 'локальный (формулы + шаблоны)'
    note: str = ''
    kind: str = ''


def _norm(s: str) -> str:
    return str(s).lower().replace('ё', 'е').replace('−', '-').replace('–', '-')


def _words(s: str) -> list:
    return re.findall(r'[a-zа-я0-9]+', _norm(s))


def _stem_hit(token: str, qwords: list) -> bool:
    stem = token[:max(4, len(token) - 2)]
    return any(w.startswith(stem) and abs(len(w) - len(token)) <= 2 for w in qwords)


class Analyst:
    def __init__(self, data: dict, result: pd.DataFrame, details: dict,
                 anomalies: pd.DataFrame | None = None, controls: pd.DataFrame | None = None):
        self.data = data
        self.result = result.copy()
        self.details = details
        self.anomalies = anomalies if anomalies is not None else eng.detect_signal_anomalies(data)
        self.controls = controls if controls is not None else eng.control_checks(data, result, details)
        self.t = eng.totals(result)
        self.names = result.set_index('op_id')['название_ОП'].to_dict()
        self.r = result.set_index('op_id')

    # ------------------------------------------------------------------ поиск
    def find_programs(self, question: str) -> list:
        q = _norm(question)
        qw = _words(question)
        found = {}
        for m in re.finditer(r'(?:op|оп)[\s\-_]*0*(\d{1,3})', q):
            oid = f'OP-{int(m.group(1)):03d}'
            if oid in self.names:
                found[oid] = 100
        for oid, name in self.names.items():
            toks = [t for t in _words(name) if len(t) >= 4]
            hits = [t for t in toks if _stem_hit(t, qw)]
            if hits and (any(len(t) >= 7 for t in hits) or len(hits) == len(toks)):
                found[oid] = max(found.get(oid, 0), sum(len(t) for t in hits))
        return [o for o, _ in sorted(found.items(), key=lambda kv: -kv[1])]

    def find_faculty(self, question: str):
        qw = _words(question)
        for fac in self.result['факультет'].dropna().unique():
            first = _words(fac)[0]
            if len(first) >= 4 and _stem_hit(first, qw) and any(w.startswith('факульт') for w in qw):
                return fac
        return None

    # ------------------------------------------------------------- маршрутизатор
    def answer(self, question: str, llm: bool = False, api_key: str | None = None,
               model: str | None = None) -> Answer:
        base = self._route(question)
        if llm and (api_key or os.environ.get('ANTHROPIC_API_KEY')):
            return self._rephrase_with_llm(question, base, api_key, model or DEFAULT_MODEL)
        return base

    def _route(self, question: str) -> Answer:
        q = _norm(question)
        ops = self.find_programs(question)
        fac = self.find_faculty(question)
        has = lambda *pats: any(re.search(p, q) for p in pats)

        if has(r'аномал', r'ошибк', r'качеств', r'сырые', r'сигнал', r'дубл', r'проблемн\w* (данн|строк|запис)'):
            return self.a_anomalies()
        if has(r'что если', r'что будет', r'сценар', r'вырастет', r'снизится', r'увелич\w+ (контингент|набор|студент)',
               r'уменьш\w+ (контингент|набор|студент)', r'what.?if', r'\b[+\-]\s?\d+\s?%'):
            return self.a_scenario(question, ops)
        if has(r'формул', r'как (считается|считают|рассчит|определя)', r'методик', r'как посчит'):
            return self.a_formulas(q)
        if has(r'убыточ', r'убыток', r'в минус', r'отрицательн\w* (результат|рентабельн)', r'не покрыва', r'нерентабельн'):
            return self.a_losses()
        if has(r'почему', r'объясни', r'причин', r'из-за чего', r'что влияет', r'разбер\w+', r'проанализируй') and ops:
            return self.a_explain(ops[0])
        if has(r'как[аоие]\w* (оп|программ)\w* (самая )?(прибыльн|лучш)', r'самая прибыльн', r'самые прибыльн',
               r'прибыльн\w* (оп|программ)', r'топ'):
            return self.a_ranking('финансовый_результат', ascending=False)
        metric = self._metric(q)
        ranking_words = has(r'сам\w+', r'наибольш', r'наименьш', r'максим', r'миним', r'\bгде\b', r'топ\b',
                            r'рейтинг', r'выше всего', r'ниже всего', r'лучш', r'худш', r'высок', r'низк')
        if len(ops) >= 2 and has(r'сравни', r'против', r'\bи\b', r'или'):
            return self.a_compare(ops[:3])
        if metric and ops and not ranking_words:
            return self.a_op_metric(ops[0], metric)
        if metric:
            asc = has(r'наименьш', r'минимальн', r'низк', r'меньше всего', r'худш', r'ниже всего')
            return self.a_ranking(metric, ascending=asc)
        if ops:
            return self.a_op_summary(ops[0])
        if fac:
            return self.a_faculty(fac)
        if has(r'факультет'):
            return self.a_faculties()
        if has(r'итого', r'в целом', r'весь университет', r'общ\w+ (картин|результат|итог)', r'сводк', r'всего', r'университет'):
            return self.a_overview()
        return self.a_help()

    def _metric(self, q: str):
        table = [
            (r'фот\w* на (одного )?студент', 'фот_на_студента'),
            (r'себестоимост', 'себестоимость_на_студента'),
            (r'доход\w* на (одного )?студент', 'доход_на_студента'),
            (r'рентабельн|маржа|маржинальн', 'рентабельность_pct'),
            (r'накладн', 'накладные'),
            (r'прямы\w+ (затрат|расход)', 'прямые_затраты'),
            (r'\bфот\b|зарплат|оплат\w+ труда', 'фот_оп'),
            (r'прибыл|финансов\w+ результат|результат', 'финансовый_результат'),
            (r'доход|выручк', 'доход'),
            (r'затрат|расход', 'итого_затраты'),
        ]
        for pat, col in table:
            if re.search(pat, q):
                return col
        return None

    # ------------------------------------------------------------ готовые ответы
    METRIC_LABEL = {
        'фот_на_студента': ('ФОТ на студента', 'тг/студента', 'ФОТ ОП / активные студенты', [SRC_PAYROLL, 'Лист «Драйверы» (студентов_активных)'], ['payroll']),
        'себестоимость_на_студента': ('Себестоимость на студента', 'тг/студента', 'Итого затраты ОП / активные студенты', [SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD], ['total', 'unit']),
        'доход_на_студента': ('Доход на студента', 'тг/студента', 'Доход ОП / активные студенты', [SRC_REVENUE], ['revenue']),
        'рентабельность_pct': ('Рентабельность', '%', 'Финансовый результат / Доход × 100', [SRC_REVENUE, SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD], ['profit', 'margin']),
        'накладные': ('Накладные затраты', 'тг', 'Σ(сумма центра × драйвер ОП / Σ драйвера)', [SRC_OVERHEAD], ['overhead']),
        'прямые_затраты': ('Прямые затраты', 'тг', 'Σ строк «Прямые_затраты» по op_id', [SRC_DIRECT], ['direct']),
        'фот_оп': ('ФОТ', 'тг', 'Σ годовой ФОТ преподавателя × доля часов на ОП', [SRC_PAYROLL], ['payroll']),
        'финансовый_результат': ('Финансовый результат', 'тг', 'Доход − (ФОТ + прямые + накладные)', [SRC_REVENUE, SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD], ['profit']),
        'доход': ('Доход', 'тг', 'Σ(полный × тариф + со скидкой × тариф × (1−скидка) + грант × финансирование)', [SRC_REVENUE], ['revenue']),
        'итого_затраты': ('Итого затраты', 'тг', 'ФОТ + прямые + накладные', [SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD], ['total']),
    }

    def _fmt_metric(self, col, v):
        if col == 'рентабельность_pct':
            return pct(v)
        return tg(v)

    def a_ranking(self, col: str, ascending: bool = False, top: int = 12) -> Answer:
        label, unit, formula, sources, fkeys = self.METRIC_LABEL[col]
        df = self.result[['op_id', 'название_ОП', 'факультет', col]].dropna(subset=[col])
        df = df.sort_values(col, ascending=ascending).reset_index(drop=True)
        best = df.iloc[0]
        word = 'наименьшее' if ascending else 'наибольшее'
        avg_line = ''
        if col in ('фот_на_студента', 'себестоимость_на_студента', 'доход_на_студента'):
            num = {'фот_на_студента': self.t['фот'], 'себестоимость_на_студента': self.t['затраты'],
                   'доход_на_студента': self.t['доход']}[col]
            avg_line = f' Среднее по университету: {tg(num / self.t["студентов"])} (Σ / {self.t["студентов"]:,.0f} студентов).'.replace(',', ' ')
        text = (f'**{word.capitalize()} значение — «{best["название_ОП"]}» ({best["op_id"]}): '
                f'{self._fmt_metric(col, best[col])}** ({label.lower()}).{avg_line}\n\n'
                f'Ниже полный рейтинг всех {len(df)} ОП по показателю «{label}».')
        if col == 'фот_на_студента':
            top3 = df.head(3) if not ascending else df.tail(3)
            text += ('\n\nВысокий ФОТ на студента обычно означает малый контингент при фиксированном '
                     'составе преподавателей: ' + '; '.join(
                         f'{r["название_ОП"]} — {self.r.loc[r["op_id"], "студентов_активных"]:.0f} студ., '
                         f'ФОТ {tg(self.r.loc[r["op_id"], "фот_оп"])}' for _, r in top3.iterrows()) + '.')
        ev = df.copy()
        ev.columns = ['op_id', 'ОП', 'Факультет', label]
        return Answer(text, ev, sources, [FORMULAS[k] for k in fkeys], [], kind='ranking')

    def a_losses(self) -> Answer:
        r = self.result
        loss = r[r['финансовый_результат'] < 0].sort_values('финансовый_результат')
        thin = r[(r['финансовый_результат'] >= 0) & (r['рентабельность_pct'] < 10)]
        if loss.empty:
            text = 'Убыточных ОП нет: у всех 12 программ финансовый результат ≥ 0.'
        else:
            lines = [f'- **{x.название_ОП}** ({x.op_id}): результат {money(x.финансовый_результат)}, '
                     f'рентабельность {pct(x.рентабельность_pct)}, доход {tg(x.доход)}, затраты {tg(x.итого_затраты)}'
                     for x in loss.itertuples()]
            text = (f'Убыточны **{len(loss)} из {len(r)}** ОП (финансовый результат < 0), суммарный убыток '
                    f'{money(loss["финансовый_результат"].sum())}:\n' + '\n'.join(lines))
        if len(thin):
            text += ('\n\nНа грани (результат ≥ 0, но рентабельность < 10%): ' + '; '.join(
                f'{x.название_ОП} ({pct(x.рентабельность_pct)})' for x in thin.itertuples()) + '.')
        text += ('\n\n_Убыток — финансовый результат учебной модели, а не кассовый остаток. Решения о судьбе ОП нельзя '
                 'принимать только по нему: нужны академические, кадровые, стратегические и социальные основания._')
        ev = r[['op_id', 'название_ОП', 'доход', 'фот_оп', 'прямые_затраты', 'накладные', 'итого_затраты',
                'финансовый_результат', 'рентабельность_pct']].sort_values('финансовый_результат')
        return Answer(text, ev, [SRC_REVENUE, SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD],
                      [FORMULAS['total'], FORMULAS['profit'], FORMULAS['margin']], kind='losses')

    # ---- объяснение ОП ------------------------------------------------------
    def explain_factors(self, op_id: str) -> pd.DataFrame:
        """Точное разложение отклонения результата ОП от «среднего по вузу на студента».

        Результат ОП − студенты × (средний результат на студента) =
            Δдоход − ΔФОТ − Δпрямые − Δнакладные, где Δ = (показатель ОП на студента − среднее) × студенты ОП.
        Сумма вкладов равна отклонению тождественно (проверяется в тестах).
        """
        r = self.r.loc[op_id]
        n = r['студентов_активных']
        T = self.t
        avg = {'доход': T['доход'] / T['студентов'], 'фот': T['фот'] / T['студентов'],
               'прямые': T['прямые'] / T['студентов'], 'накладные': T['накладные'] / T['студентов']}
        own = {'доход': r['доход'] / n, 'фот': r['фот_оп'] / n, 'прямые': r['прямые_затраты'] / n,
               'накладные': r['накладные'] / n}
        rows = []
        for key, label, sign in (('доход', 'Доход на студента', +1), ('фот', 'ФОТ на студента', -1),
                                 ('прямые', 'Прямые затраты на студента', -1), ('накладные', 'Накладные на студента', -1)):
            contrib = sign * (own[key] - avg[key]) * n
            rows.append({'Фактор': label, 'ОП, ₸/студента': own[key], 'Среднее по вузу, ₸/студента': avg[key],
                         'Вклад в результат, ₸': contrib})
        return pd.DataFrame(rows)

    def _overhead_by_driver(self, op_id: str) -> dict:
        oh = self.details['overhead']
        g = oh.groupby(['op_id', 'драйвер'])['накладные'].sum().unstack(fill_value=0)
        return g.loc[op_id].to_dict() if op_id in g.index else {}

    def a_explain(self, op_id: str) -> Answer:
        r = self.r.loc[op_id]
        T = self.t
        n = r['студентов_активных']
        fac = self.explain_factors(op_id)
        avg_profit_pp = T['результат'] / T['студентов']
        deviation = r['финансовый_результат'] - n * avg_profit_pp
        fac_sorted = fac.reindex(fac['Вклад в результат, ₸'].abs().sort_values(ascending=False).index)
        lines = []
        shown = 0
        for _, f in fac_sorted.iterrows():
            c = f['Вклад в результат, ₸']
            if shown >= 5:
                break
            direction = 'повышает' if c > 0 else 'снижает'
            lines.append(f'{shown + 1}. **{f["Фактор"]}**: {tg(f["ОП, ₸/студента"])} против среднего '
                         f'{tg(f["Среднее по вузу, ₸/студента"])} → {direction} результат на {tg(abs(c))}.')
            shown += 1
        # дополнительные факторы
        rev_total = r['доход']
        disc_loss = float(self.details['revenue'].query('op_id == @op_id').eval(
            'контракт_со_скидкой * тариф_контракт_тг * скидка_доля').sum())
        grant_share = r['доход_грант'] / rev_total * 100 if rev_total else float('nan')
        univ_grant_share = self.result['доход_грант'].sum() / T['доход'] * 100
        hours_pp = r['часов_ппс'] / n if n else float('nan')
        univ_hours_pp = self.result['часов_ппс'].sum() / T['студентов']
        extras = [
            f'Структура дохода: грант даёт {pct(grant_share)} дохода ОП (по вузу — {pct(univ_grant_share)}); '
            f'скидки уменьшают доход на {tg(disc_loss)}.',
            f'Нагрузка ППС: {hours_pp:.1f} ч на студента (по вузу — {univ_hours_pp:.1f}); '
            f'преподавателей на ОП: {int(r["преподавателей"])}; часов в год: {r["часов_ппс"]:,.0f}.'.replace(',', ' '),
        ]
        by_driver = self._overhead_by_driver(op_id)
        ohd = ', '.join(f'{eng.DRIVER_RU.get(k, k)}: {tg(v)}' for k, v in by_driver.items())
        extras.append(f'Накладные по драйверам: {ohd}. Площадь ОП — {r["площадь_м2"]:.0f} м² '
                      f'({r["площадь_м2"] / n:.2f} м² на студента), лаб. часы — {r["лабораторные_часы"]:.0f}.')
        # управленческие варианты (считаются движком)
        options, opt_ev = self._options(op_id)
        verdict = ('убыточна' if r['финансовый_результат'] < 0 else
                   'прибыльна, но с низкой маржой' if r['рентабельность_pct'] < 10 else 'прибыльна')
        head = (f'**{r["название_ОП"]} ({op_id}) {verdict}**: доход {money(r["доход"])}, затраты {money(r["итого_затраты"])}, '
                f'результат {money(r["финансовый_результат"])}, рентабельность {pct(r["рентабельность_pct"])}, '
                f'себестоимость студента {tg(r["себестоимость_на_студента"])} при {n:.0f} студентах.\n\n'
                f'**Почему так — факторы** (разложение отклонения от среднего по вузу на студента; '
                f'сумма вкладов = {tg(deviation, True)} = результат ОП − {n:.0f} × средний результат на студента '
                f'{tg(avg_profit_pp)}):\n')
        text = (head + '\n'.join(lines) + '\n\n**Дополнительно:**\n' + '\n'.join(f'- {e}' for e in extras)
                + '\n\n**Управленческие варианты** (эффект посчитан движком, не ИИ):\n' + options
                + '\n\n_Финансовый результат — не повод закрывать программу сам по себе: учитывайте академические, кадровые, '
                  'стратегические и социальные основания._')
        ev = pd.concat([fac, pd.DataFrame([{'Фактор': 'ИТОГО отклонение от среднего', 'Вклад в результат, ₸': fac['Вклад в результат, ₸'].sum()}])],
                       ignore_index=True)
        return Answer(text, ev, [SRC_REVENUE, SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD],
                      [FORMULAS[k] for k in ('revenue', 'payroll', 'direct', 'overhead', 'profit', 'unit')],
                      ['Сравнение со средним по вузу на студента: Σ показателя всех ОП / Σ активных студентов.',
                       'Варианты — сценарии на тех же данных; предположения указаны рядом с каждым.'] ,
                      kind='explain')

    def _options(self, op_id: str):
        base_p = float(self.r.loc[op_id, 'финансовый_результат'])
        # 1) +10% платных студентов
        s1 = eng.simulate_enrollment(self.data, {op_id: 0.10}, grant_follows=False)
        d1 = float(s1['delta'].loc[s1['delta']['op_id'] == op_id, 'финансовый_результат_Δ'].iloc[0])
        be = eng.breakeven_growth(self.data, op_id, grant_follows=False)
        be_txt = ('ОП уже безубыточна.' if be == 0 else
                  f'Безубыточность в этой модели при росте платных студентов на ≈{be:.0%}.' if be is not None else
                  'Безубыточность недостижима ростом платного контингента в разумных пределах.')
        # 2) тариф +5%
        s2 = eng.simulate_enrollment(self.data, {}, tariff_change={op_id: 0.05})
        d2 = float(s2['delta'].loc[s2['delta']['op_id'] == op_id, 'финансовый_результат_Δ'].iloc[0])
        # 3) ФОТ −5%
        d3 = 0.05 * float(self.r.loc[op_id, 'фот_оп'])
        pl = self.details['payroll']
        t_here = set(pl.loc[pl['op_id'] == op_id, 'teacher_id'])
        shared = int(sum(1 for t in t_here if pl.loc[pl['teacher_id'] == t, 'op_id'].nunique() > 1))
        text = (f'- **A. Рост платного набора на 10%** (грантовые места не меняются): результат {tg(d1, True)}. {be_txt} '
                f'_Риски:_ спрос ограничен (по «История_набора» зачисление нестабильно), потребуются аудитории и группы; '
                f'модель линейна и не учитывает пороги.\n'
                f'- **Б. Повышение тарифа контракта на 5%**: результат {tg(d2, True)}. '
                f'_Риски:_ отток абитуриентов к конкурентам, снижение конверсии; эффект зависит от эластичности спроса, которой нет в данных.\n'
                f'- **В. Оптимизация нагрузки/ФОТ на 5%**: ≈ {tg(d3, True)} (5% × ФОТ ОП {tg(self.r.loc[op_id, "фот_оп"])}). '
                f'_Риски:_ качество преподавания и аккредитационные требования; {shared} из {len(t_here)} преподавателей ОП '
                f'ведут занятия и на других ОП — экономия может переместиться, а не исчезнуть.')
        ev = pd.DataFrame([
            {'Вариант': 'A. +10% платных студентов', 'Эффект на результат ОП, ₸': d1},
            {'Вариант': 'Б. Тариф контракта +5%', 'Эффект на результат ОП, ₸': d2},
            {'Вариант': 'В. ФОТ −5%', 'Эффект на результат ОП, ₸': d3}])
        return text, ev

    # ---- прочие ответы --------------------------------------------------------
    def a_op_summary(self, op_id: str) -> Answer:
        x = self.r.loc[op_id]
        text = (f'**{x["название_ОП"]} ({op_id}), {x["факультет"]}**\n\n'
                f'- Доход: {money(x["доход"])} (полный контракт {tg(x["доход_контракт_полный"])}, со скидкой '
                f'{tg(x["доход_контракт_скидка"])}, грант {tg(x["доход_грант"])})\n'
                f'- ФОТ: {money(x["фот_оп"])}\n- Прямые затраты: {money(x["прямые_затраты"])}\n'
                f'- Накладные: {money(x["накладные"])}\n- Итого затраты: {money(x["итого_затраты"])}\n'
                f'- **Финансовый результат: {money(x["финансовый_результат"])}**, рентабельность {pct(x["рентабельность_pct"])}\n'
                f'- Студентов: {x["студентов_активных"]:.0f}, себестоимость студента {tg(x["себестоимость_на_студента"])}\n\n'
                f'Чтобы разобрать причины, спросите: «Почему у {x["название_ОП"].lower()} такой результат?»')
        ev = pd.DataFrame([{'Статья': k, 'Сумма, ₸': x[c]} for k, c in (
            ('Доход', 'доход'), ('ФОТ', 'фот_оп'), ('Прямые затраты', 'прямые_затраты'), ('Накладные', 'накладные'),
            ('Итого затраты', 'итого_затраты'), ('Финансовый результат', 'финансовый_результат'))])
        return Answer(text, ev, [SRC_REVENUE, SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD],
                      [FORMULAS[k] for k in ('revenue', 'payroll', 'direct', 'overhead', 'profit', 'margin')], kind='op')

    def a_op_metric(self, op_id: str, col: str) -> Answer:
        label, unit, formula, sources, fkeys = self.METRIC_LABEL[col]
        v = self.r.loc[op_id, col]
        rank = int(self.result[col].rank(ascending=False, method='min')[self.result['op_id'] == op_id].iloc[0])
        text = (f'**{label} — {self.names[op_id]} ({op_id}): {self._fmt_metric(col, v)}**. '
                f'Это {rank}-е место из {len(self.result)} по убыванию показателя.\n\nФормула: {formula}.')
        ev = self.a_ranking(col).evidence
        return Answer(text, ev, sources, [FORMULAS[k] for k in fkeys], kind='metric')

    def a_compare(self, ops: list) -> Answer:
        cols = ['op_id', 'название_ОП', 'студентов_активных', 'доход', 'фот_оп', 'прямые_затраты', 'накладные',
                'итого_затраты', 'финансовый_результат', 'рентабельность_pct', 'себестоимость_на_студента']
        ev = self.result[self.result['op_id'].isin(ops)][cols]
        best = ev.sort_values('рентабельность_pct', ascending=False).iloc[0]
        text = (f'Сравнение {len(ev)} ОП. Лучшая рентабельность — **{best["название_ОП"]}: {pct(best["рентабельность_pct"])}** '
                f'(результат {money(best["финансовый_результат"])}). Полные цифры — в таблице ниже.')
        return Answer(text, ev, [SRC_REVENUE, SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD],
                      [FORMULAS['profit'], FORMULAS['margin']], kind='compare')

    def a_faculty(self, fac: str) -> Answer:
        g = self.result[self.result['факультет'] == fac]
        rev, cost = g['доход'].sum(), g['итого_затраты'].sum()
        text = (f'**{fac}** ({len(g)} ОП): доход {money(rev)}, затраты {money(cost)}, результат {money(rev - cost)}, '
                f'рентабельность {pct((rev - cost) / rev * 100 if rev else float("nan"))}.')
        return Answer(text, g[['op_id', 'название_ОП', 'доход', 'итого_затраты', 'финансовый_результат', 'рентабельность_pct']],
                      [SRC_REVENUE, SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD], [FORMULAS['profit'], FORMULAS['margin']], kind='faculty')

    def a_faculties(self) -> Answer:
        g = (self.result.groupby('факультет', as_index=False)
             .agg(ОП=('op_id', 'count'), доход=('доход', 'sum'), затраты=('итого_затраты', 'sum'),
                  результат=('финансовый_результат', 'sum')))
        g['рентабельность_pct'] = g['результат'] / g['доход'] * 100
        g = g.sort_values('результат', ascending=False)
        top, bot = g.iloc[0], g.iloc[-1]
        text = (f'Лучший факультет по результату — **{top["факультет"]}: {money(top["результат"])}** '
                f'({pct(top["рентабельность_pct"])}); наименьший результат — **{bot["факультет"]}: {money(bot["результат"])}** '
                f'({pct(bot["рентабельность_pct"])}).')
        return Answer(text, g, [SRC_REVENUE, SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD], [FORMULAS['profit']], kind='faculties')

    def a_overview(self) -> Answer:
        T = self.t
        text = (f'**Университет в целом (12 ОП, 2025–2026):** доход {money(T["доход"])}, затраты {money(T["затраты"])} '
                f'(ФОТ {tg(T["фот"])}, прямые {tg(T["прямые"])}, накладные {tg(T["накладные"])}), '
                f'**результат {money(T["результат"])}**, рентабельность {pct(T["рентабельность_pct"])}. '
                f'Убыточных ОП: {T["убыточных"]}. Контрольные суммы сходятся: '
                f'{"да" if self.controls["OK"].all() else "НЕТ — см. вкладку Data Quality"}.')
        return Answer(text, self.result[['op_id', 'название_ОП', 'доход', 'итого_затраты', 'финансовый_результат', 'рентабельность_pct']],
                      [SRC_REVENUE, SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD], [FORMULAS['total'], FORMULAS['profit']], kind='overview')

    def a_anomalies(self) -> Answer:
        a = self.anomalies
        if a.empty:
            return Answer('В листе «Сырые_сигналы» аномалий не найдено.', None, [SRC_SIGNALS], [], kind='anomalies')
        lines = [f'- **{x["ID строки"]}** (строка Excel {x["Строка Excel"]}) — {x["Тип ошибки"]}: {x["Почему подозрительно"]} '
                 f'→ _{x["Рекомендуемое действие"]}_' for _, x in a.iterrows()]
        text = (f'В «Сырые_сигналы» найдено **{len(a)} проблем(ы)**. Ни одна из них не попадает в расчёт P&L: '
                f'лист в финансовой модели не используется (проверено тестом: расчёт без листа идентичен).\n\n' + '\n'.join(lines))
        return Answer(text, a.drop(columns=['код'], errors='ignore'), [SRC_SIGNALS],
                      ['Правила: дубликат ID; op_id отсутствует в «ОП»; значение < 0; процент > 100; пусто; период вне допустимых.'],
                      kind='anomalies')

    def a_formulas(self, q: str) -> Answer:
        keys = list(FORMULAS)
        if re.search(r'рентабельн', q):
            keys = ['profit', 'margin']
        elif re.search(r'фот|зарплат', q):
            keys = ['payroll']
        elif re.search(r'накладн', q):
            keys = ['overhead']
        elif re.search(r'доход', q):
            keys = ['revenue']
        elif re.search(r'себестоим', q):
            keys = ['total', 'unit']
        text = 'Расчёт идёт по методике кейса:\n\n' + '\n'.join(f'- {FORMULAS[k]}' for k in keys)
        text += ('\n\nПримеры с реальными цифрами смотрите во вкладке «Образовательные программы» '
                 '(построчная детализация ФОТ и накладных).')
        return Answer(text, None, [SRC_REVENUE, SRC_PAYROLL, SRC_DIRECT, SRC_OVERHEAD], [FORMULAS[k] for k in keys],
                      ['Учебная модель: тенге, один год; налог на прибыль, амортизация, стипендии, инвестиции не моделируются.'],
                      kind='formulas')

    def a_scenario(self, question: str, ops: list) -> Answer:
        q = _norm(question)
        m = re.search(r'([+\-]?)\s?(\d+(?:[.,]\d+)?)\s?%', q)
        if m:
            val = float(m.group(2).replace(',', '.')) / 100
            neg = m.group(1) == '-' or re.search(r'сниз|уменьш|сократ|упад|меньше|падени', q)
            val = -val if (neg and not (m.group(1) == '+')) else val
        else:
            val = 0.10
        if not ops:
            return Answer('Укажите, для какой ОП считать сценарий, например: «Что если контингент на информатике вырастет на 10%?». '
                          'Либо откройте вкладку «What-If» для интерактивного расчёта.', kind='scenario')
        op = ops[0]
        s = eng.simulate_enrollment(self.data, {op: val}, grant_follows=True)
        d = s['delta']
        x = d[d['op_id'] == op].iloc[0]
        tb, ts = s['totals_base'], s['totals_scenario']
        text = (f'**Сценарий: контингент «{self.names[op]}» ({op}) {val:+.0%}** (студентов {x["студентов_активных_до"]:.0f} → '
                f'{x["студентов_активных_после"]:.1f}).\n\n'
                f'- Доход ОП: {money(x["доход_до"])} → {money(x["доход_после"])} ({tg(x["доход_Δ"], True)})\n'
                f'- ФОТ и прямые — без изменений (ФОТ {tg(x["фот_оп_после"])}, прямые {tg(x["прямые_затраты_после"])})\n'
                f'- Накладные ОП: {tg(x["накладные_до"])} → {tg(x["накладные_после"])} ({tg(x["накладные_Δ"], True)}) — '
                f'общая сумма 500 млн неизменна, центры по students перераспределяются\n'
                f'- **Результат ОП: {money(x["финансовый_результат_до"])} → {money(x["финансовый_результат_после"])} '
                f'({tg(x["финансовый_результат_Δ"], True)})**, рентабельность {pct(x["рентабельность_pct_до"])} → {pct(x["рентабельность_pct_после"])}\n'
                f'- Университет: результат {money(tb["результат"])} → {money(ts["результат"])} ({tg(ts["результат"] - tb["результат"], True)})')
        ev = pd.DataFrame([{'Показатель': lab, 'До': x[f'{c}_до'], 'После': x[f'{c}_после'], 'Δ': x[f'{c}_Δ']}
                           for lab, c in (('Студентов', 'студентов_активных'), ('Доход, ₸', 'доход'), ('ФОТ, ₸', 'фот_оп'),
                                          ('Прямые затраты, ₸', 'прямые_затраты'), ('Накладные, ₸', 'накладные'),
                                          ('Итого затраты, ₸', 'итого_затраты'), ('Финансовый результат, ₸', 'финансовый_результат'),
                                          ('Рентабельность, %', 'рентабельность_pct'))])
        return Answer(text, ev, [SRC_REVENUE, SRC_OVERHEAD, 'Вкладка «What-If» — тот же расчёт с настраиваемыми допущениями'],
                      [FORMULAS['revenue'], FORMULAS['overhead']],
                      eng.scenario_assumptions({op: val}, self.names, True, 0, 0), kind='scenario')

    def a_help(self) -> Answer:
        text = ('Я отвечаю на вопросы по рассчитанным данным университета. Все суммы считает код, я лишь объясняю.\n\n'
                'Попробуйте:\n' + '\n'.join(f'- {e}' for e in EXAMPLES))
        return Answer(text, None, [], [], kind='help')

    # ------------------------------------------------------------------ LLM слой
    def facts_json(self, base: Answer) -> str:
        cols = ['op_id', 'название_ОП', 'факультет', 'доход', 'фот_оп', 'прямые_затраты', 'накладные',
                'итого_затраты', 'финансовый_результат', 'рентабельность_pct', 'студентов_активных',
                'себестоимость_на_студента', 'фот_на_студента', 'доход_на_студента']
        facts = {
            'университет': {k: (None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 2)) for k, v in self.t.items()},
            'ОП': json.loads(self.result[cols].round(2).to_json(orient='records', force_ascii=False)),
            'аномалии_сырых_сигналов': json.loads(self.anomalies.drop(columns=['код'], errors='ignore').astype(object)
                                                  .where(self.anomalies.notna(), None).to_json(orient='records', force_ascii=False)),
            'готовый_ответ_кода': base.text,
            'доказательства': (json.loads(base.evidence.round(2).to_json(orient='records', force_ascii=False))
                               if base.evidence is not None else None),
        }
        return json.dumps(facts, ensure_ascii=False)

    def _rephrase_with_llm(self, question, base: Answer, api_key, model) -> Answer:
        try:
            import anthropic  # type: ignore
        except Exception:
            base.note = 'Пакет anthropic не установлен — показан локальный ответ.'
            return base
        facts = self.facts_json(base)
        system = (
            'Ты ИИ-аналитик финансов университета (учебный датасет, тенге, 2025–2026). Правила: '
            '1) используй ТОЛЬКО числа из блока FACTS, не вычисляй новых сумм и не округляй иначе; '
            '2) если данных для ответа нет — так и скажи; 3) всегда называй источник (лист Excel) и формулу; '
            '4) не утверждай, что ОП нужно закрыть по финансовым причинам; 5) отвечай по-русски, кратко, '
            'сохраняя структуру и все ключевые цифры из «готовый_ответ_кода».')
        try:
            client = anthropic.Anthropic(api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'))
            msg = client.messages.create(
                model=model, max_tokens=1500, system=system,
                messages=[{'role': 'user', 'content': f'Вопрос: {question}\n\nFACTS:\n{facts}'}])
            text = ''.join(b.text for b in msg.content if getattr(b, 'type', '') == 'text').strip()
        except Exception as exc:  # сеть, ключ, лимиты
            base.note = f'LLM недоступна ({type(exc).__name__}) — показан локальный ответ.'
            return base
        bad = unverified_numbers(text, facts + ' ' + question)
        if bad:
            base.note = ('Ответ LLM отклонён проверкой чисел (нет в расчёте: ' + ', '.join(bad[:6])
                         + ') — показан проверенный локальный ответ.')
            return base
        return Answer(text, base.evidence, base.sources, base.formulas, base.assumptions,
                      mode=f'LLM ({model}) + проверка чисел кодом', note='Все числа ответа найдены в расчётных данных.',
                      kind=base.kind)


# --------------------------------------------------------------------------
# Защита от выдуманных чисел
# --------------------------------------------------------------------------
_NUM = re.compile(r'(?<![\w.])[-−+]?\d[\d   ]*(?:[.,]\d+)?')


def _to_float(tok: str):
    t = tok.replace(' ', '').replace(' ', '').replace(' ', '').replace('−', '-').replace(',', '.').lstrip('+')
    try:
        return float(t)
    except ValueError:
        return None


def numbers_in(text: str) -> list:
    out = []
    for m in _NUM.finditer(text):
        v = _to_float(m.group(0))
        if v is not None:
            out.append((m.group(0).strip(), v))
    return out


def unverified_numbers(answer_text: str, facts_text: str) -> list:
    """Числа из ответа LLM, которых нет в расчётных данных.

    Допускаются только округления: тенге — до целого (±1), «млн»/«млрд» — до 1–2 знаков,
    проценты и малые величины — до 2 знаков. Годы и числа ≤ 20 (перечисления) игнорируются.
    """
    base = sorted({abs(v) for _, v in numbers_in(facts_text)})
    arr = np.array(base) if base else np.array([0.0])
    bad = []
    for raw, v in numbers_in(answer_text):
        av = abs(v)
        if av <= 20 or 2000 <= av <= 2100:
            continue
        tol_abs = 1.0 if av >= 1000 else 0.006
        ok = bool(np.any(np.abs(arr - av) <= tol_abs))
        ok = ok or bool(np.any(np.abs(arr / 1e6 - av) <= 0.051)) or bool(np.any(np.abs(arr / 1e9 - av) <= 0.051))
        if not ok:
            bad.append(raw)
    return bad
