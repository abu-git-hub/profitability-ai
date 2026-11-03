"""Тесты: независимый пересчёт «в лоб» (openpyxl, чистый Python) сверяется с engine.py."""
import copy
import sys
from collections import defaultdict
from pathlib import Path

import openpyxl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ai_analyst as ai  # noqa: E402
import engine as eng  # noqa: E402

XLSX = ROOT / 'data' / 'Profitability_AI_Test_Data.xlsx'


@pytest.fixture(scope='module')
def data():
    return eng.load_workbook(XLSX)


@pytest.fixture(scope='module')
def calc(data):
    return eng.calculate_details(data)


def rows(sheet):
    ws = openpyxl.load_workbook(XLSX, data_only=True)[sheet]
    it = ws.iter_rows(values_only=True)
    head = next(it)
    return [dict(zip(head, r)) for r in it]


def independent():
    rev = defaultdict(float)
    for r in rows('Контингент'):
        rev[r['op_id']] += (r['контракт_полный'] * r['тариф_контракт_тг']
                            + r['контракт_со_скидкой'] * r['тариф_контракт_тг'] * (1 - r['скидка_доля'])
                            + r['грант'] * r['финансирование_грант_тг'])
    annual = {r['teacher_id']: r['зарплата_в_месяц_тг'] * r['месяцев_оплаты'] * (1 + r['начисления_работодателя_доля'])
              for r in rows('Штат_ППС')}
    load = rows('Нагрузка_ППС')
    tot = defaultdict(float)
    for r in load:
        tot[r['teacher_id']] += r['часы_в_год']
    fot = defaultdict(float)
    for r in load:
        fot[r['op_id']] += annual[r['teacher_id']] * r['часы_в_год'] / tot[r['teacher_id']]
    direct = defaultdict(float)
    for r in rows('Прямые_затраты'):
        direct[r['op_id']] += r['сумма_в_год_тг']
    drv = {r['op_id']: r for r in rows('Драйверы')}
    m = {'students': 'студентов_активных', 'area_m2': 'площадь_м2', 'lab_hours': 'лабораторные_часы'}
    over = defaultdict(float)
    for c in rows('Общие_затраты'):
        col = m[c['драйвер_распределения']]
        s = sum(d[col] for d in drv.values())
        for op, d in drv.items():
            over[op] += c['сумма_в_год_тг'] * d[col] / s
    return rev, fot, direct, over, drv


def test_all_programs_and_controls(data, calc):
    result, details = calc
    assert len(result) == 12
    ctl = eng.control_checks(data, result, details)
    assert ctl['OK'].all(), ctl
    assert abs(result['фот_оп'].sum() - 736_465_200) <= 1
    assert abs(result['накладные'].sum() - 500_000_000) <= 1


def test_matches_independent_calculation(calc):
    result, _ = calc
    rev, fot, direct, over, drv = independent()
    for r in result.itertuples():
        assert abs(r.доход - rev[r.op_id]) <= 1
        assert abs(r.фот_оп - fot[r.op_id]) <= 1
        assert abs(r.прямые_затраты - direct[r.op_id]) <= 1
        assert abs(r.накладные - over[r.op_id]) <= 1
        cost = fot[r.op_id] + direct[r.op_id] + over[r.op_id]
        profit = rev[r.op_id] - cost
        assert abs(r.финансовый_результат - profit) <= 1
        assert abs(r.рентабельность_pct - profit / rev[r.op_id] * 100) <= 0.01
        assert abs(r.себестоимость_на_студента - cost / drv[r.op_id]['студентов_активных']) <= 1


def test_six_anomalies_detected_exactly(data):
    a = eng.detect_signal_anomalies(data)
    assert len(a) == 6
    assert set(a['ID строки']) == {'SRC-004', 'SRC-901', 'SRC-902', 'SRC-903', 'SRC-904', 'SRC-905'}
    assert set(a['код']) == {'duplicate_id', 'unknown_op', 'negative', 'pct_over_100', 'missing_value', 'bad_period'}
    assert (a['Строка Excel'] > 1).all()


def test_signals_do_not_affect_pnl(data, calc):
    stripped = dict(data)
    stripped['Сырые_сигналы'] = data['Сырые_сигналы'].iloc[0:0]
    r2 = eng.calculate_financials(stripped)
    assert (r2['финансовый_результат'] - calc[0]['финансовый_результат']).abs().max() == 0


def test_no_false_positives_on_clean_signals(data):
    clean = dict(data)
    clean['Сырые_сигналы'] = data['Сырые_сигналы'].iloc[:24]
    assert eng.detect_signal_anomalies(clean).empty


def test_zero_students_gives_nan_not_error(data):
    d = {k: v.copy() for k, v in data.items()}
    d['Драйверы'].loc[d['Драйверы']['op_id'] == 'OP-004', 'студентов_активных'] = 0
    r = eng.calculate_financials(d)
    assert r.loc[r['op_id'] == 'OP-004', 'себестоимость_на_студента'].isna().all()


def test_scenario_base_rules(data, calc):
    base = calc[0]
    s = eng.simulate_enrollment(data, {'OP-004': 0.10})
    new, old = s['scenario'].set_index('op_id'), base.set_index('op_id')
    assert abs(new.loc['OP-004', 'доход'] - old.loc['OP-004', 'доход'] * 1.1) <= 1       # доход ∝ студентам
    assert (new['фот_оп'] - old['фот_оп']).abs().max() <= 1e-6                           # ФОТ неизменен
    assert (new['прямые_затраты'] - old['прямые_затраты']).abs().max() <= 1e-6           # прямые неизменны
    assert abs(new['накладные'].sum() - 500_000_000) <= 1                                # сумма накладных неизменна
    assert new.loc['OP-004', 'накладные'] > old.loc['OP-004', 'накладные']               # перераспределение
    assert new.loc['OP-001', 'накладные'] < old.loc['OP-001', 'накладные']
    # площадь и лаб. центры не меняются: накладные по ним у OP-004 те же
    oh = eng.calculate_details(data)[1]['overhead']
    assert set(oh['драйвер']) == {'students', 'area_m2', 'lab_hours'}


def test_scenario_zero_change_is_identity(data, calc):
    s = eng.simulate_enrollment(data, {'OP-001': 0.0})
    assert (s['delta']['финансовый_результат_Δ'].abs() < 1e-6).all()


def test_breakeven(data):
    be = eng.breakeven_growth(data, 'OP-011')
    assert 0 < be < 0.2
    s = eng.simulate_enrollment(data, {'OP-011': be}, grant_follows=False)['scenario']
    assert abs(float(s.loc[s['op_id'] == 'OP-011', 'финансовый_результат'].iloc[0])) < 1000
    assert eng.breakeven_growth(data, 'OP-001') == 0.0


def test_forecast(data):
    fc = eng.forecast_enrollment(data)
    assert len(fc) == 12 and (fc['нижняя граница'] <= fc['верхняя граница']).all()


def test_missing_sheet_raises():
    wb = openpyxl.load_workbook(XLSX)
    del wb['Драйверы']
    import io
    b = io.BytesIO()
    wb.save(b)
    b.seek(0)
    with pytest.raises(eng.DataError):
        eng.load_workbook(b)


# ------------------------------ ИИ-аналитик ------------------------------
@pytest.fixture(scope='module')
def analyst(data, calc):
    return ai.Analyst(data, calc[0], calc[1])


def test_ai_losses(analyst):
    a = analyst.answer('Какие ОП убыточны?')
    assert a.kind == 'losses'
    for name in ('Прикладная математика', 'Переводческое дело', 'Сестринское дело'):
        assert name in a.text
    assert '3 из 12' in a.text


def test_ai_highest_fot_per_student(analyst):
    a = analyst.answer('Где самый высокий ФОТ на студента?')
    assert a.kind == 'ranking' and 'Прикладная математика' in a.text


def test_ai_explain_factors_reconcile(analyst):
    f = analyst.explain_factors('OP-004')
    r = analyst.r.loc['OP-004']
    avg = analyst.t['результат'] / analyst.t['студентов']
    assert abs(f['Вклад в результат, ₸'].sum() - (r['финансовый_результат'] - r['студентов_активных'] * avg)) < 1
    a = analyst.answer('Почему у прикладной математики такой результат?')
    assert a.kind == 'explain' and a.text.count('Риски') >= 2 and a.formulas and a.sources


def test_ai_program_name_matching(analyst):
    assert analyst.find_programs('финансовый результат') == []          # слово «финансовый» — не ОП «Финансы»
    assert analyst.find_programs('по финансам')[0] == 'OP-010'
    assert analyst.find_programs('общей медицине')[0] == 'OP-012'
    assert analyst.find_programs('информатика')[0] == 'OP-001'
    assert analyst.find_programs('информационные системы')[0] == 'OP-002'
    assert analyst.find_programs('ОП-007')[0] == 'OP-007'


def test_ai_scenario_numbers_match_engine(analyst, data):
    a = analyst.answer('Что если контингент на сестринском деле вырастет на 10%?')
    s = eng.simulate_enrollment(data, {'OP-011': 0.10})
    after = s['scenario'].set_index('op_id').loc['OP-011', 'финансовый_результат']
    assert ai.tg(after) in a.text


def test_ai_anomalies(analyst):
    a = analyst.answer('Какие аномалии в сырых сигналах?')
    for sid in ('SRC-004', 'SRC-901', 'SRC-902', 'SRC-903', 'SRC-904', 'SRC-905'):
        assert sid in a.text


def test_number_guard():
    facts = '{"доход": 44409000.0, "результат": -42368719.2}'
    assert ai.unverified_numbers('Доход 44 409 000 ₸, результат −42,37 млн ₸', facts) == []
    assert ai.unverified_numbers('Доход 50 000 000 ₸', facts) == ['50 000 000']


def test_number_guard_rejects_lookalike(data, calc):
    """Выдуманное число, близкое по порядку к реальному, не должно проходить."""
    a = ai.Analyst(data, calc[0], calc[1])
    base = a.answer('Какие ОП убыточны?')
    facts = a.facts_json(base)
    assert ai.unverified_numbers('Убыток −99 999 999 ₸', facts) == ['−99 999 999']
    assert ai.unverified_numbers(base.text, facts) == []
