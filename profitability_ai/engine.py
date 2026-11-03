"""Математический движок Profitability AI.

Все суммы считаются здесь, детерминированным кодом (pandas). Ни интерфейс,
ни ИИ-аналитик не вычисляют деньги самостоятельно — они только читают
результаты этого модуля.

Содержание:
    * load_workbook / validate_structure — загрузка и проверка структуры Excel;
    * calculate_financials / calculate_details — P&L по ОП и детализация;
    * control_checks — приёмочные контрольные суммы;
    * detect_signal_anomalies / detect_core_warnings — Data Quality;
    * simulate_enrollment / breakeven_growth — сценарный анализ (What-If);
    * forecast_enrollment — прогноз набора по истории (бонус).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Схема входного файла
# --------------------------------------------------------------------------
REQUIRED_SHEETS = [
    'ОП', 'Контингент', 'Штат_ППС', 'Нагрузка_ППС',
    'Прямые_затраты', 'Общие_затраты', 'Драйверы', 'История_набора', 'Сырые_сигналы'
]

REQUIRED_COLUMNS = {
    'ОП': ['op_id', 'название_ОП', 'факультет'],
    'Контингент': ['op_id', 'курс', 'контракт_полный', 'контракт_со_скидкой', 'грант',
                   'тариф_контракт_тг', 'финансирование_грант_тг', 'скидка_доля'],
    'Штат_ППС': ['teacher_id', 'зарплата_в_месяц_тг', 'месяцев_оплаты', 'начисления_работодателя_доля'],
    'Нагрузка_ППС': ['teacher_id', 'op_id', 'часы_в_год'],
    'Прямые_затраты': ['cost_id', 'op_id', 'категория', 'сумма_в_год_тг'],
    'Общие_затраты': ['cost_center_id', 'статья', 'сумма_в_год_тг', 'драйвер_распределения'],
    'Драйверы': ['op_id', 'студентов_активных', 'площадь_м2', 'лабораторные_часы'],
    'История_набора': ['op_id', 'учебный_год', 'заявок', 'зачислено', 'план_набора'],
    'Сырые_сигналы': ['source_row_id', 'op_id', 'показатель', 'учебный_год', 'значение'],
}

NUMERIC_COLUMNS = {
    'Контингент': ['курс', 'контракт_полный', 'контракт_со_скидкой', 'грант',
                   'тариф_контракт_тг', 'финансирование_грант_тг', 'скидка_доля'],
    'Штат_ППС': ['зарплата_в_месяц_тг', 'месяцев_оплаты', 'начисления_работодателя_доля'],
    'Нагрузка_ППС': ['часы_в_год'],
    'Прямые_затраты': ['сумма_в_год_тг'],
    'Общие_затраты': ['сумма_в_год_тг'],
    'Драйверы': ['студентов_активных', 'площадь_м2', 'лабораторные_часы'],
    'История_набора': ['заявок', 'зачислено', 'план_набора'],
}

# Соответствие кода драйвера колонке листа «Драйверы» (по методике кейса).
DRIVER_MAP = {
    'students': 'студентов_активных',
    'area_m2': 'площадь_м2',
    'lab_hours': 'лабораторные_часы',
}
DRIVER_RU = {
    'students': 'активные студенты',
    'area_m2': 'площадь, м²',
    'lab_hours': 'лабораторные часы',
}

TOLERANCE_TG = 1.0          # допуск по суммам (₸)
TOLERANCE_PCT = 0.01        # допуск по рентабельности (п.п.)
EXPECTED_PAYROLL_CONTROL = 736_465_200   # контрольные суммы из брифа (для тестового файла)
EXPECTED_OVERHEAD_CONTROL = 500_000_000


class DataError(ValueError):
    """Файл нельзя обработать (нет листов/колонок)."""


# --------------------------------------------------------------------------
# Загрузка и проверка структуры
# --------------------------------------------------------------------------
def validate_structure(sheets: dict) -> list[str]:
    """Возвращает список критичных проблем структуры (пустой список = ОК)."""
    problems = []
    for sheet, cols in REQUIRED_COLUMNS.items():
        if sheet not in sheets:
            problems.append(f'Отсутствует лист «{sheet}»')
            continue
        missing = [c for c in cols if c not in sheets[sheet].columns]
        if missing:
            problems.append(f'Лист «{sheet}»: нет колонок {", ".join(missing)}')
        if len(sheets[sheet]) == 0 and sheet != 'Сырые_сигналы':
            problems.append(f'Лист «{sheet}» пуст')
    return problems


def load_workbook(file) -> dict:
    """Читает Excel, проверяет структуру, приводит числовые колонки к числам.

    Нечисловые значения в обязательных числовых колонках превращаются в NaN
    (и затем попадают в отчёт Data Quality), а не роняют расчёт.
    """
    try:
        xls = pd.ExcelFile(file)
    except Exception as exc:  # битый файл / не xlsx
        raise DataError(f'Файл не удалось открыть как Excel (.xlsx): {exc}') from exc
    missing = [s for s in REQUIRED_SHEETS if s not in xls.sheet_names]
    if missing:
        raise DataError('В Excel отсутствуют листы: ' + ', '.join(missing))
    sheets = {s: pd.read_excel(xls, sheet_name=s) for s in REQUIRED_SHEETS}
    for sh in sheets.values():
        sh.columns = [str(c).strip() for c in sh.columns]
    problems = validate_structure(sheets)
    if problems:
        raise DataError('; '.join(problems))
    for sheet, cols in NUMERIC_COLUMNS.items():
        for c in cols:
            sheets[sheet][c] = pd.to_numeric(sheets[sheet][c], errors='coerce')
    # Сырые_сигналы намеренно НЕ приводим и не используем в расчёте.
    for sheet in ('ОП', 'Контингент', 'Нагрузка_ППС', 'Прямые_затраты', 'Драйверы', 'История_набора'):
        sheets[sheet]['op_id'] = sheets[sheet]['op_id'].astype(str).str.strip()
    sheets['Штат_ППС']['teacher_id'] = sheets['Штат_ППС']['teacher_id'].astype(str).str.strip()
    sheets['Нагрузка_ППС']['teacher_id'] = sheets['Нагрузка_ППС']['teacher_id'].astype(str).str.strip()
    return sheets


# --------------------------------------------------------------------------
# Финансовый расчёт
# --------------------------------------------------------------------------
def _compute(data: dict):
    """Единая точка расчёта. Возвращает (result, details)."""
    programs = data['ОП'].copy()
    students = data['Контингент'].copy()
    staff = data['Штат_ППС'].copy()
    workload = data['Нагрузка_ППС'].copy()
    direct = data['Прямые_затраты'].copy()
    overhead = data['Общие_затраты'].copy()
    drivers = data['Драйверы'].copy()

    num_cols = ['контракт_полный', 'контракт_со_скидкой', 'грант',
                'тариф_контракт_тг', 'финансирование_грант_тг', 'скидка_доля']
    students[num_cols] = students[num_cols].fillna(0)

    # ---- Доход ----------------------------------------------------------
    students['доход_контракт_полный'] = students['контракт_полный'] * students['тариф_контракт_тг']
    students['доход_контракт_скидка'] = (
        students['контракт_со_скидкой'] * students['тариф_контракт_тг'] * (1 - students['скидка_доля'])
    )
    students['доход_грант'] = students['грант'] * students['финансирование_грант_тг']
    students['доход'] = (students['доход_контракт_полный']
                         + students['доход_контракт_скидка'] + students['доход_грант'])
    students['студентов'] = students['контракт_полный'] + students['контракт_со_скидкой'] + students['грант']
    revenue = students.groupby('op_id', as_index=False).agg(
        доход=('доход', 'sum'),
        доход_контракт_полный=('доход_контракт_полный', 'sum'),
        доход_контракт_скидка=('доход_контракт_скидка', 'sum'),
        доход_грант=('доход_грант', 'sum'),
        студентов_контингент=('студентов', 'sum'),
        студентов_грант=('грант', 'sum'),
    )

    # ---- ФОТ ------------------------------------------------------------
    staff_num = ['зарплата_в_месяц_тг', 'месяцев_оплаты', 'начисления_работодателя_доля']
    staff[staff_num] = staff[staff_num].fillna(0)
    staff['годовой_фот'] = (staff['зарплата_в_месяц_тг'] * staff['месяцев_оплаты']
                            * (1 + staff['начисления_работодателя_доля']))
    workload['часы_в_год'] = workload['часы_в_год'].fillna(0)
    workload['всего_часов_преподавателя'] = workload.groupby('teacher_id')['часы_в_год'].transform('sum')
    workload = workload.merge(staff[['teacher_id', 'годовой_фот']], on='teacher_id',
                              how='left', validate='many_to_one')
    workload['доля_часов'] = np.where(workload['всего_часов_преподавателя'] > 0,
                                      workload['часы_в_год'] / workload['всего_часов_преподавателя'], 0.0)
    workload['фот_оп'] = workload['годовой_фот'].fillna(0) * workload['доля_часов']
    payroll = workload.groupby('op_id', as_index=False).agg(
        фот_оп=('фот_оп', 'sum'), часов_ппс=('часы_в_год', 'sum'),
        преподавателей=('teacher_id', 'nunique'))

    # ---- Прямые затраты -------------------------------------------------
    direct['сумма_в_год_тг'] = direct['сумма_в_год_тг'].fillna(0)
    direct_sum = (direct.groupby('op_id', as_index=False)['сумма_в_год_тг'].sum()
                  .rename(columns={'сумма_в_год_тг': 'прямые_затраты'}))

    # ---- Накладные ------------------------------------------------------
    overhead['сумма_в_год_тг'] = overhead['сумма_в_год_тг'].fillna(0)
    for c in DRIVER_MAP.values():
        drivers[c] = drivers[c].fillna(0)
    parts = []
    for _, center in overhead.iterrows():
        code = str(center['драйвер_распределения']).strip()
        col = DRIVER_MAP.get(code)
        if col is None:
            raise DataError(f"Неизвестный драйвер распределения «{code}» "
                            f"в центре {center['cost_center_id']} (допустимы: {', '.join(DRIVER_MAP)})")
        total = drivers[col].sum()
        if total == 0:
            raise DataError(f'Сумма драйвера «{col}» равна нулю — распределение невозможно')
        p = drivers[['op_id', col]].rename(columns={col: 'значение_драйвера'}).copy()
        p['cost_center_id'] = center['cost_center_id']
        p['статья'] = center['статья']
        p['драйвер'] = code
        p['сумма_центра'] = center['сумма_в_год_тг']
        p['сумма_драйверов'] = total
        p['доля'] = p['значение_драйвера'] / total
        p['накладные'] = p['сумма_центра'] * p['доля']
        parts.append(p)
    overhead_detail = pd.concat(parts, ignore_index=True)
    overhead_sum = overhead_detail.groupby('op_id', as_index=False)['накладные'].sum()

    # ---- Сборка ---------------------------------------------------------
    result = programs.merge(revenue, on='op_id', how='left')
    result = result.merge(payroll, on='op_id', how='left')
    result = result.merge(direct_sum, on='op_id', how='left')
    result = result.merge(overhead_sum, on='op_id', how='left')
    result = result.merge(drivers[['op_id', 'студентов_активных', 'площадь_м2', 'лабораторные_часы']],
                          on='op_id', how='left')
    numeric = ['доход', 'доход_контракт_полный', 'доход_контракт_скидка', 'доход_грант',
               'студентов_контингент', 'студентов_грант', 'фот_оп', 'часов_ппс', 'преподавателей',
               'прямые_затраты', 'накладные']
    result[numeric] = result[numeric].fillna(0)
    result['итого_затраты'] = result['фот_оп'] + result['прямые_затраты'] + result['накладные']
    result['финансовый_результат'] = result['доход'] - result['итого_затраты']
    result['рентабельность_pct'] = np.where(result['доход'] != 0,
                                            result['финансовый_результат'] / result['доход'].replace(0, np.nan) * 100,
                                            np.nan)
    stud = result['студентов_активных'].fillna(0)
    result['себестоимость_на_студента'] = np.where(
        stud != 0, result['итого_затраты'] / stud.replace(0, np.nan), np.nan)
    result['доход_на_студента'] = np.where(stud != 0, result['доход'] / stud.replace(0, np.nan), np.nan)
    result['фот_на_студента'] = np.where(stud != 0, result['фот_оп'] / stud.replace(0, np.nan), np.nan)

    details = {
        'revenue': students,
        'payroll': workload,
        'staff': staff,
        'direct': direct,
        'overhead': overhead_detail,
    }
    return result, details


def calculate_financials(data: dict) -> pd.DataFrame:
    """P&L по каждой ОП. Лист «Сырые_сигналы» в расчёте НЕ участвует."""
    return _compute(data)[0]


def calculate_details(data: dict):
    """Возвращает (result, details) — итог и построчную детализацию для drill-down и ИИ."""
    return _compute(data)


def totals(result: pd.DataFrame) -> dict:
    rev = result['доход'].sum()
    cost = result['итого_затраты'].sum()
    profit = rev - cost
    stud = result['студентов_активных'].sum()
    return {
        'доход': rev, 'фот': result['фот_оп'].sum(), 'прямые': result['прямые_затраты'].sum(),
        'накладные': result['накладные'].sum(), 'затраты': cost, 'результат': profit,
        'рентабельность_pct': profit / rev * 100 if rev else float('nan'),
        'студентов': stud,
        'себестоимость_на_студента': cost / stud if stud else float('nan'),
        'убыточных': int((result['финансовый_результат'] < 0).sum()),
    }


# --------------------------------------------------------------------------
# Контрольные суммы (приёмочные испытания)
# --------------------------------------------------------------------------
def control_checks(data: dict, result: pd.DataFrame, details: dict | None = None) -> pd.DataFrame:
    if details is None:
        result, details = _compute(data)
    staff = details['staff']
    rows = []

    def add(name, expected, actual, tol=TOLERANCE_TG, note=''):
        diff = actual - expected
        rows.append({'Проверка': name, 'Ожидается': expected, 'Получено': actual,
                     'Разница': diff, 'Допуск': tol, 'OK': bool(abs(diff) <= tol), 'Пояснение': note})

    add('Σ распределённого ФОТ = Σ годового ФОТ всех преподавателей',
        staff['годовой_фот'].sum(), result['фот_оп'].sum(),
        note='Источник: Штат_ППС × Нагрузка_ППС')
    add('Σ накладных по ОП = Σ центров затрат',
        data['Общие_затраты']['сумма_в_год_тг'].sum(), result['накладные'].sum(),
        note='Источник: Общие_затраты × Драйверы')
    add('Σ прямых затрат по ОП = Σ листа «Прямые_затраты»',
        data['Прямые_затраты']['сумма_в_год_тг'].sum(), result['прямые_затраты'].sum())
    add('Σ дохода по ОП = Σ дохода по строкам «Контингент»',
        details['revenue']['доход'].sum(), result['доход'].sum())
    add('Результат = Доход − (ФОТ + прямые + накладные)',
        result['доход'].sum() - (result['фот_оп'].sum() + result['прямые_затраты'].sum() + result['накладные'].sum()),
        result['финансовый_результат'].sum())
    add('Все ОП справочника присутствуют в расчёте',
        float(data['ОП']['op_id'].nunique()), float(result['op_id'].nunique()), tol=0)
    # Эталонные суммы из брифа проверяем только если файл — это тестовый датасет.
    if abs(staff['годовой_фот'].sum() - EXPECTED_PAYROLL_CONTROL) < 1:
        add('Эталон брифа: ФОТ = 736 465 200 ₸', EXPECTED_PAYROLL_CONTROL, result['фот_оп'].sum())
    if abs(data['Общие_затраты']['сумма_в_год_тг'].sum() - EXPECTED_OVERHEAD_CONTROL) < 1:
        add('Эталон брифа: накладные = 500 000 000 ₸', EXPECTED_OVERHEAD_CONTROL, result['накладные'].sum())
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Data Quality: «Сырые_сигналы»
# --------------------------------------------------------------------------
ANOMALY_TYPES = {
    'duplicate_id': ('Дубликат идентификатора',
                     'Оставить одну запись; выяснить у владельца источника, какая из двух верна.'),
    'unknown_op': ('Неизвестная ОП',
                   'Не загружать; исправить op_id в источнике или добавить программу в справочник «ОП».'),
    'negative': ('Отрицательное значение',
                 'Вернуть запись в источник на исправление; до этого не использовать.'),
    'pct_over_100': ('Процент вне диапазона 0–100',
                     'Проверить единицы измерения (доля vs %) и источник; до исправления не использовать.'),
    'missing_value': ('Пропуск значения',
                      'Запросить недостающее значение у владельца данных; не подставлять ноль.'),
    'bad_period': ('Недопустимый учебный год',
                   'Исправить период в источнике (допустимы годы из «История_набора»/«ОП»).'),
    'unknown_metric': ('Неизвестный показатель',
                       'Уточнить словарь показателей; до уточнения не использовать.'),
    'missing_id': ('Нет идентификатора строки',
                   'Присвоить идентификатор в источнике; без ID запись нельзя отследить.'),
}


def valid_periods(data: dict) -> set:
    periods = set(data['История_набора']['учебный_год'].dropna().astype(str))
    periods |= set(data['ОП']['учебный_год'].dropna().astype(str)) if 'учебный_год' in data['ОП'] else set()
    return periods


def detect_signal_anomalies(data: dict) -> pd.DataFrame:
    """Ищет ошибки в «Сырые_сигналы». Лист в P&L не используется ни при каких условиях."""
    sig = data['Сырые_сигналы'].copy()
    sig['excel_строка'] = sig.index + 2          # 1 — заголовок
    known_ops = set(data['ОП']['op_id'])
    periods = valid_periods(data)
    known_metrics = {'applications', 'conversion_pct'}
    issues = []

    def flag(row, code, why):
        label, action = ANOMALY_TYPES[code]
        issues.append({
            'ID строки': row['source_row_id'] if pd.notna(row['source_row_id']) else '(пусто)',
            'Строка Excel': int(row['excel_строка']),
            'op_id': row['op_id'], 'Показатель': row['показатель'],
            'Период': row['учебный_год'], 'Значение': row['значение'],
            'Тип ошибки': label, 'код': code,
            'Почему подозрительно': why, 'Рекомендуемое действие': action,
        })

    first_seen = {}
    for _, r in sig.iterrows():
        sid = r['source_row_id']
        if pd.isna(sid) or str(sid).strip() == '':
            flag(r, 'missing_id', 'Идентификатор строки не заполнен.')
        else:
            if sid in first_seen:
                f = first_seen[sid]
                same = (f['op_id'] == r['op_id'] and f['показатель'] == r['показатель']
                        and f['учебный_год'] == r['учебный_год'] and f['значение'] == r['значение'])
                flag(r, 'duplicate_id',
                     f'ID {sid} уже встречался в строке Excel {int(f["excel_строка"])}'
                     + (' с теми же данными (точная копия).' if same else ' с другими данными.')
                     + f' Комментарий источника: «{r.get("источник_комментарий")}».')
            else:
                first_seen[sid] = r
        if r['op_id'] not in known_ops:
            flag(r, 'unknown_op', f'op_id «{r["op_id"]}» отсутствует в справочнике ОП (12 программ).')
        value = pd.to_numeric(r['значение'], errors='coerce')
        if pd.isna(value):
            flag(r, 'missing_value', 'Значение пустое или не является числом.')
        else:
            if value < 0:
                flag(r, 'negative', f'Значение {value:g} < 0, а {r["показатель"]} не может быть отрицательным.')
            if str(r['показатель']).endswith('_pct') and value > 100:
                flag(r, 'pct_over_100', f'{r["показатель"]} = {value:g}% — больше 100%.')
        if str(r['учебный_год']) not in periods:
            flag(r, 'bad_period', f'Период «{r["учебный_год"]}» не входит в допустимые: {", ".join(sorted(periods))}.')
        if r['показатель'] not in known_metrics:
            flag(r, 'unknown_metric', f'Показатель «{r["показатель"]}» не описан в словаре.')
    cols = ['ID строки', 'Строка Excel', 'op_id', 'Показатель', 'Период', 'Значение',
            'Тип ошибки', 'Почему подозрительно', 'Рекомендуемое действие', 'код']
    return pd.DataFrame(issues, columns=cols)


def detect_core_warnings(data: dict, result: pd.DataFrame | None = None) -> pd.DataFrame:
    """Доп. проверки основных листов (влияют на P&L, поэтому показываются отдельно)."""
    rows = []

    def add(sev, sheet, what, details):
        rows.append({'Серьёзность': sev, 'Лист': sheet, 'Что проверено': what, 'Находка': details})

    ops = set(data['ОП']['op_id'])
    if data['ОП']['op_id'].duplicated().any():
        add('Критично', 'ОП', 'Уникальность op_id', 'Дубликаты: ' + ', '.join(data['ОП'][data['ОП']['op_id'].duplicated()]['op_id']))
    for sheet in ('Контингент', 'Нагрузка_ППС', 'Прямые_затраты', 'Драйверы', 'История_набора'):
        bad = sorted(set(data[sheet]['op_id']) - ops)
        if bad:
            add('Критично', sheet, 'op_id есть в справочнике ОП', 'Неизвестные op_id: ' + ', '.join(bad))
    if set(data['Драйверы']['op_id']) != ops:
        add('Критично', 'Драйверы', 'Все ОП имеют драйверы',
            'Нет драйверов для: ' + ', '.join(sorted(ops - set(data['Драйверы']['op_id']))))
    # пропуски/нечисловые значения в числовых колонках
    for sheet, cols in NUMERIC_COLUMNS.items():
        for c in cols:
            n = int(data[sheet][c].isna().sum())
            if n:
                add('Критично', sheet, f'Числовое значение «{c}»', f'{n} пустых/нечисловых ячеек (в расчёте = 0)')
            elif (data[sheet][c] < 0).any():
                add('Критично', sheet, f'Неотрицательность «{c}»', 'Есть отрицательные значения')
    k = data['Контингент']
    if ((k['скидка_доля'] < 0) | (k['скидка_доля'] > 1)).any():
        add('Критично', 'Контингент', 'Скидка в диапазоне 0–1', 'Есть скидки вне диапазона 0–1')
    if k.duplicated(['op_id', 'курс']).any():
        add('Предупреждение', 'Контингент', 'Уникальность (op_id, курс)', 'Есть повторы пар ОП+курс — доход может быть задвоен')
    staff, wl = data['Штат_ППС'], data['Нагрузка_ППС']
    if staff['teacher_id'].duplicated().any():
        add('Критично', 'Штат_ППС', 'Уникальность teacher_id', 'Есть повторяющиеся преподаватели')
    no_staff = sorted(set(wl['teacher_id']) - set(staff['teacher_id']))
    if no_staff:
        add('Критично', 'Нагрузка_ППС', 'Преподаватель есть в штате',
            'Нет в Штат_ППС: ' + ', '.join(no_staff[:10]) + ' (ФОТ не распределён)')
    no_load = sorted(set(staff['teacher_id']) - set(wl['teacher_id']))
    if no_load:
        add('Предупреждение', 'Штат_ППС', 'Преподаватель имеет нагрузку',
            'Нет нагрузки, ФОТ не будет распределён: ' + ', '.join(no_load[:10]))
    # контингент vs драйвер students
    k_sum = k.groupby('op_id')[['контракт_полный', 'контракт_со_скидкой', 'грант']].sum().sum(axis=1)
    d = data['Драйверы'].set_index('op_id')['студентов_активных']
    diff = (k_sum - d).dropna()
    diff = diff[diff.abs() > 0.5]
    if len(diff):
        add('Предупреждение', 'Драйверы', 'Студенты в «Драйверы» = Σ «Контингент»',
            'Расхождения: ' + '; '.join(f'{i}: {v:+.0f}' for i, v in diff.items()))
    h = data['История_набора']
    over = h[h['зачислено'] > h['заявок']]
    if len(over):
        add('Предупреждение', 'История_набора', 'Зачислено ≤ заявок',
            'Строки: ' + ', '.join(f'{r.op_id} {r.учебный_год}' for r in over.itertuples()))
    if not data['Общие_затраты']['драйвер_распределения'].astype(str).str.strip().isin(list(DRIVER_MAP)).all():
        add('Критично', 'Общие_затраты', 'Известный драйвер', 'Есть неизвестные драйверы распределения')
    if not rows:
        add('OK', '—', 'Все дополнительные проверки основных листов', 'Замечаний нет')
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Сценарный анализ (What-If)
# --------------------------------------------------------------------------
def simulate_enrollment(data: dict, changes: dict, grant_follows: bool = True,
                        payroll_var_share: float = 0.0, direct_var_share: float = 0.0,
                        tariff_change: dict | None = None) -> dict:
    """Пересчитывает P&L при изменении контингента.

    changes        {op_id: доля}, например {'OP-004': 0.10} = +10% студентов.
    grant_follows  True — меняются все категории (полный, со скидкой, грант);
                   False — грантовые места фиксированы государственным заказом.
    payroll_var_share / direct_var_share — какая доля ФОТ / прямых затрат растёт
                   вместе со студентами (базовый вариант кейса = 0: затраты фиксированы).
    tariff_change  {op_id: доля} — изменение тарифа контракта (опционально).

    Базовые допущения (по кейсу): доход пропорционален числу студентов; ФОТ и
    прямые затраты неизменны; суммы накладных центров неизменны, но центры с
    драйвером students перераспределяются по обновлённым студентам; площадь и
    лабораторные часы не меняются.
    """
    base_result, _ = _compute(data)
    new = {k: v.copy() for k, v in data.items()}
    k = new['Контингент']
    for c in ('контракт_полный', 'контракт_со_скидкой', 'грант', 'тариф_контракт_тг'):
        k[c] = k[c].astype(float)
    new['Драйверы']['студентов_активных'] = new['Драйверы']['студентов_активных'].astype(float)
    new['Прямые_затраты']['сумма_в_год_тг'] = new['Прямые_затраты']['сумма_в_год_тг'].astype(float)
    d = new['Драйверы'].set_index('op_id')
    direct = new['Прямые_затраты']
    for op, pct in changes.items():
        if op not in set(data['ОП']['op_id']):
            raise DataError(f'Неизвестная ОП: {op}')
        m = k['op_id'] == op
        k.loc[m, ['контракт_полный', 'контракт_со_скидкой']] *= (1 + pct)
        old_students = float(d.loc[op, 'студентов_активных'])
        paid_share = float(data['Контингент'].loc[data['Контингент']['op_id'] == op,
                                                  ['контракт_полный', 'контракт_со_скидкой']].sum().sum())
        grant_cnt = float(data['Контингент'].loc[data['Контингент']['op_id'] == op, 'грант'].sum())
        if grant_follows:
            k.loc[m, 'грант'] *= (1 + pct)
            d.loc[op, 'студентов_активных'] = old_students * (1 + pct)
        else:
            d.loc[op, 'студентов_активных'] = old_students + paid_share * pct
        # доля ФОТ/прямых, переменная вместе с контингентом
        factor = d.loc[op, 'студентов_активных'] / old_students if old_students else 1
        if direct_var_share:
            direct.loc[direct['op_id'] == op, 'сумма_в_год_тг'] *= 1 + direct_var_share * (factor - 1)
    for op, tp in (tariff_change or {}).items():
        k.loc[k['op_id'] == op, 'тариф_контракт_тг'] *= (1 + tp)
    new['Драйверы'] = d.reset_index()
    # Переменную часть ФОТ масштабируем после расчёта: изменение часов нагрузки
    # меняло бы только распределение, а не годовой ФОТ преподавателя.
    new_result, _ = _compute(new)
    if payroll_var_share:
        for op, pct in changes.items():
            old_row = base_result.loc[base_result['op_id'] == op].iloc[0]
            factor = float(new['Драйверы'].set_index('op_id').loc[op, 'студентов_активных']
                           / data['Драйверы'].set_index('op_id').loc[op, 'студентов_активных'])
            idx = new_result['op_id'] == op
            new_result.loc[idx, 'фот_оп'] = old_row['фот_оп'] * (1 + payroll_var_share * (factor - 1))
        new_result['итого_затраты'] = new_result['фот_оп'] + new_result['прямые_затраты'] + new_result['накладные']
        new_result['финансовый_результат'] = new_result['доход'] - new_result['итого_затраты']
        new_result['рентабельность_pct'] = np.where(
            new_result['доход'] != 0,
            new_result['финансовый_результат'] / new_result['доход'].replace(0, np.nan) * 100, np.nan)
        s = new_result['студентов_активных'].replace(0, np.nan)
        new_result['себестоимость_на_студента'] = new_result['итого_затраты'] / s

    cmp_cols = ['доход', 'фот_оп', 'прямые_затраты', 'накладные', 'итого_затраты',
                'финансовый_результат', 'рентабельность_pct', 'студентов_активных',
                'себестоимость_на_студента']
    delta = base_result[['op_id', 'название_ОП']].copy()
    for c in cmp_cols:
        delta[f'{c}_до'] = base_result[c].values
        delta[f'{c}_после'] = new_result[c].values
        delta[f'{c}_Δ'] = new_result[c].values - base_result[c].values
    return {'base': base_result, 'scenario': new_result, 'delta': delta,
            'totals_base': totals(base_result), 'totals_scenario': totals(new_result)}


def scenario_assumptions(changes: dict, names: dict, grant_follows: bool,
                         payroll_var_share: float, direct_var_share: float,
                         tariff_change: dict | None = None) -> list[str]:
    ch = ', '.join(f'{names.get(o, o)} ({o}): {p:+.0%}' for o, p in changes.items())
    a = [f'Изменение контингента: {ch}.',
         'Платные студенты (полный контракт и со скидкой) изменяются на указанный процент; '
         'тариф и доля скидки остаются прежними, доход пропорционален числу студентов.',
         ('Грантовые студенты изменяются на тот же процент.' if grant_follows else
          'Грантовые места фиксированы (государственный заказ), меняются только платные студенты.'),
         ('ФОТ остаётся неизменным (преподаватели уже наняты).' if not payroll_var_share else
          f'{payroll_var_share:.0%} ФОТ ОП меняется пропорционально контингенту.'),
         ('Прямые затраты остаются неизменными.' if not direct_var_share else
          f'{direct_var_share:.0%} прямых затрат ОП меняется пропорционально контингенту.'),
         'Общая сумма накладных (500 млн ₸ в тестовом файле) неизменна; центры с драйвером students '
         'перераспределяются между всеми 12 ОП по обновлённым студентам.',
         'Площадь (м²) и лабораторные часы не меняются — центры по этим драйверам распределяются как раньше.',
         'Модель линейна: допускаются дробные студенты, нет эффекта порогов (новые группы, потоки, аудитории).']
    if tariff_change:
        a.insert(2, 'Изменение тарифа контракта: ' + ', '.join(f'{o}: {p:+.0%}' for o, p in tariff_change.items()) + '.')
    return a


def breakeven_growth(data: dict, op_id: str, grant_follows: bool = False,
                     max_growth: float = 10.0) -> float | None:
    """Прирост студентов (в долях), при котором финансовый результат ОП = 0 (бисекция).

    По умолчанию растёт только платный контингент (грантовые места фиксированы) —
    так оценка консервативнее. None — безубыточность недостижима в модели.
    """
    def profit(p):
        r = simulate_enrollment(data, {op_id: p}, grant_follows=grant_follows)['scenario']
        return float(r.loc[r['op_id'] == op_id, 'финансовый_результат'].iloc[0])

    if profit(0) >= 0:
        return 0.0
    hi = max_growth
    if profit(hi) < 0:
        return None
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if profit(mid) < 0:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------
# Прогноз набора (бонус)
# --------------------------------------------------------------------------
T_CRIT_80 = {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476}   # t-критерий 80% интервала, df = n − 2


def forecast_enrollment(data: dict) -> pd.DataFrame:
    """Прогноз «зачислено» на следующий год по трём наблюдениям.

    Методы: последнее значение, среднее, линейный тренд, прирост на средний YoY.
    Диапазон = [min, max] методов, расширенный 80%-прогнозным интервалом линейной
    регрессии (df = 1). С тремя точками интервал неизбежно очень широк — это и есть
    честная неопределённость.
    """
    h = data['История_набора'].copy()
    names = data['ОП'].set_index('op_id')['название_ОП']
    years = sorted(h['учебный_год'].astype(str).unique())
    out = []
    for op, g in h.groupby('op_id'):
        g = g.assign(y=g['учебный_год'].astype(str)).sort_values('y')
        if len(g) < 2:
            continue
        x = np.arange(len(g), dtype=float)
        y = g['зачислено'].astype(float).values
        slope, intercept = np.polyfit(x, y, 1)
        x_next = float(len(g))
        trend = intercept + slope * x_next
        yoy = np.diff(y) / np.where(y[:-1] == 0, np.nan, y[:-1])
        yoy_mean = float(np.nanmean(yoy)) if len(yoy) else 0.0
        growth = y[-1] * (1 + yoy_mean)
        methods = {'последнее': y[-1], 'среднее': y.mean(), 'тренд': trend, 'YoY': growth}
        n = len(g)
        resid = y - (intercept + slope * x)
        if n > 2:
            s = math.sqrt((resid ** 2).sum() / (n - 2))
            se = s * math.sqrt(1 + 1 / n + (x_next - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
            half = T_CRIT_80.get(n - 2, 1.3) * se
        else:
            half = float('nan')
        low = min(min(methods.values()), trend - (half if not math.isnan(half) else 0))
        high = max(max(methods.values()), trend + (half if not math.isnan(half) else 0))
        plan_last = float(g['план_набора'].iloc[-1])
        out.append({
            'op_id': op, 'название_ОП': names.get(op, op),
            **{f'зачислено {r.y}': r.зачислено for r in g.itertuples()},
            'прогноз (тренд)': trend, 'нижняя граница': max(low, 0), 'верхняя граница': high,
            'ширина диапазона, %': (high - low) / trend * 100 if trend else float('nan'),
            'прирост YoY, %': yoy_mean * 100,
            'выполнение плана (посл. год), %': y[-1] / plan_last * 100 if plan_last else float('nan'),
            'наблюдений': n,
        })
    return pd.DataFrame(out)
