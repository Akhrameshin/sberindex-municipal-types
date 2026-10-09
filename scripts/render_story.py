"""Сборка дашборда-лонгрида: все числа вычисляются из расчётных файлов results/v2 (отдельных ручных чисел нет).
Шаблон: src/story/{style.css, theme.js, app.js, body.html}; результат — dashboard_story.html."""
from pathlib import Path
import sys, json, base64, io, zipfile
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from atlas_v2 import *
from atlas_experiments import transitions

OUT = ROOT / 'results/v2'
STORY = ROOT / 'src/story'

SHORT = {'Низкие расходы, продовольственный профиль': 'Низкие расходы', 'Смешанная занятость и средние расходы': 'Смешанные',
         'Промышленность и сервисное потребление': 'Промышленные', 'Государственный сектор и умеренные расходы': 'Госсектор',
         'Высокий уровень расходов и зарплат': 'Высокие расходы', 'Высокий уровень расходов': 'Высокие расходы'}
NOTE = {'Низкие расходы, продовольственный профиль': 'самые низкие расходы и зарплаты, наибольшая доля продовольствия',
        'Смешанная занятость и средние расходы': 'средний уровень расходов, смешанная занятость',
        'Промышленность и сервисное потребление': 'повышенные расходы, наибольшие доли общепита и раскрытой промышленности',
        'Государственный сектор и умеренные расходы': 'умеренные расходы, наибольшая раскрытая доля госсектора',
        'Высокий уровень расходов и зарплат': 'наибольшие расходы и зарплаты',
        'Высокий уровень расходов': 'наибольшие расходы на жителя'}
FEATURES = ['Уровень расходов (лог)', 'Структура: здоровье', 'Структура: общепит', 'Структура: продовольствие', 'Структура: маркетплейсы',
            'Структура: транспорт', 'Структура: прочее', 'Население (лог)', 'Зарплата (лог)', 'Занятость: первичный сектор', 'Занятость: промышленность',
            'Занятость: торговля, гостиницы', 'Занятость: транспорт', 'Занятость: рыночные услуги', 'Занятость: госсектор', 'Занятость: нераспределённая']
REGIMES = {'fe': 'устойчивые эффекты узлов', 'no_fe': 'без эффектов узлов', 'highnoise': 'шум в три раза выше', 'drift': 'дрейф центров типов',
           'knn_graph': 'сеть из тех же признаков', 'comp_graph': 'сеть с дополнительной информацией'}


def code_archive():
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for folder in ['src', 'scripts', 'configs', 'tests', 'assets', 'results/v2']:
            for p in sorted((ROOT / folder).rglob('*')):
                if p.is_file() and p.name not in ['dashboard_payload.json', 'manifest.json'] and '__pycache__' not in p.parts and (p.suffix not in ['.html', '.png', '.log', '.pdf'] or p.name == 'tests.log'):
                    z.write(p, str(p.relative_to(ROOT)))
        for p in [ROOT / 'src/story/body.html', ROOT / 'src/story/assets/sber-logo-bw.png', ROOT / 'data/processed/panel.parquet',
                  *sorted((ROOT / 'data/external').glob('*.csv')), *sorted((ROOT / 'data/external').glob('*.json')),
                  *[ROOT / name for name in ['README.md', 'Makefile', 'requirements-core.txt', 'requirements-research.txt', 'SHA256SUMS', 'LICENSE']]]:
            if p.is_file(): z.write(p, str(p.relative_to(ROOT)))
    return b.getvalue()


def geometry(a, mapping):
    geo = json.loads((ROOT / 'assets/geography.json').read_text())
    for tid, group in mapping.groupby('analysis_tid'):
        parts = [geo['municipalities'][str(t)] for t in group.source_tid if str(t) in geo['municipalities']]
        if parts: geo['municipalities'][tid] = {'path': ''.join(g['path'] for g in parts), 'x': float(np.mean([g['x'] for g in parts])), 'y': float(np.mean([g['y'] for g in parts]))}
    m = geo['municipalities']
    return {'paths': [m[t]['path'] for t in a['ids']], 'px': [round(m[t]['x'], 1) for t in a['ids']], 'py': [round(m[t]['y'], 1) for t in a['ids']], 'W': geo['width'], 'H': geo['height']}


def pct(x, d=1): return f'{100 * x:.{d}f}'.replace('.', ',') + '%'
def num(x, d=3): return f'{x:.{d}f}'.replace('.', ',').replace('-', '−')


def rank_intervals(raw):
    """Место метода по эталонному AVI («net») и по силуэту SW («feat») в каждой из девяти оценок (дата × seed)."""
    out = {m: {'feat': [], 'net': []} for m in raw.method.unique()}
    for _, g in raw.groupby(['t', 'seed']):
        g = g.set_index('method')
        for key, col in (('feat', 'SW'), ('net', 'AVI_ref')):
            for m, v in g[col].rank(ascending=False, method='min').items(): out[m][key].append(float(v))
    return {m: {k: [float(np.median(v)), float(min(v)), float(max(v))] for k, v in d.items()} for m, d in out.items()}


def build():
    info = json.loads((OUT / 'summary.json').read_text())
    panel, mapping = load_panel(); a = build_arrays(panel); x, _ = scale_arrays(a)
    ids = a['ids']; N = len(ids); T = x.shape[0]; main = int(info['selected_k'])
    graphs = make_graphs(x)
    raw = a['raw']; total = raw[:, :, 0]; other = total - raw[:, :, 1:].sum(2); parts = np.concatenate([raw[:, :, 1:], other[:, :, None]], 2)
    annual_share = np.stack([parts[t:t + 12].sum(0) / parts[t:t + 12].sum(0).sum(1, keepdims=True) for t in range(T)])
    annual_level = np.stack([total[t:t + 12].mean(0) for t in range(T)])
    meta = a['meta']
    # признаки для паспорта: отклонение от медианы, σ
    F = np.c_[a['spending'][-1], a['context']]
    Z = ((F - np.nanmedian(F, 0)) / np.nanstd(F, 0)).clip(-3, 3); Z = np.nan_to_num(Z)
    ws = np.sqrt(CFG['features']['spending_weight'] / 7); wc = np.sqrt((1 - CFG['features']['spending_weight']) / 9)
    dimw = [ws] * 7 + [wc] * 9
    smoothed = smooth_features(x, graphs, CFG['model']['smoothing_steps'])
    # соседи по сети
    g = graphs[-1].tocsr(); xn = x[-1][:, :7] / (np.linalg.norm(x[-1][:, :7], axis=1, keepdims=True) + 1e-12)
    nb, sim = [], []
    for i in range(N):
        row = g.getrow(i); order = np.argsort(-row.data, kind='stable')[:10]; js = row.indices[order]
        nb.append([int(j) for j in js]); sim.append([round(float(xn[i] @ xn[j]), 2) for j in js])
    region = meta.region_name.fillna('?').to_numpy()
    same_region = float(np.mean([region[i] == region[j] for i in range(N) for j in nb[i]]))
    D = {'months': [str(m)[:7] for m in a['months']], 'T': T, 'N': N, 'ids': ids, 'main': main, 'geo': geometry(a, mapping),
         'mo': {'name': meta.municipal_district_name.fillna('?').tolist(), 'region': region.tolist(), 'pop': [int(v) if np.isfinite(v) else 0 for v in meta['pop'].to_numpy()]},
         'feat': {'names': FEATURES, 'z': np.round(Z, 2).tolist()}, 'dims': FEATURES, 'dimw': [round(float(v), 4) for v in dimw],
         'x_raw': np.round(x[-1], 3).tolist(), 'xs': np.round(smoothed[-1], 3).tolist(),
         'shares': {'cats': ['здоровье', 'общепит', 'продовольствие', 'маркетплейсы', 'транспорт', 'прочее'], 'v': np.round(annual_share[-1] * 100, 2).tolist()},
         'spend': [int(round(v)) for v in annual_level[-1]], 'nb': nb, 'sim': sim, 'same_region_share': same_region}
    # модели
    rob = pd.read_csv(OUT / 'robustness.csv'); D['K'] = {}
    for k in sorted(int(v) for v in info['models']):
        z = np.load(OUT / f'model_K{k}.npz'); f = fit_model(x, k, graphs=graphs)
        if not np.array_equal(f['labels'], z['labels']): raise RuntimeError('Метки модели не совпали с сохранёнными')
        names = [info['models'][str(k)]['names'][str(i)] for i in range(k)]
        lab = f['labels']; end = lab[-1]
        f0 = fit_model(x, k, graphs=graphs, steps=0); l0 = match_labels(lab[-1], f0['labels'][-1], k)
        conf = z['confidence']; margin = z['margin']
        boundary = ((conf[-1] < CFG['validation']['confidence_boundary']) | (margin[-1] < 0)).astype(int)
        flow = np.zeros((k, k), int)
        for i in range(N): flow[lab[0, i], lab[-1, i]] += 1
        changes = [int((lab[t] != lab[t - 1]).sum()) for t in range(1, T)]
        market = [[float(np.median(annual_share[t][end == c, 3]) * 100) for t in range(T)] for c in range(k)]
        dshare = (annual_share[-1] - annual_share[0]) * 100; dlevel = (annual_level[-1] / annual_level[0] - 1) * 100
        yoy = {'share': [[float(np.median(dshare[end == c, j])) for j in range(6)] for c in range(k)], 'level': [float(np.median(dlevel[end == c])) for c in range(k)]}
        jac = z['jaccard']
        hennig = [{'n': int((end == c).sum()), 'mean': float(jac[:, c].mean()), 'min': float(jac[:, c].min()), 'share': float((jac[:, c] > .75).mean())} for c in range(k)]
        D['K'][str(k)] = {'names': names, 'short': [SHORT.get(n, n) for n in names], 'note': [NOTE.get(n, '') for n in names], 'n': np.bincount(end, minlength=k).tolist(),
                          'L': lab.astype(int).tolist(), 'centers': np.round(f['centers'], 3).tolist(), 'dist': np.round(f['distances'], 3).tolist(),
                          'medmin': float(np.median(f['distances'].min(2))), 'lam': CFG['model']['transition_penalty'],
                          'ari_s0': float(adjusted_rand_score(end, l0)), 'diff_s0': float(np.mean(end != l0)),
                          'conf': np.round(conf[-1], 3).tolist(), 'boundary': boundary.tolist(), 'second': z['alternative'][-1].astype(int).tolist(),
                          'changed': [int(len(set(lab[:, i])) > 1) for i in range(N)],
                          'tmed': [np.round(np.median(Z[end == c], 0), 2).tolist() for c in range(k)],
                          'flow': flow.tolist(), 'changes_by_month': changes, 'marketplaces': [[round(v, 2) for v in r] for r in market],
                          'yoy': {'share': [[round(v, 2) for v in r] for r in yoy['share']], 'level': [round(v, 1) for v in yoy['level']]}, 'hennig': hennig,
                          'ari_boot': float(z['ari'].mean()), 'ari_boot_low': float(np.quantile(z['ari'], .025)), 'ari_boot_high': float(np.quantile(z['ari'], .975))}
    # методы
    base = pd.read_csv(OUT / 'baselines.csv'); ranks = rank_intervals(pd.read_csv(OUT / 'baselines_raw.csv'))
    D['methods'] = [{'name': r.method, 'ari': float(r.seed_ARI), 'sw': float(r.SW), 'ch': float(r.CH), 'sdbw': float(r.S_Dbw), 'avi': float(r.AVI_ref), 'avu': float(r.AVU_ref),
                     'anui': float(r.ANUI_ref), 'mq': float(r.MQ_ref), 'q': float(r.Q), 'seconds': float(r.seconds), 'feat': ranks[r.method]['feat'], 'net': ranks[r.method]['net']} for r in base.itertuples()]
    sy = pd.read_csv(OUT / 'synthetic.csv'); G = lambda reg, m, c: float(sy[(sy['режим'] == reg) & (sy['метод'] == m)][c].iloc[0])
    D['synth'] = [{'key': reg, 'regime': REGIMES[reg], 'nmi': G(reg, 'GS-TKM (наш)', 'NMI'), 'nmi0': G(reg, 'GS-TKM s=0 (без графа)', 'NMI'), 'nmi_km': G(reg, 'k-means по месяцам + Hungarian', 'NMI'),
                   'false': G(reg, 'GS-TKM (наш)', 'ложных_смен'), 'false0': G(reg, 'GS-TKM s=0 (без графа)', 'ложных_смен'), 'false_lam0': G(reg, 'GS-TKM λ=0 (без штрафа)', 'ложных_смен'),
                   'mig': G(reg, 'GS-TKM (наш)', 'мигранты_0'), 'mig2': G(reg, 'GS-TKM (наш)', 'мигранты_2')} for reg in REGIMES]
    ext = pd.read_csv(OUT / 'external_validation.csv')
    D['valid'] = [{'name': r.indicator, 'lens': r.lens, 'n': int(r.n), 'regions': int(r.regions), 'd': float(r.delta_adj_r2), 'lo': float(r.region_boot_low), 'hi': float(r.region_boot_high), 'q': float(r.q_BH),
                   'dc': None if pd.isna(r.delta_adj_r2_region_population_wage) else float(r.delta_adj_r2_region_population_wage),
                   'dclo': None if pd.isna(r.controlled_boot_low) else float(r.controlled_boot_low), 'dchi': None if pd.isna(r.controlled_boot_high) else float(r.controlled_boot_high)} for r in ext.itertuples()]
    nm = pd.read_csv(OUT / 'null_models.csv')
    D['null'] = [{'index': i, 'z': {n: (None if sub.z.isna().all() else float(sub.z.mean())) for n, sub in g_.groupby('null')}, 'p': {n: float(sub.p.max()) for n, sub in g_.groupby('null')}} for i, g_ in nm.groupby('index', sort=False)]
    D['n_null'] = int(round(1 / nm.p.min() - 1))
    cv = pd.read_csv(OUT / 'practical_cv.csv'); D['cv'] = [{'protocol': r.protocol, 'model': r.model, 'mae': float(r.MAE_log_points), 'r2': float(r.R2), 'gain': float(r.type_MAE_gain), 'glo': float(r.gain_low), 'ghi': float(r.gain_high)} for r in cv.itertuples()]
    net = pd.read_csv(OUT / 'network_rules.csv')
    D['edge'] = [{'name': r.rule, 'ari': float(r.ARI_to_main), 'jacc': float(r.edge_jaccard), 'within': float(r.within_region), 'sw': float(r.SW), 'edges': int(r.edges), 'changed': float(r.changed)} for r in net.itertuples()]
    ks = pd.read_csv(OUT / 'k_selection.csv'); D['kcurve'] = [{'K': int(r.K), 'min_share': float(r.min_share), 'ari': float(r.ARI_mean), 'ari_low': float(r.ARI_low), 'ok': bool(r.eligible)} for r in ks.itertuples()]
    sens = pd.read_csv(OUT / 'sensitivity.csv', dtype={'value': str}); D['sens'] = [{'factor': r.factor, 'value': r.value, 'ari': float(r.ARI), 'changed': float(r.changed_share)} for r in sens.itertuples()]
    cal = pd.read_csv(OUT / 'assignment_calibration.csv'); D['cal'] = [{'months': int(r.months), 'n': int(r.n), 'coverage': float(r.coverage), 'agree': float(r.agreement_selected), 'selected': int(r.n_selected), 'wilson': float(r.wilson_low)} for r in cal.itertuples()]
    inc = pd.read_csv(OUT / 'incomplete_assignment.csv'); D['incomplete'] = {'n': int(len(inc)), 'assigned': int((inc.status != 'unassigned').sum())}
    tr = pd.read_csv(OUT / f'transitions_K{main}.csv', dtype={'tid': str}); ev = tr[tr.robust_descriptive].sort_values('trajectory_support_lower_bound', ascending=False)
    D['events'] = [{'i': ids.index(r.tid), 'name': r.name, 'region': r.region, 'month': r.month, 'from': int(r._5), 'to': int(r.to), 'agree': float(r.subsample_agreement), 'lb': float(r.trajectory_support_lower_bound)} for r in ev.itertuples()]
    D['trans'] = {'n': int(len(tr)), 'tid': int(tr.tid.nunique()), 'persist': int(tr.persists_3_endpoints.sum()), 'robust': int(len(ev))}
    S = lambda fac, val, col: float(sens[(sens.factor == fac) & (sens.value == str(val))][col].iloc[0])
    D['drift'] = {'main': S('neighbors', 15, 'changed_share'), 'centered': S('window_centering', 'all_spending', 'changed_share'), 'level': S('window_centering', 'level', 'changed_share'),
                  'ari_centered': S('window_centering', 'all_spending', 'ARI')}
    D['graph'] = info['graph']; D['counts'] = {'original': info['original_n'], 'aggregated': info['aggregated_panel_n'], 'complete': info['n'], 'incomplete': info['incomplete_n'], 'context_missing': info['context_missing_n']}
    D['lens_ari'] = float(info['models'][str(main)]['label_lens_ARI'])
    D['runid'] = info['lineage']['run_id']
    D['tests'] = int(__import__('re').search(r'(\d+) passed', (OUT / 'tests.log').read_text())[1])
    D['ledger'] = ledger(D, main)
    return D


def ledger(D, main):
    K = D['K'][str(main)]; v = {r['name']: r for r in D['valid'] if r['lens'] == 'combined'}; ctrl = [r for r in D['valid'] if r['lens'] == 'combined' and r['dc'] is not None]
    ext_pos = sum(r['lo'] > 0 for r in ctrl); ctrl_pos = sum(r['dclo'] > 0 for r in ctrl)
    gs = next(m for m in D['methods'] if m['name'] == 'GS-TKM'); s = {r['key']: r for r in D['synth']}
    cvg = next(r for r in D['cv'] if r['protocol'] == 'Регионы целиком')
    null_ok = [r['index'] for r in D['null'] if all((p or 0) <= 1 / (D['n_null'] + 1) + 1e-9 for p in r['p'].values())]
    avu = next(r for r in D['null'] if r['index'] == 'AVU_ref'); c12 = next(r for r in D['cal'] if r['months'] == 12)
    sens = {(r['factor'], r['value']): r for r in D['sens']}
    L = []
    def add(st, t, r, w): L.append({'s': st, 't': t, 'r': r, 'w': w})
    add('ok', f'Разбиение воспроизводится при пересборке на подвыборках', f'Среднее согласие (ARI) при 50 пересборках на 80% территорий равно {num(K["ari_boot"])}.', 'раздел 9.2')
    add('ok', f'Метки не зависят от случайной инициализации', f'ARI между запусками с разными seed — {num(gs["ari"])}.', 'раздел 7.3')
    add('cv', f'Число типов K = {main} выбрано по правилу', f'Допустимы K = 3 и K = 5, выбран наибольший. Правило сформулировано по ходу анализа и не регистрировалось заранее.', 'раздел 5.3')
    add('ok', 'Типы различаются по внешним показателям внутри регионов', f'Региональный интервал добавочного R² целиком выше нуля для {ext_pos} из {len(ctrl)} показателей общей линзы.', 'раздел 11.2')
    add('cv', 'Связь типов с внешними показателями не сводится к размеру и зарплатам', f'После контроля населения и зарплаты интервал выше нуля только у {ctrl_pos} из {len(ctrl)} показателей; остаточный вклад мал.', 'раздел 11.2')
    hotel = v['Гостиничные места / житель, 2024']
    add('no', 'Тип связан с числом гостиничных мест', f'Добавочный R² {num(hotel["d"], 4)}, интервал [{num(hotel["lo"], 4)}; {num(hotel["hi"], 4)}] включает ноль.', 'раздел 11.2')
    add('no', 'Тип улучшает прогноз роста расходов сверх признаков и региона', f'Прирост MAE от типа {num(cvg["gain"], 4)} лог. пункта, интервал [{num(cvg["glo"], 4)}; {num(cvg["ghi"], 4)}] включает ноль.', 'раздел 12.1')
    add('ok', 'Индексы качества лучше нулевых моделей', f'Все индексы, кроме эталонного AVU, лучше всех {D["n_null"]} нулевых разбиений (p ≤ {num(1 / (D["n_null"] + 1))}). Оценки графовых индексов циклические: граф использован при сглаживании.', 'раздел 6.5')
    add('no', 'Эталонный AVU отличает разбиение от случайного', f'При всех нулевых моделях эталонный AVU хуже нуля (z = {num(avu["z"]["labels"], 1)} при перестановке меток): определение на уровне пар кластеров вырождено.', 'разделы 6.3, 6.5')
    add('no', 'Метод лидирует по индексам качества', f'AVI = {num(gs["avi"])} у GS-TKM против {num(max(m["avi"] for m in D["methods"]))} у лучшего метода; по SW метод также не первый.', 'раздел 7.4')
    add('ok', 'Сеть улучшает восстановление типов, когда несёт дополнительную информацию', f'Синтетика: NMI {num(s["comp_graph"]["nmi"])} с графовым сглаживанием против {num(s["comp_graph"]["nmi0"])} без него.', 'раздел 7.6')
    add('no', 'Сеть улучшает восстановление типов, когда построена из тех же признаков', f'Синтетика (как в реальной обработке): NMI {num(s["knn_graph"]["nmi"])} против {num(s["knn_graph"]["nmi0"])}.', 'раздел 7.6')
    add('ok', 'Штраф за смену типа подавляет ложные смены', f'Синтетика: доля ложных смен {num(s["knn_graph"]["false_lam0"])} без штрафа и {num(s["knn_graph"]["false"])} со штрафом.', 'раздел 7.6')
    add('no', 'Метод устойчив к дрейфу центров типов', f'Синтетика с дрейфом: NMI {num(s["drift"]["nmi"])} против {num(s["drift"]["nmi_km"])} у k-means по окнам.', 'раздел 7.6')
    d = D['drift']
    add('cv', 'Смены типа за год отражают изменение положения территории', f'Часть смен отражает общий сдвиг потребления: доля сменивших тип снижается с {pct(d["main"])} до {pct(d["centered"])} после вычитания медианы окна.', 'раздел 9.4')
    add('ok', 'Есть смены типа, устойчивые в описательном смысле', f'{D["trans"]["robust"]} из {D["trans"]["n"]} смен сохраняются на трёх конечных точках с нижней границей согласия не ниже 0,75.', 'раздел 10.2')
    add('no', 'Результат не зависит от весов блоков, сглаживания и длины окна', f'ARI с основной типологией: вес расходов 0,25 — {num(sens[("spending_weight", "0.25")]["ari"], 2)}; без сглаживания — {num(sens[("smoothing_steps", "0")]["ari"], 2)}; окно 3 мес. — {num(sens[("window_months", "3")]["ari"], 2)}.', 'раздел 9.3')
    add('cv', 'Территориям с неполной историей можно присвоить тип', f'При 12 месяцах согласие с полной моделью {pct(c12["agree"])} (нижняя граница Вильсона {pct(c12["wilson"])}); зависит от версии scikit-learn.', 'раздел 12.2')
    return L


def nb(n): return f'{int(n):,}'.replace(',', '\u00a0')


def texts(D):
    main = D['main']; K = D['K'][str(main)]; edge = {r['name']: r for r in D['edge']}; sens = {(r['factor'], r['value']): r for r in D['sens']}
    g = D['graph']; syn = {r['key']: r for r in D['synth']}; gs = next(m for m in D['methods'] if m['name'] == 'GS-TKM')
    cal = next(r for r in D['cal'] if r['months'] == 12); ctrl = [r for r in D['valid'] if r['lens'] == 'combined']
    never = 1 - np.mean(K['changed'])
    graph_best = max(D['methods'], key=lambda m: m['avi']); sw_best = max(D['methods'], key=lambda m: m['sw'])
    types = '; '.join(f'«{n}» ({nb(c)} МО)' for n, c in zip(K['names'], K['n']))
    mk = [r[3] for r in K['yoy']['share']]; lv = K['yoy']['level']
    jac_min = min(h['mean'] for h in K['hennig'])
    t = {'n': nb(D['N']), 'K': main, 'orig': nb(D['counts']['original']), 'agg': nb(D['counts']['aggregated']), 'inc': nb(D['counts']['incomplete']), 'assigned': D['incomplete']['assigned'],
         'cal_agree': pct(cal['agree']), 'cal_wilson': pct(cal['wilson']), 'ctxmiss': D['counts']['context_missing'], 'lens_ari': num(D['lens_ari'], 2),
         'edges': nb(g['edges']), 'degmin': g['degree_min'], 'degmax': g['degree_max'], 'degmed': f"{g['degree_median']:.0f}", 'within_edges': pct(g['within_region']),
         'same_top10': pct(D['same_region_share'], 0),
         'ari_euclid': num(edge['Евклид профиля']['ari'], 2), 'ari_dtw': num(edge['DTW годовых профилей']['ari'], 2), 'ari_growth': num(edge['Корреляция приростов']['ari'], 2), 'ari_lag': num(edge['Лаговая корреляция']['ari'], 2),
         'within_geo': pct(edge['География: haversine']['within']), 'ari_geo': num(edge['География: haversine']['ari'], 2),
         'jacc_lag': num(edge['Лаговая корреляция']['jacc'], 2), 'jacc_growth': num(edge['Корреляция приростов']['jacc'], 2),
         's0_ari': num(sens[('smoothing_steps', '0')]['ari'], 2), 's0_changed': pct(sens[('smoothing_steps', '0')]['changed']), 's2_ari': num(sens[('smoothing_steps', '2')]['ari'], 2),
         'ch_main': pct(sens[('transition_penalty', '0.5')]['changed']), 'ch_tau0': pct(sens[('transition_penalty', '0')]['changed']), 'ch_tau1': pct(sens[('transition_penalty', '1.0')]['changed']),
         'false_lam0': num(syn['knn_graph']['false_lam0']), 'false_gs': num(syn['knn_graph']['false']),
         'seed_ari': num(gs['ari']), 'ari_boot': num(K['ari_boot']), 'changed_main': pct(D['drift']['main']), 'changed_centered': pct(D['drift']['centered']),
         'jac_min': num(jac_min), 'type_list': types + '.', 'never_pct': pct(never), 'trans_n': D['trans']['n'], 'robust_n': D['trans']['robust'],
         'ext_pos': sum(r['lo'] > 0 for r in ctrl), 'ext_total': len(ctrl), 'ext_ctrl_pos': sum(r['dc'] is not None and r['dclo'] > 0 for r in ctrl),
         'mkt_lo': num(min(mk), 1), 'mkt_hi': num(max(mk), 1), 'lvl_lo': num(min(lv), 0), 'lvl_hi': num(max(lv), 0), 'tests': D['tests'],
         'rank_note': (f'Положение метода зависит от группы индексов. Лучший по эталонному AVI — «{graph_best["name"]}» ({num(graph_best["avi"])}), лучший по SW — «{sw_best["name"]}» ({num(sw_best["sw"])}); '
                       f'у GS-TKM AVI = {num(gs["avi"])}, SW = {num(gs["sw"])}. Надёжного победителя по индексам одного окна нет, а оценки графовых индексов цикличны: граф использован при сглаживании.')}
    return t



def render():
    if CFG['features']['window'] != 12: raise ValueError('Дашборд описывает годовые окна')
    D = build()
    report = ROOT / 'report/methodology.pdf'; md = ROOT / 'report/methodology.md'
    D['reportPDF'] = base64.b64encode(report.read_bytes()).decode() if report.exists() else ''
    D['reportMD'] = md.read_text() if md.exists() else ''
    D['repo'] = CFG.get('presentation', {}).get('repository_url', '')
    D['files'] = {name: (OUT / name).read_text() for name in ['external_validation.csv', 'baselines.csv', 'sensitivity.csv', 'practical_cv.csv', 'null_models.csv', 'synthetic.csv', 'incomplete_assignment.csv'] if (OUT / name).exists()}
    ver = OUT / 'verification.json'
    D['verification'] = json.loads(ver.read_text()).get('display', '') if ver.exists() else ''
    repo = CFG.get('presentation', {}).get('repository_url', '')
    logo = 'data:image/png;base64,' + base64.b64encode((STORY / 'assets/sber-logo-bw.png').read_bytes()).decode()
    body = (STORY / 'body.html').read_text().replace('__LOGOBW__', logo).replace('__REPO__', repo)
    import re
    fill = texts(D); body = re.sub(r'%%(\w+)%%', lambda m: str(fill[m.group(1)]), body)
    payload = json.dumps(D, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    html = ('<!doctype html>\n<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
            '<meta name="description" content="Типы местных экономик России: интерактивный отчёт к конкурсу СберИндекса 2026">\n'
            '<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 32 32\'%3E%3Crect width=\'32\' height=\'32\' rx=\'7\' fill=\'%23050907\'/%3E%3Ccircle cx=\'10\' cy=\'11\' r=\'4\' fill=\'%233a6aa7\'/%3E%3Ccircle cx=\'22\' cy=\'11\' r=\'4\' fill=\'%23b9daff\'/%3E%3Ccircle cx=\'16\' cy=\'22\' r=\'4\' fill=\'%23ebaa2d\'/%3E%3C/svg%3E">\n'
            '<title>Типы местных экономик России</title>\n<style>' + (STORY / 'style.css').read_text() + '</style></head><body>\n' + body +
            '\n<script id="payload" type="application/json">' + payload + '</script>\n<script>const D=JSON.parse(document.getElementById("payload").textContent);</script>\n'
            '<script>' + (STORY / 'theme.js').read_text() + '</script>\n<script>' + (STORY / 'app.js').read_text() + '</script></body></html>\n')
    (ROOT / 'dashboard_story.html').write_text(html, encoding='utf-8')
    print('Dashboard bytes', len(html.encode()), flush=True)


if __name__ == '__main__':
    render()
