"""Сборка текста статьи: scripts/article_template.md + results/v2/*.csv → report/methodology.md.
Все числа и таблицы берутся из расчётных файлов; ручных чисел в тексте нет."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'results' / 'v2'
SUM = json.loads((R / 'summary.json').read_text())
K = SUM['selected_k']
KS = str(K)


def f(x, d=3, en=False, sign=False):
    s = f'{x:+.{d}f}'.replace('-', '−') if sign else f'{x:.{d}f}'
    s = s.replace('-', '−')
    return s if en else s.replace('.', ',')


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def interval(lo, hi, d=3):
    return f'[{f(lo, d)}; {f(hi, d)}]'


def main():
    ks = pd.read_csv(R / 'k_selection.csv')
    rob = pd.read_csv(R / 'robustness.csv')
    net = pd.read_csv(R / 'network_rules.csv')
    base = pd.read_csv(R / 'baselines.csv').set_index('method')
    sens = pd.read_csv(R / 'sensitivity.csv')
    ext = pd.read_csv(R / 'external_validation.csv')
    cv = pd.read_csv(R / 'practical_cv.csv')
    cal = pd.read_csv(R / 'assignment_calibration.csv')
    tr = pd.read_csv(R / f'transitions_K{K}.csv')
    ty = pd.read_csv(R / f'types_K{K}.csv')
    inc = pd.read_csv(R / 'incomplete_assignment.csv')
    prof = pd.read_csv(R / f'profiles_K{K}.csv')
    model = SUM['models'][KS]
    names = {int(k): v for k, v in model['names'].items()}
    V, T = {}, {}

    # --- общие величины
    V.update(n=f"{SUM['n']:,}".replace(',', ' '), K=K, orig=f"{SUM['original_n']:,}".replace(',', ' '),
             agg=f"{SUM['aggregated_panel_n']:,}".replace(',', ' '), incomplete=SUM['incomplete_n'], ctxmiss=SUM['context_missing_n'])
    mp = pd.read_csv(R / 'territory_mapping.csv')
    V['city_members'] = f"{SUM['original_n'] - SUM['aggregated_panel_n'] + 2} внутригородских МО"
    ba = np.array(model['bootstrap_ARI'])
    V.update(ari=f(ba.mean()), ari_en=f(ba.mean(), en=True), ari_low=f(np.percentile(ba, 2.5)), ari_high=f(np.percentile(ba, 97.5)), boot_n=len(ba))
    rb = rob.groupby('K').ARI_mean.first()
    V.update(ari3=f(rb[3]), ari4=f(rb[4]))
    V['ari_lens'] = f(model['label_lens_ARI'])
    V['n_methods'] = len(base)
    g = SUM['graph']
    V.update(edges=f"{g['edges']:,}".replace(',', ' '), deg_min=g['degree_min'], deg_max=g['degree_max'], deg_med=f(g['degree_median'], 0),
             within=f(g['within_region'] * 100, 1) + '%', cross=f((1 - g['within_region']) * 100, 1) + '%')

    # --- сеть
    nr = net.set_index('rule')
    V.update(ari_euclid=f(nr.loc['Евклид профиля', 'ARI_to_main'], 2), ari_dtw=f(nr.loc['DTW годовых профилей', 'ARI_to_main'], 2),
             ari_growth=f(nr.loc['Корреляция приростов', 'ARI_to_main'], 2), ari_lag=f(nr.loc['Лаговая корреляция', 'ARI_to_main'], 2),
             within_geo=f(nr.loc['География: haversine', 'within_region'] * 100, 1) + '%', ari_geo=f(nr.loc['География: haversine', 'ARI_to_main'], 2),
             jacc_lag=f(nr.loc['Лаговая корреляция', 'edge_jaccard'], 2), jacc_growth=f(nr.loc['Корреляция приростов', 'edge_jaccard'], 2))
    T['network'] = md_table(['Правило', 'Рёбер', 'Внутри региона', 'Жаккар к основной сети', 'ARI к основной типологии', 'SW', 'Сменили тип'],
        [[r.rule, f'{int(r.edges):,}'.replace(',', ' '), f(r.within_region * 100, 1) + '%', f(r.edge_jaccard, 3), f(r.ARI_to_main, 3), f(r.SW, 3), f(r.changed * 100, 1) + '%'] for r in net.itertuples()])

    # --- выбор K и индексы
    T['kselect'] = md_table(['K', 'Наименьший тип', 'ARI (среднее, 10 пересборок)', 'ARI (минимум)', 'Допустим'],
        [[int(r.K), f(r.min_share * 100, 1) + '%', f(r.ARI_mean), f(r.ARI_low), 'да' if r.eligible else 'нет'] for r in ks.itertuples()])
    T['icvi_k'] = md_table(['K', 'SW', 'CH', 'S_Dbw', 'AVI', 'AVU', 'ANUI', 'MQ', 'Q'],
        [[int(r.K), f(r.SW), f(r.CH, 1), f(r.S_Dbw), f(r.AVI), f(r.AVU), f(r.ANUI), f(r.MQ), f(r.Q)] for r in ks.itertuples()])
    T['avu'] = md_table(['K', '$\\mathrm{AVU}_{ref}$', '$(K-1)/(2K-3)$', 'AVU (вершинный)', '$\\mathrm{MQ}_{ref}$', '$\\mathrm{MQ}_{ref}/\\mathrm{AVI}_{ref}$'],
        [[int(r.K), f(r.AVU_ref), f((r.K - 1) / (2 * r.K - 3)), f(r.AVU), f(r.MQ_ref), f(r.MQ_ref / r.AVI_ref, 2)] for r in ks.itertuples()])

    # --- сравнение методов
    order = base.sort_values('network_rank').index
    cols = ['SW', 'CH', 'S_Dbw', 'AVI_ref', 'AVU_ref', 'ANUI_ref', 'MQ_ref', 'Q', 'seed_ARI']
    T['baselines'] = md_table(['Метод', 'SW', 'CH', 'S_Dbw', 'AVI', 'AVU', 'ANUI', 'MQ', 'Q', 'ARI между seed'],
        [[('**' + m + '**') if m == 'GS-TKM' else m] + [f(base.loc[m, c], 1 if c == 'CH' else 3) for c in cols] for m in order])
    T['baselines_full'] = md_table(['Метод', 'AVI', 'AVI$_{ref}$', 'AVU', 'AVU$_{ref}$', 'ANUI', 'MQ', 'Время, с'],
        [[m] + [f(base.loc[m, c], 3) for c in ['AVI', 'AVI_ref', 'AVU', 'AVU_ref', 'ANUI']] + [f(base.loc[m, 'MQ'], 3), f(base.loc[m, 'seconds'], 1)] for m in order])
    gs, km, cc = base.loc['GS-TKM'], base.loc['k-means (срезы)'], base.loc['Общие центры, без сети']
    graph_best = base.AVI_ref.idxmax(); feat_best = base.SW.idxmax()
    V['BASELINE_TEXT'] = (
        f'GS-TKM не занимает первых мест по индексам: лучший по эталонному AVI — «{graph_best}» ({f(base.loc[graph_best, "AVI_ref"])}), лучший по SW — «{feat_best}» ({f(base.loc[feat_best, "SW"])}); '
        f'у GS-TKM AVI = {f(gs.AVI_ref)} и SW = {f(gs.SW)}. По сравнению с k-means на срезах сглаживание повышает AVI с {f(km.AVI_ref)} до {f(gs.AVI_ref)} и снижает SW с {f(km.SW)} до {f(gs.SW)}: '
        f'признаковая компактность отдаётся за согласованность с сетью. Методы, оптимизирующие модулярность (Leiden, спектральный, многослойный Leiden), выигрывают по графовым индексам, но по SW и CH уступают методам на признаках. '
        f'Согласие запусков с разными seed у GS-TKM высокое ({f(gs.seed_ARI)}) и сопоставимо с Ward ({f(base.loc["Ward (срезы)", "seed_ARI"])}), спектральным методом ({f(base.loc["Spectral (срезы)", "seed_ARI"])}) и k-means ({f(base.loc["k-means (срезы)", "seed_ARI"])}), '
        f'но выше, чем у KEFRiN Euclidean ({f(base.loc["KEFRiN Euclidean (авторы)", "seed_ARI"])}), KEFRiN Cosine ({f(base.loc["KEFRiN Cosine (авторы)", "seed_ARI"])}), DMoN ({f(base.loc["DMoN", "seed_ARI"])}) и CANUS ({f(base.loc["CANUS (авторы)", "seed_ARI"])}). '
        f'Метки GS-TKM согласованы между окнами по построению (общие центры), чего нет у методов, запускаемых на срезах. '
        f'Штраф за смену типа почти не меняет оценки по отдельным окнам (AVI {f(gs.AVI_ref)} и {f(base.loc["GS-TKM без Витерби", "AVI_ref"])} без него): он влияет на траектории между окнами, а индексы вычисляются внутри окна.')
    V['ABLATION_TEXT'] = (
        f'Сглаживание по сети поднимает эталонный AVI с {f(cc.AVI_ref)} до {f(gs.AVI_ref)}, MQ — с {f(cc.MQ_ref)} до {f(gs.MQ_ref)} и модулярность Q — с {f(cc.Q)} до {f(gs.Q)}, '
        f'но снижает SW с {f(cc.SW)} до {f(gs.SW)} и CH с {f(cc.CH, 1)} до {f(gs.CH, 1)}: сглаженный признак сближает соседей по сети и отдаляет их от исходного профиля.')
    abl = ['Общие центры, без сети', 'GS-TKM без Витерби', 'GS-TKM']
    T['ablation'] = md_table(['Вариант', 'SW', 'CH', 'S_Dbw', 'AVI', 'MQ', 'Q', 'ARI между seed'],
        [[m, f(base.loc[m, 'SW']), f(base.loc[m, 'CH'], 1), f(base.loc[m, 'S_Dbw']), f(base.loc[m, 'AVI_ref']), f(base.loc[m, 'MQ_ref']), f(base.loc[m, 'Q']), f(base.loc[m, 'seed_ARI'])] for m in abl])

    # --- типы
    pr = prof.set_index('type')
    for i in range(K):
        V[f'n{i}'] = int(pr.loc[i, 'n'])
    T['profiles'] = md_table(['Тип', 'Название', 'МО', 'Расходы, руб.', 'Зарплата, руб.', 'Еда', 'Общеп.', 'Марк.', 'Пром.', 'Гос.'],
        [[i + 1, names[i], int(r.n), f'{r.spend:,.0f}'.replace(',', ' '), f'{r.wage:,.0f}'.replace(',', ' '), f(r.food * 100, 1) + '%', f(r.catering * 100, 1) + '%', f(r.market * 100, 1) + '%',
          f(r.manufacturing * 100, 1) + '%', f(r.public * 100, 1) + '%'] for i, r in pr.iterrows()])
    last = ty[ty.month == ty.month.max()].set_index('tid')
    lab = {}
    for k in (3, 4, 5):
        t = pd.read_csv(R / f'types_K{k}.csv'); t = t[t.month == t.month.max()].set_index('tid'); lab[k] = t.type
    ids = lab[5].index
    T['lens'] = md_table(['K', 'Наименьший тип в последнем окне', 'ARI с K = 5'],
        [[k, f(ks.set_index('K').loc[k, 'min_share'] * 100, 1) + '%', f(adjusted_rand_score(lab[k].loc[ids], lab[5].loc[ids]))] for k in (3, 4, 5)])
    rk = rob[rob.K == K].sort_values('type')
    T['robust'] = md_table(['Тип', 'Название', 'МО', 'Жаккар (среднее)', 'Диапазон пересборок', 'ARI (среднее)'],
        [[int(r.type) + 1, r.name, int(r.n), f(r.Jaccard_mean), interval(r.Jaccard_low, r.Jaccard_high, 2), f(r.ARI_mean)] for r in rk.itertuples()])
    jm = rk.loc[rk.Jaccard_mean.idxmin()]
    V.update(jac_min=f(jm.Jaccard_mean), name_low=jm['name'])

    # --- чувствительность
    FACT = {'spending_weight': 'Вес блока расходов', 'neighbors': 'Число соседей', 'smoothing_steps': 'Шаги сглаживания', 'transition_penalty': 'Штраф (доли медианы)',
            'seed': 'Seed k-means', 'unreported_coordinate': 'Координата неопубликованной занятости', 'city_policy': 'Столицы', 'window_months': 'Длина окна, мес.',
            'window_centering': 'Вычет медианы по территориям'}
    VAL = {'excluded': 'исключена', 'original': 'внутригородские МО раздельно', 'drop': 'столицы исключены', 'level': 'только уровень расходов', 'all_spending': 'все координаты расходов'}
    main_ch = sens[(sens.factor == 'neighbors') & (sens.value == '15')].changed_share.iloc[0]
    rows = []
    for r in sens.itertuples():
        v = VAL.get(str(r.value), str(r.value).replace('.', ','))
        if r.factor == 'seed': v = str(int(r.value) + 1)
        rows.append([FACT[r.factor], v, f(r.ARI, 3), f(r.changed_share * 100, 1) + '%'])
    T['sensitivity'] = md_table(['Параметр', 'Значение', 'ARI с основной типологией', 'Сменили тип'], rows)
    S = lambda fac, val, col: sens[(sens.factor == fac) & (sens.value.astype(str) == str(val))][col].iloc[0]
    mild = sens[sens.factor.isin(['neighbors', 'seed', 'unreported_coordinate'])].ARI
    mild = mild[mild < 0.9999]
    V.update(w25=f(S('spending_weight', 0.25, 'ARI')), w75=f(S('spending_weight', 0.75, 'ARI')), s0=f(S('smoothing_steps', 0, 'ARI')),
             ch_main=f(main_ch * 100, 1) + '%', ch_tau0=f(S('transition_penalty', 0, 'changed_share') * 100, 1) + '%', ch_tau1=f(S('transition_penalty', 1.0, 'changed_share') * 100, 1) + '%',
             min_mild=f(mild.min(), 2), city_orig=f(S('city_policy', 'original', 'ARI'), 2), city_drop=f(S('city_policy', 'drop', 'ARI'), 2),
             win3=f(S('window_months', 3, 'ARI'), 2), win3_ch=f(S('window_months', 3, 'changed_share') * 100, 1) + '%',
             changed_main=f(main_ch * 100, 1), changed_main_en=f(main_ch * 100, 1, True),
             changed_centered=f(S('window_centering', 'all_spending', 'changed_share') * 100, 1), changed_centered_en=f(S('window_centering', 'all_spending', 'changed_share') * 100, 1, True),
             ari_centered=f(S('window_centering', 'all_spending', 'ARI'), 2), changed_level=f(S('window_centering', 'level', 'changed_share') * 100, 1), ari_level=f(S('window_centering', 'level', 'ARI'), 2))
    V['ari_city'] = f(S('city_policy', 'original', 'ARI'), 2)
    for k in ('ch_main', 'ch_tau0', 'ch_tau1', 'win3_ch'):
        V[k] = V[k].rstrip('%')

    # --- неопределённость
    lt = last.subsample_agreement
    V.update(agr_med=f(lt.median(), 2), agr_ge90=f((lt >= 0.9).mean() * 100, 1), agr_lt50=f((lt < 0.5).mean() * 100, 1), neg_margin=f((last.signed_margin < 0).mean() * 100, 1))

    # --- динамика
    V.update(trans_n=len(tr), trans_tid=tr.tid.nunique(), trans_share=f(tr.tid.nunique() / SUM['n'] * 100, 1), robust_n=int(tr.robust_descriptive.sum()))
    fl = tr.groupby(['from', 'to']).size().unstack(fill_value=0).reindex(index=range(K), columns=range(K), fill_value=0)
    T['flow'] = md_table(['Из \\ в'] + [str(j + 1) for j in range(K)] + ['Всего'],
        [[f'{i + 1} ({names[i]})'] + [int(fl.loc[i, j]) if i != j else '—' for j in range(K)] + [int(fl.loc[i].sum())] for i in range(K)])
    top = fl.stack().sort_values(ascending=False)
    t1, t2, t3 = top.index[:3]
    src = int(fl.sum(axis=1).idxmax())
    V['FLOW_TEXT'] = (f'Крупнейшие потоки: из типа {t1[0] + 1} в тип {t1[1] + 1} ({int(top.iloc[0])} смен), из типа {t2[0] + 1} в тип {t2[1] + 1} ({int(top.iloc[1])}) и из типа {t3[0] + 1} в тип {t3[1] + 1} ({int(top.iloc[2])}); '
                      f'вместе они составляют {f(top.iloc[:3].sum() / len(tr) * 100, 0)}% всех смен. Эти направления нужно читать с учётом общего сдвига потребления (раздел 9.4): '
                      'движение территорий между соседними типами не обязательно означает изменение их положения относительно прочих территорий. '
                      f'Из типа {src + 1} уходит {f(fl.loc[src].sum() / len(tr) * 100, 0)}% всех смен; при пересборке на подвыборках наименьшее среднее значение Жаккара ({f(jm.Jaccard_mean)}) имеет тип {int(jm.type) + 1}.')
    ev = tr[tr.robust_descriptive].sort_values('trajectory_support_lower_bound', ascending=False)
    T['events'] = md_table(['Территория', 'Регион', 'Месяц', 'Из', 'В', 'Согласие', 'Запас', 'Нижняя граница'],
        [[r.name, r.region, r.month, int(r._5) + 1, int(r.to) + 1, f(r.subsample_agreement, 2), f(r.signed_margin, 2), f(r.trajectory_support_lower_bound, 2)] for r in ev.itertuples()])

    # --- внешняя проверка
    LENS = {'combined': 'Общая', 'spend': 'Потреб.'}
    T['external'] = md_table(['Линза', 'Показатель', 'МО', '$\\Delta\\bar R^2$ [интервал по регионам]', 'q (BH)', 'После контроля населения и зарплаты [интервал]'],
        [[LENS[r.lens], r.indicator, int(r.n), f'{f(r.delta_adj_r2, 4)} {interval(r.region_boot_low, r.region_boot_high, 4)}', f(r.q_BH, 3),
          '—' if pd.isna(r.delta_adj_r2_region_population_wage) else f'{f(r.delta_adj_r2_region_population_wage, 4)} {interval(r.controlled_boot_low, r.controlled_boot_high, 4)}'] for r in ext.itertuples()])
    cmb = ext[ext.lens == 'combined']
    V.update(ext_pos=int((cmb.region_boot_low > 0).sum()), ext_total=len(cmb), ext_ctrl_pos=int((cmb.controlled_boot_low > 0).sum()))
    V['ext_ctrl_text'] = '; '.join(f'{r.indicator.split(",")[0].lower()} — {f(r.delta_adj_r2_region_population_wage, 4)}' for r in cmb.itertuples())
    tr_row = cmb[cmb.indicator.str.startswith('Оборот торговли')].iloc[0]
    V['cov_trade'] = f(tr_row.coverage * 100, 0)

    # --- практическая проверка и калибровка
    T['cv'] = md_table(['Схема', 'Модель', 'MAE, лог. пункты', '$R^2$'], [[r.protocol, r.model, f(r.MAE_log_points, 4), f(r.R2, 3)] for r in cv.itertuples()])
    g_ = cv[cv.protocol == 'Регионы целиком'].iloc[0]
    V['cv_gain'] = f(g_.type_MAE_gain, 4); V['cv_gain_en'] = f(g_.type_MAE_gain, 4, True)
    V['cv_gain_text'] = f'для схемы «регионы целиком» {f(g_.type_MAE_gain, 4)} лог. пункта (интервал {interval(g_.gain_low, g_.gain_high, 4)})'
    T['calib'] = md_table(['Месяцев истории', 'МО', 'Покрытие', 'Согласие (все)', 'Присвоено', 'Согласие среди присвоенных', 'Нижняя граница Вильсона'],
        [[int(r.months), int(r.n), f(r.coverage * 100, 1) + '%', f(r.agreement * 100, 1) + '%', int(r.n_selected), f(r.agreement_selected * 100, 1) + '%', f(r.wilson_low * 100, 1) + '%'] for r in cal.itertuples()])
    c12 = cal[cal.months == 12].iloc[0]
    V.update(cal_agree=f(c12.agreement_selected * 100, 1), cal_wilson=f(c12.wilson_low * 100, 1), assigned=int((inc.status != 'unassigned').sum()))

    # --- нулевые модели
    nm = pd.read_csv(R / 'null_models.csv')
    n_null = int(round(1 / nm.p.min() - 1))
    V['n_null'] = n_null
    NAMES = {'SW': 'SW', 'CH': 'CH', 'S_Dbw': 'S_Dbw', 'AVI': 'AVI (вершинный)', 'AVU': 'AVU (вершинный)', 'ANUI': 'ANUI (вершинный)', 'MQ': 'MQ (вершинный)', 'Q': 'Q',
             'AVI_ref': 'AVI (эталон)', 'AVU_ref': 'AVU (эталон)', 'ANUI_ref': 'ANUI (эталон)', 'MQ_ref': 'MQ (эталон)'}
    rows = []
    for key, label in NAMES.items():
        cells = []
        for null in ('labels', 'region', 'config'):
            sub = nm[(nm['index'] == key) & (nm.null == null)]
            if sub.empty or sub.z.isna().all():
                cells.append('—'); continue
            pmax = sub.p.max()
            cells.append(f'{f(sub.z.mean(), 1)} ({"p ≤ " + f(1 / (n_null + 1), 3) if pmax <= 1 / (n_null + 1) + 1e-9 else "p = " + f(pmax, 3)})')
        rows.append([label] + cells)
    T['null'] = md_table(['Индекс', 'Перестановка меток: z (p)', 'Внутри регионов: z (p)', 'Перестройка рёбер: z (p)'], rows)

    # --- синтетика
    sy = pd.read_csv(R / 'synthetic.csv')
    G = lambda reg, m, col: sy[(sy['режим'] == reg) & (sy['метод'] == m)][col].iloc[0]
    GS, S0, L0, KM, TL = 'GS-TKM (наш)', 'GS-TKM s=0 (без графа)', 'GS-TKM λ=0 (без штрафа)', 'k-means по месяцам + Hungarian', 'temporal Leiden (только сеть)'
    REG = {'fe': 'fe', 'no_fe': 'no_fe', 'highnoise': 'highnoise', 'drift': 'drift', 'knn_graph': 'knn_graph', 'comp_graph': 'comp_graph'}
    shown = [(GS, 'GS-TKM'), (L0, 'GS-TKM без штрафа'), (S0, 'GS-TKM без сглаживания по сети'), (KM, 'k-means по окнам'), (TL, 'Leiden по сети')]
    rows = []
    for reg in REG:
        for m, label in shown:
            rows.append([reg if m == GS else '', ('**' + label + '**') if m == GS else label, f(G(reg, m, 'NMI')), f(G(reg, m, 'мигранты_0'), 2), f(G(reg, m, 'мигранты_2'), 2), f(G(reg, m, 'ложных_смен'))])
    T['synth'] = md_table(['Режим', 'Метод', 'NMI', 'Мигранты (в месяц смены)', 'Мигранты (через 2 мес.)', 'Ложные смены'], rows)
    V.update(syn_gs_comp=f(G('comp_graph', GS, 'NMI')), syn_s0_comp=f(G('comp_graph', S0, 'NMI')), syn_gs_knn=f(G('knn_graph', GS, 'NMI')), syn_s0_knn=f(G('knn_graph', S0, 'NMI')),
             syn_false_lam0_knn=f(G('knn_graph', L0, 'ложных_смен')), syn_false_gs_knn=f(G('knn_graph', GS, 'ложных_смен')),
             syn_mig0_fe=f(G('fe', GS, 'мигранты_0'), 2), syn_mig2_fe=f(G('fe', GS, 'мигранты_2'), 2),
             syn_gs_drift=f(G('drift', GS, 'NMI')), syn_km_drift=f(G('drift', KM, 'NMI')), syn_gs_nofe=f(G('no_fe', GS, 'NMI')), syn_s0_nofe=f(G('no_fe', S0, 'NMI')))

    # --- тесты
    log = (R / 'tests.log').read_text()
    m = re.search(r'(\d+) passed', log)
    V['tests'] = m.group(1) if m else '—'

    md = (ROOT / 'scripts' / 'article_template.md').read_text(encoding='utf-8')
    md = re.sub(r'<<T:(\w+)>>', lambda m_: T[m_.group(1)], md)
    md = re.sub(r'<<(\w+)>>', lambda m_: str(V[m_.group(1)]), md)
    left = re.findall(r'<<[^>]*>>', md)
    assert not left, left
    out = ROOT / 'report' / 'methodology.md'
    out.write_text(md, encoding='utf-8')
    print('готово:', out, len(md.split()), 'слов')


if __name__ == '__main__':
    main()
