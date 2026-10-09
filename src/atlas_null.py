"""Значимость индексов качества против трёх нулевых моделей на основной типологии версии 2.
Окна: декабрь 2023, июнь 2024, декабрь 2024. Нулевые модели:
  labels  — метки переставлены случайно (размеры типов сохраняются);
  region  — метки переставлены внутри регионов (сохраняется региональный состав типов);
  config  — рёбра графа перестроены с сохранением степеней, веса переставлены (только графовые индексы).
z > 0 означает «лучше нуля» для всех индексов (для S_Dbw и AVU знак инвертирован). p — односторонний эмпирический, минимум 1/(n+1).
Признаковые индексы при модели config не меняются по построению и не считаются."""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, igraph as ig, scipy.sparse as sp
from scipy.spatial.distance import cdist
from atlas_v2 import *
from icvi import all_indices, graph_indices, DIRECTION
from icvi_reference import graph_indices_pattern

OUT = ROOT / 'results/v2'
N_NULL = 200
DIRECTION = dict(DIRECTION, AVI_ref=1, AVU_ref=-1, ANUI_ref=1, MQ_ref=1)
GRAPH = ['AVI', 'AVU', 'ANUI', 'MQ', 'Q', 'AVI_ref', 'AVU_ref', 'ANUI_ref', 'MQ_ref']


def indices(x, graph, labels, distances=None):
    values = all_indices(x, graph, labels, distances)
    values.update(graph_indices_pattern(graph, labels))
    return {key: float(value) for key, value in values.items()}


def rewire(graph, rng):
    c = sp.triu(graph).tocoo()
    g = ig.Graph(n=graph.shape[0], edges=list(zip(c.row.tolist(), c.col.tolist())))
    weights = rng.permutation(c.data)
    g.rewire(n=10 * g.ecount())
    e = np.array(g.get_edgelist())
    m = sp.coo_matrix((weights[:len(e)], (e[:, 0], e[:, 1])), shape=graph.shape)
    return (m + m.T).tocsr()


def run(n_null=N_NULL):
    seed = CFG['seed']
    ig.set_random_number_generator(random.Random(seed))   # igraph использует собственный генератор
    rng = np.random.default_rng(seed)
    summary = json.loads((OUT / 'summary.json').read_text())
    k = summary['selected_k']
    panel, _ = load_panel()
    arrays = build_arrays(panel)
    x, _ = scale_arrays(arrays)
    fit = fit_model(x, k)
    stored = pd.read_csv(OUT / f'types_K{k}.csv', dtype={'tid': str})
    last = stored[stored.month == stored.month.max()].set_index('tid').type.loc[arrays['ids']].to_numpy()
    assert np.array_equal(last, fit['labels'][-1]), 'метки не совпали с сохранёнными'
    region = (panel.assign(tid=panel.tid.astype(str)).drop_duplicates('tid').set_index('tid').region_name.reindex(arrays['ids']).fillna('?')).to_numpy()
    groups = [np.flatnonzero(region == r) for r in np.unique(region)]
    rows = []
    for t in (0, len(x) // 2, len(x) - 1):
        month = arrays['months'][t] if 'months' in arrays else t
        distances = cdist(x[t], x[t])
        graph = fit['graphs'][t]
        labels = fit['labels'][t]
        observed = indices(x[t], graph, labels, distances)
        null = {'labels': {key: [] for key in observed}, 'region': {key: [] for key in observed}, 'config': {key: [] for key in GRAPH}}
        for _ in range(n_null):
            shuffled = rng.permutation(labels)
            within = labels.copy()
            for ix in groups:
                within[ix] = rng.permutation(labels[ix])
            for name, lab in (('labels', shuffled), ('region', within)):
                values = indices(x[t], graph, lab, distances)
                for key in values: null[name][key].append(values[key])
            shuffled_graph = rewire(graph, rng)
            values = graph_indices(shuffled_graph, labels)
            values.update(graph_indices_pattern(shuffled_graph, labels))
            for key in GRAPH: null['config'][key].append(float(values[key]))
        for name, table in null.items():
            for key, sample in table.items():
                sample = np.array(sample); sign = DIRECTION[key]
                rows.append({'month': str(month), 'null': name, 'index': key, 'value': observed[key], 'null_mean': sample.mean(), 'null_sd': sample.std(),
                             'z': sign * (observed[key] - sample.mean()) / (sample.std() + 1e-12),
                             'p': (1 + np.sum(sign * sample >= sign * observed[key])) / (len(sample) + 1)})
        print('окно', month, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'null_models.csv', index=False)
    return table


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else N_NULL)
