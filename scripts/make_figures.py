"""Рисунки для статьи: все строятся из results/v2 и подготовленных входов, ничего не вводится вручную."""
from pathlib import Path
import sys, json, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from atlas_v2 import *

OUT = ROOT / 'results/v2'; FIG = ROOT / 'report/figures'; FIG.mkdir(parents=True, exist_ok=True)
COL = ['#2d7662', '#dc982d', '#477db4', '#985e98', '#bd6048', '#586da3']
INK, MUTED, GRID = '#172c26', '#5e7167', '#d9e0d6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlecolor': INK})
comma = lambda v, d=2: f'{v:.{d}f}'.replace('.', ',')


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches='tight', facecolor='white'); plt.close(fig); print('рисунок', name, flush=True)


def parse_path(d):
    """Подпути SVG-строки вида «M x y L x y … Z» → вершины и коды matplotlib."""
    verts, codes = [], []
    for sub in re.findall(r'M[^M]*', d):
        pts = [tuple(map(float, p.split())) for p in re.findall(r'[-\d.]+ [-\d.]+', sub)]
        if len(pts) < 3: continue
        verts += pts + [pts[0]]; codes += [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1) + [MPath.CLOSEPOLY]
    return verts, codes


def fig_map(info, k):
    geo = json.loads((ROOT / 'assets/geography.json').read_text()); p, mapping = load_panel(); a = build_arrays(p)
    z = np.load(OUT / f'model_K{k}.npz'); lab = z['labels'][-1]; names = info['models'][str(k)]['names']
    mun = dict(geo['municipalities'])
    for tid, group in mapping.groupby('analysis_tid'):
        parts = [geo['municipalities'][str(t)]['path'] for t in group.source_tid if str(t) in geo['municipalities']]
        if parts: mun[tid] = {'path': ''.join(parts)}
    fig, ax = plt.subplots(figsize=(8.6, 4.9)); ax.set_aspect('equal'); ax.axis('off')
    for tid, d in mun.items():          # фон: территории без типа (неполные ряды)
        if tid not in a['ids']:
            v, c = parse_path(d['path'])
            if v: ax.add_patch(PathPatch(MPath(v, c), facecolor='#e7ebe5', edgecolor='white', lw=.15))
    for i, tid in enumerate(a['ids']):
        v, c = parse_path(mun[tid]['path']) if tid in mun else ([], [])
        if v: ax.add_patch(PathPatch(MPath(v, c), facecolor=COL[int(lab[i])], edgecolor='white', lw=.15))
    ax.set_xlim(0, geo['width']); ax.set_ylim(geo['height'], 0)
    handles = [plt.Line2D([0], [0], marker='s', ls='', color=COL[t], markersize=8, label=f'{t+1}. {names[str(t)]} ({int((lab==t).sum())})') for t in range(k)]
    handles.append(plt.Line2D([0], [0], marker='s', ls='', color='#e7ebe5', markersize=8, label=f'без типа: неполные ряды ({info["incomplete_n"]})'))
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=7.5, ncol=2, bbox_to_anchor=(0, 0), columnspacing=1.5)
    save(fig, 'fig_map.png')


def fig_profiles(info, k):
    p, _ = load_panel(); a = build_arrays(p); lab = np.load(OUT / f'model_K{k}.npz')['labels'][-1]; names = info['models'][str(k)]['names']
    X = np.c_[a['spending'][-1], np.where(np.isfinite(a['context']), a['context'], np.nan)]
    labels = SPEND_LABELS + CONTEXT_LABELS; med = np.nanmedian(X, axis=0); sd = np.nanstd(X, axis=0) + 1e-12
    Z = np.array([np.nanmean((X[lab == t] - med) / sd, axis=0) for t in range(k)]).T
    fig, ax = plt.subplots(figsize=(8.2, 6.4)); im = ax.imshow(Z, cmap='PuOr_r', vmin=-1.6, vmax=1.6, aspect='auto')
    ax.set_xticks(range(k)); ax.set_xticklabels([f'{t+1}\n(n={int((lab==t).sum())})' for t in range(k)]); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7.5)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]): ax.text(j, i, comma(Z[i, j], 1), ha='center', va='center', fontsize=7, color='white' if abs(Z[i, j]) > 1 else INK)
    ax.set_title('Профили типов: отклонение среднего по типу от медианы всех территорий, в σ'); ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02); cb.set_label('σ'); cb.outline.set_visible(False)
    save(fig, 'fig_profiles.png')


def fig_k_selection():
    s = pd.read_csv(OUT / 'k_selection.csv'); fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    col = [COL[0] if e else '#b9c3bb' for e in s.eligible]
    ax[0].bar(s.K.astype(str), s.ARI_mean, color=col); ax[0].axhline(.85, color=INK, ls='--', lw=.9); ax[0].text(3.45, .857, 'порог 0,85', ha='right', fontsize=7.5, color=INK)
    for x, v in zip(s.K.astype(str), s.ARI_mean): ax[0].text(x, v + .006, comma(v, 3), ha='center', fontsize=8)
    ax[0].set_ylim(.6, 1); ax[0].set_xlabel('число типов K'); ax[0].set_ylabel('средний ARI, 10 пересборок'); ax[0].set_title('Устойчивость разбиения')
    ax[1].bar(s.K.astype(str), s.min_share * 100, color=col); ax[1].axhline(3, color=INK, ls='--', lw=.9); ax[1].text(3.45, 3.5, 'порог 3%', ha='right', fontsize=7.5, color=INK)
    for x, v in zip(s.K.astype(str), s.min_share * 100): ax[1].text(x, v + .5, comma(v, 1) + '%', ha='center', fontsize=8)
    ax[1].set_xlabel('число типов K'); ax[1].set_ylabel('доля наименьшего типа, %'); ax[1].set_title('Размер наименьшего типа')
    fig.tight_layout(); save(fig, 'fig_k_selection.png')


def fig_seed_agreement():
    b = pd.read_csv(OUT / 'baselines.csv').sort_values('seed_ARI'); fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(b.method, b.seed_ARI, color=[COL[1] if m == 'GS-TKM' else '#9aa5b1' for m in b.method]); ax.set_xlim(0, 1.04)
    for i, v in enumerate(b.seed_ARI): ax.text(v + .008, i, comma(v, 2), va='center', fontsize=8)
    ax.axvline(1, color=MUTED, lw=.6); ax.set_xlabel('ARI между запусками с seed 42, 43, 44 (последнее окно; 1 — запуски совпадают)')
    save(fig, 'fig_seed_agreement.png')


def fig_bootstrap(k):
    r = pd.read_csv(OUT / 'robustness.csv'); r = r[r.K == k]; fig, ax = plt.subplots(figsize=(8, 3.1)); y = np.arange(len(r))[::-1]
    ax.hlines(y, r.Jaccard_low, r.Jaccard_high, color=COL[0], lw=3); ax.plot(r.Jaccard_mean, y, 'o', color=INK, ms=6)
    ax.set_yticks(y); ax.set_yticklabels([f'{int(t)+1}. {n}' for t, n in zip(r.type, r.name)], fontsize=8)
    for yy, m in zip(y, r.Jaccard_mean): ax.text(m, yy + .22, comma(m, 3), ha='center', fontsize=7.5)
    ax.set_xlabel('индекс Жаккара типа при пересборках: среднее и квантили 2,5–97,5%'); ax.set_xlim(.7, 1)
    save(fig, 'fig_bootstrap.png')


LAB = {'spending_weight': 'вес блока расходов', 'neighbors': 'число соседей k', 'smoothing_steps': 'шагов сглаживания', 'transition_penalty': 'штраф за смену',
       'seed': 'seed', 'unreported_coordinate': 'нераспределённая занятость', 'city_policy': 'Москва и Петербург', 'window_months': 'длина окна, мес.', 'window_centering': 'центрирование по окнам'}
VAL = {'excluded': 'исключена', 'original': '247 МО отдельно', 'drop': 'исключены', 'level': 'только уровень', 'all_spending': 'все координаты'}


def fig_sensitivity():
    s = pd.read_csv(OUT / 'sensitivity.csv'); s = s[~((s.factor == 'seed') & (s.value.astype(str) != '0') & (s.value.astype(str) != '3'))]
    s = s[~((s.ARI > .9999) & (s.factor != 'seed'))].copy(); s['lab'] = [f'{LAB[f]}: {VAL.get(str(v), str(v))}' for f, v in zip(s.factor, s.value)]
    s = s.sort_values('ARI'); fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.4), sharey=True)
    ax[0].barh(s.lab, s.ARI, color='#7fa89b'); ax[0].set_xlabel('ARI к основной типологии'); ax[0].set_xlim(0, 1)
    for i, v in enumerate(s.ARI): ax[0].text(v + .01, i, comma(v, 2), va='center', fontsize=7.5)
    ax[1].barh(s.lab, s.changed_share * 100, color='#d9a65a'); ax[1].set_xlabel('сменили тип за год, % территорий')
    for i, v in enumerate(s.changed_share * 100): ax[1].text(v + .4, i, comma(v, 1), va='center', fontsize=7.5)
    ax[1].tick_params(axis='y', length=0); fig.tight_layout(); save(fig, 'fig_sensitivity.png')


def fig_drift(info, k):
    p, _ = load_panel(); a = build_arrays(p); x, _ = scale_arrays(a); f = fit_model(x, k)
    xc, _ = scale_arrays(centre_windows(a)); fc = fit_model(xc, k)
    names = info['models'][str(k)]['names']; months = [str(m)[:7] for m in a['months']]
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.4), sharey=True)
    for j, (title, fit) in enumerate([('Основная модель', f), ('Координаты расходов центрированы по окнам', fc)]):
        sizes = np.array([np.bincount(l, minlength=k) for l in fit['labels']])
        for t in range(k): ax[j].plot(range(len(months)), sizes[:, t], color=COL[t], lw=2, label=f'{t+1}. {names[str(t)]}')
        ax[j].set_title(title); ax[j].set_xticks([0, 4, 8, 12]); ax[j].set_xticklabels([months[i] for i in [0, 4, 8, 12]]); ax[j].set_xlabel('конец окна')
        ch = (fit['labels'][0] != fit['labels'][-1]).mean(); ax[j].text(.02, .96, f'сменили тип за год: {comma(ch*100,1)}%', transform=ax[j].transAxes, fontsize=8, va='top', color=INK)
    ax[0].set_ylabel('число территорий'); h, l = ax[0].get_legend_handles_labels()
    fig.tight_layout(); fig.legend(h, l, loc='upper center', bbox_to_anchor=(.5, 0), ncol=2, frameon=False, fontsize=7.5); save(fig, 'fig_drift.png')


def main():
    info = json.loads((OUT / 'summary.json').read_text()); k = info['selected_k']
    fig_map(info, k); fig_profiles(info, k); fig_k_selection(); fig_seed_agreement(); fig_bootstrap(k); fig_sensitivity(); fig_drift(info, k)


if __name__ == '__main__':
    main()
