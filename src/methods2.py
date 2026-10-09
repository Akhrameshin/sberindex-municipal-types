import numpy as np, scipy.sparse as sp
from sklearn.cluster import KMeans

def dmon(X, A, K, epochs=300, seed=0, hid=64, dropout=0.2, lr=5e-3):
    """Deep Modularity Networks (Tsitsulin et al., 2023): GCN + потеря = −модулярность + коллапс-регуляризатор."""
    import torch
    torch.manual_seed(seed)
    A = sp.csr_matrix(A); A = A.maximum(A.T); n = A.shape[0]
    At = torch.tensor(A.toarray(), dtype=torch.float32); deg = At.sum(1); m = deg.sum() / 2
    Ah = At + torch.eye(n); dh = Ah.sum(1).pow(-0.5); Ah = dh[:, None] * Ah * dh[None, :]
    x = torch.tensor(X, dtype=torch.float32)
    W1 = torch.nn.Linear(x.shape[1], hid); W2 = torch.nn.Linear(hid, hid); C = torch.nn.Linear(hid, K)
    opt = torch.optim.Adam([*W1.parameters(), *W2.parameters(), *C.parameters()], lr=lr)
    for _ in range(epochs):
        h = torch.relu(Ah @ W1(x)); h = torch.nn.functional.dropout(h, dropout, True)
        h = torch.relu(Ah @ W2(h)); S = torch.softmax(C(h), 1)
        B = S.T @ At @ S - (S.T @ deg[:, None]) @ (deg[None, :] @ S) / (2 * m)
        mod = -torch.trace(B) / (2 * m)
        cs = S.sum(0); coll = cs.norm() / n * np.sqrt(K) - 1
        loss = mod + coll; opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h = torch.relu(Ah @ W1(x)); h = torch.relu(Ah @ W2(h)); return torch.softmax(C(h), 1).argmax(1).numpy()

def pooled_kmeans_viterbi(Xt, K, lam, seed=0):
    """Общие центроиды по всем месяцам + траектории меток с штрафом λ за смену (Витерби по каждому МО)."""
    pool = np.vstack(Xt); km = KMeans(K, n_init=5, random_state=seed).fit(pool)
    D = np.stack([((x[:, None, :] - km.cluster_centers_[None]) ** 2).sum(2) for x in Xt])   # T×N×K
    pen = lam * np.median(D.min(2))
    T, N, _ = D.shape; cost = D[0].copy(); back = np.zeros((T, N, K), int)
    for t in range(1, T):
        best = cost.min(1, keepdims=True); arg = cost.argmin(1)
        stay = cost; sw = best + pen
        use_stay = stay <= sw
        back[t] = np.where(use_stay, np.arange(K)[None], arg[:, None])
        cost = D[t] + np.minimum(stay, sw)
    path = np.zeros((T, N), int); path[-1] = cost.argmin(1)
    for t in range(T - 1, 0, -1): path[t - 1] = back[t][np.arange(N), path[t]]
    return path, km


def pattern_labels(shares, adjacent=False, noise=0.0, seed=0):
    """Порядково-инвариантная паттерн-кластеризация (по идее Алескерова–Мячина; реализация своя, по опубликованному описанию метода).
    Доли категорий переводятся в ранг-процентиль по всей панели (все МО × все месяцы), паттерн МО — знаки попарных сравнений
    его показателей (полный порядок: все пары; adjacent: только соседние оси). Кластер — все МО с одинаковым паттерном.
    Метрики и числа кластеров нет, результат инвариантен к любому монотонному преобразованию долей.
    shares: T×N×C. Возвращает T×N целочисленных меток; метка = код паттерна, поэтому она согласована между месяцами без выравнивания."""
    from scipy.stats import rankdata
    T, N, C = shares.shape; x = shares.reshape(-1, C)
    if noise > 0: x = x + np.random.default_rng(seed).normal(0, noise, x.shape) * x.std(0)
    r = np.stack([rankdata(x[:, c]) / len(x) for c in range(C)], 1)
    pairs = [(c, c + 1) for c in range(C - 1)] if adjacent else [(a, b) for a in range(C) for b in range(a + 1, C)]
    code = sum((r[:, a] > r[:, b]).astype(np.int64) << i for i, (a, b) in enumerate(pairs))
    return np.unique(code, return_inverse=True)[1].reshape(T, N)
