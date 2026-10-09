"""SHA-256 всех результатов конвейера (data/processed/*.csv, *.npy). Два чистых прогона должны дать одинаковый файл:
  python scripts/hash_outputs.py > run1.sha && ...повторный make... && python scripts/hash_outputs.py > run2.sha && diff run1.sha run2.sha
Таблицы сортируются по имени; npy хешируются вместе с формой и типом, чтобы одинаковые байты с разной формой не совпали."""
import hashlib, sys
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parent.parent / "data" / "processed"
for f in sorted(list(root.glob("*.csv")) + list(root.glob("*.npy"))):
    h = hashlib.sha256()
    if f.suffix == ".npy":
        a = np.load(f); h.update(str((a.shape, a.dtype)).encode()); h.update(np.ascontiguousarray(a).tobytes())
    else:
        h.update(f.read_bytes())
    print(f"{h.hexdigest()}  {f.name}")
