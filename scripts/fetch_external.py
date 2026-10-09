"""Берёт реализации KEFRiN и CANUS авторов (Shalileh & Mirkin, 2022) на закреплённом коммите.

В репозиторий код не включён: у авторов нет файла лицензии, поэтому перераспространять его нельзя.
Нужен ровно один файл: kefrin.py самодостаточен (numpy, sklearn, stdlib), остальное в том репозитории
к делу не относится. Раньше здесь был git clone всего репозитория — 352 МБ ради 23 КБ, и именно он
обрывался на полпути («unexpected disconnect while reading sideband packet»). Теперь файл качается
по raw-ссылке с закреплённого коммита и сверяется с закреплённым SHA-256: и быстрее, и проверяемо.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_data import _download, sha                      # общий загрузчик: stdlib + ретраи

# репозиторий, закреплённый коммит, файлы с SHA-256, каталог назначения. У обоих репозиториев нет файла лицензии: в наш репозиторий код не включаем.
# CANUS (Shalileh, 2025, IEEE Access): canus.py самодостаточен (numpy + torch); остальное в репозитории — наборы данных на ~100 МБ, они не нужны.
SOURCES = [("KEFRiN", "Sorooshi/KEFRiN", "f9f96b1a778cb8d2e5ba85baae452dc813c4f110",
            {"kefrin.py": "4616134cd0f427a62e561b3b5c315196be90440aa473dcec191c27fd5a3cc175"}, Path("external/KEFRiN")),
           ("CANUS", "Sorooshi/CANUS", "754622af6eff9604a590e862a108c4c4212acf2d",
            {"canus.py": "dbc14daff4f6b2a4b9d377deb1bef5c76a2c390678d47922fa337983e0faa7a3"}, Path("external/CANUS"))]
RETRIES = 3


def fetch(title, repo, commit, files, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for name, want in files.items():
        p = dst / name
        if p.exists() and sha(p) == want:
            print(f"{title} {name}: уже на месте, SHA-256 совпадает")
            continue
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{name}"
        for attempt in range(1, RETRIES + 1):
            try:
                _download(url, p)
                got = sha(p)
                if got != want:
                    p.unlink()
                    print(f"\n{title} {name}: SHA-256 не совпал\n  ожидался {want}\n  получен  {got}", file=sys.stderr)
                    return 3
                print(f"{title} {name} @ {commit[:8]}: SHA-256 совпал")
                break
            except Exception as e:
                print(f"  попытка {attempt}/{RETRIES}: {type(e).__name__}: {e}")
                if p.exists(): p.unlink()
        else:
            print(f"\n{title} {name}: не удалось скачать после {RETRIES} попыток.\n"
                  "Нужен только для сравнения методов (make compare); остальной конвейер от него не зависит.", file=sys.stderr)
            return 4
    return 0


def main():
    for src in SOURCES:
        rc = fetch(*src)
        if rc: return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
