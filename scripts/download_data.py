"""Загрузка сырых данных. Все источники открытые, регистрация не нужна.
Контрольные суммы пишутся в data/raw/SHA256SUMS и в корень репозитория, сверяются при --verify.

ВАЖНО про TLS. Сертификат www.sberbank.com выдан «Russian Trusted Root CA» (Минцифры РФ): после
отказа западных центров сертификации выдавать сертификаты подсанкционным банкам Сбер перешёл на
национальный корень. Этого корня нет ни в certifi, ни в хранилище macOS
по умолчанию, поэтому загрузка падает с CERTIFICATE_VERIFY_FAILED, хотя сеть доступна. Росстат
отдаётся через Яндекс по GlobalSign и качается без проблем.

Три способа это пройти, в порядке предпочтительности:
  1. Установить корневой сертификат Минцифры в системное хранилище (https://www.gosuslugi.ru/crt).
     Тогда он подхватывается автоматически: ниже включается truststore, который берёт доверие из ОС.
  2. Указать свой набор корней: REQUESTS_CA_BUNDLE=/путь/к/bundle.pem
  3. Запустить с --trust-sha256: файл скачивается без проверки TLS, но обязан совпасть с
     версионированной контрольной суммой из SHA256SUMS. Доверие переносится с канала на хеш:
     подменённый файл будет отвергнут и удалён. Без записи в манифесте этот режим файл не примет.
"""
import hashlib, shutil, ssl, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

try:                      # доверие из хранилища ОС вместо вшитого certifi (Python 3.10+)
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CFG
from rangezip import open_zip

RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
SB = "https://www.sberbank.com/common"
FILES = {"hackathonlicence.zip": f"{SB}/img/uploaded/files/pdf/sberindex/hackathonlicence.zip",
         "t_dict_municipal.rar": f"{SB}/files/t_dict_municipal.rar",
         "indeks-mobilnosti.parquet": "https://sberindex.ru/api/dataset/v1/download/indeks-mobilnosti/parquet"}
ROSSTAT = "https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/data_bdmo_118_v20250918/by_indicator/data_section{}_112_v20250918.zip"
INDICATORS = {"Y48112027": 31, "Y48423005": 32, "Y48423007": 32}      # население, занятость и зарплата по ОКВЭД2


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


CA_HINT = """
  Сертификат {host} подписан национальным корнем РФ («Russian Trusted Root CA»),
  которого нет в certifi. Сеть при этом работает. Варианты:
    1. установить корень Минцифры в хранилище ОС (https://www.gosuslugi.ru/crt) и повторить;
    2. REQUESTS_CA_BUNDLE=/путь/к/bundle.pem make data
    3. make data-sha256   — без проверки TLS, но с обязательной сверкой SHA-256 из SHA256SUMS
"""
UA = "sberindex-cluster/1.0 (reproducibility pipeline)"


def expected_sums():
    """Версионированные контрольные суммы: корень репозитория — эталон, копия в data/raw — рабочая."""
    for f in (Path("SHA256SUMS"), RAW / "SHA256SUMS"):
        if f.exists():
            return dict(l.split("  ", 1)[::-1] for l in f.read_text().splitlines() if "  " in l)
    return {}


def _download(url, dest, ctx=None, timeout=600):
    """Качает через stdlib, а не requests: urllib3 строго сверяет Content-Length и падает на
    теле редиректа API СберИндекса (IncompleteRead), а в режиме verify=False теряет SNI."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r, open(dest, "wb") as out:
        shutil.copyfileobj(r, out, 1 << 20)


def _verify_sha(p, name, want, where):
    if want is None:
        print(f"  скачан ({where}); в манифесте записи нет, сверять не с чем")
        return p
    got = sha(p)
    if got != want:
        p.unlink()
        print(f"\nSHA-256 не совпал для {name}:\n  ожидался {want}\n  получен  {got}\n"
              "Файл удалён.", file=sys.stderr)
        raise SystemExit(3)
    print(f"  SHA-256 совпал с манифестом ({where})")
    return p


def fetch(name, url, trust_sha=False, retries=3):
    p = RAW / name
    if p.exists():
        return _verify_sha(p, name, expected_sums().get(name), "существующий файл")
    print("скачиваю", name)
    want = expected_sums().get(name)
    host = url.split("/")[2]
    cert_error = None
    for attempt in range(1, retries + 1):
        try:
            _download(url, p)                                   # строгая проверка сертификата
            return _verify_sha(p, name, want, "проверенный канал")
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
                cert_error = e; break                           # повторы тут не помогут
            print(f"  попытка {attempt}/{retries}: {e.reason}")
        except Exception as e:
            print(f"  попытка {attempt}/{retries}: {type(e).__name__}: {e}")
        if p.exists(): p.unlink()
        if attempt < retries: time.sleep(2 * attempt)
    if cert_error is None:
        print(f"\n{name}: загрузка не удалась после {retries} попыток.", file=sys.stderr)
        raise SystemExit(4)
    if not trust_sha:
        print(f"\nTLS: не удалось проверить сертификат {host}.{CA_HINT.format(host=host)}", file=sys.stderr)
        raise SystemExit(2) from cert_error
    if not want:
        print(f"\n--trust-sha256 отклонил {name}: в SHA256SUMS нет записи, сверять не с чем.", file=sys.stderr)
        raise SystemExit(2) from cert_error
    print(f"  сертификат {host} не проверен; целостность проверим по SHA-256 из манифеста")
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    for attempt in range(1, retries + 1):
        try:
            _download(url, p, ctx)
            return _verify_sha(p, name, want, "непроверенный канал")
        except Exception as e:
            print(f"  попытка {attempt}/{retries}: {type(e).__name__}: {e}")
            if p.exists(): p.unlink()
            if attempt < retries: time.sleep(2 * attempt)
    print(f"\n{name}: загрузка не удалась после {retries} попыток.", file=sys.stderr)
    raise SystemExit(4)


def main(verify=False, trust_sha=False):
    for n, u in FILES.items(): fetch(n, u, trust_sha)
    if not (RAW / "hack").exists(): subprocess.run(["bsdtar", "-xf", str(RAW / "hackathonlicence.zip"), "-C", str(RAW / "hack")], check=True) if (RAW / "hack").mkdir() is None else None
    if not (RAW / "dict").exists(): (RAW / "dict").mkdir(); subprocess.run(["bsdtar", "-xf", str(RAW / "t_dict_municipal.rar"), "-C", str(RAW / "dict")], check=True)
    (RAW / "rosstat").mkdir(exist_ok=True)
    for code, sec in INDICATORS.items():
        z = None
        for y in CFG["data"]["rosstat_years_download"]:
            out = RAW / f"rosstat/{code}_{y}.csv"
            if out.exists(): continue
            z = z or open_zip(ROSSTAT.format(sec)); print("rosstat", code, y)
            out.write_bytes(z.read(f"data_{code}_parts/data_{code}_year{y}_112_v20250918.csv"))
    # SHA256SUMS в корне репозитория версионируется: по нему можно проверить, что скачаны те же файлы.
    # data/ целиком в .gitignore, поэтому копия внутри data/raw — рабочая, а эталон — в корне.
    sums, tracked = RAW / "SHA256SUMS", Path("SHA256SUMS")
    files = sorted([p for p in RAW.glob("*") if p.is_file() and p.name != "SHA256SUMS"] + list((RAW / "rosstat").glob("*.csv")))
    cur = {str(p.relative_to(RAW)): sha(p) for p in files}
    ref = tracked if tracked.exists() else sums
    if verify and ref.exists():
        old = dict(l.split("  ", 1)[::-1] for l in ref.read_text().splitlines())
        bad = [k for k, v in cur.items() if old.get(k) not in (None, v)]
        miss = [k for k in old if k not in cur]
        print(f"контрольные суммы ({ref}):", "OK" if not bad and not miss else f"РАСХОЖДЕНИЕ изменены={bad} отсутствуют={miss}")
        return 1 if (bad or miss) else 0
    body = "\n".join(f"{v}  {k}" for k, v in cur.items()) + "\n"
    sums.write_text(body)
    if not tracked.exists(): tracked.write_text(body)
    print("записан рабочий SHA256SUMS; существующий эталон сохранён,", len(cur), "файлов")
    return 0


if __name__ == "__main__":
    sys.exit(main("--verify" in sys.argv, "--trust-sha256" in sys.argv) or 0)
