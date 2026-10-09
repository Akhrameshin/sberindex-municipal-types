"""Сборка PDF статьи: report/methodology.md → LaTeX; для публикации используется PDF, собранный в Overleaf (report/methodology.pdf).
Нужны pandoc и tectonic (на macOS: brew install pandoc tectonic). Шрифты STIX Two (в macOS по умолчанию) и Menlo.
  python scripts/build_pdf.py            # PDF локально
  python scripts/build_pdf.py --overleaf # проект для Overleaf: report/overleaf.zip (main.tex + figures/)"""
import re, subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"; LATEX = REPORT / "latex"
REPO = "https://github.com/Akhrameshin/sberindex-municipal-types/blob/main/"


def split_front(md):
    parts = md.split("\n---\n", 1)
    front, body = parts[0], parts[1]
    title = re.search(r"^# (.+)$", front, re.M).group(1).strip()
    paras = [p.strip() for p in front.split("\n\n") if p.strip()]
    authors = next(p for p in paras if p.startswith("**") and "Аннотация" not in p and "Ключевые" not in p).strip("*")
    venue = next(p for p in paras if p.startswith("*") and not p.startswith("**")).strip("*")
    grab = lambda key: next(p for p in paras if p.startswith(f"**{key}")).split("**", 2)[2].strip()
    return dict(title=title, authors=authors, venue=venue, abstract_ru=grab("Аннотация")), body


def prepare_body(body):
    body = re.sub(r"\]\(\.\./([^)]+)\)", lambda m: "](" + REPO + m.group(1) + ")", body)          # ссылки на файлы репозитория — на GitHub
    body = re.sub(r"^(#{2,3}) (\d+(?:\.\d+)*)\. ", r"\1 ", body, flags=re.M)                       # номера разделов ставит LaTeX
    body = re.sub(r"^(## Приложение [A-Z]\..*)$", r"\1 {.unnumbered}", body, flags=re.M)
    body = body.replace("10⁻²", "1e-2")
    body = re.sub(r"```.*?```", lambda m: m.group(0).replace("≤", "<=").replace("→", "->"), body, flags=re.S)       # в моноширинных блоках — ASCII-стрелки, чтобы колонки не съезжали
    body = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "\u00a0", body)                                         # «9 566» не переносится
    return body.strip() + "\n"


def main():
    overleaf = "--overleaf" in sys.argv
    for tool in ("pandoc", "tectonic"):
        if not shutil.which(tool): sys.exit(f"нужен {tool}: brew install pandoc tectonic")
    md = (REPORT / "methodology.md").read_text(encoding="utf-8")
    meta, body = split_front(md)
    (LATEX / "body.md").write_text(prepare_body(body), encoding="utf-8")
    y = "\n".join(f"{k}: |\n  " + v.replace("\n", "\n  ") for k, v in meta.items()) + "\n"
    (LATEX / "meta.yaml").write_text("---\n" + y + "---\n", encoding="utf-8")
    out = "overleaf/main.tex" if overleaf else "paper_local.tex"
    (REPORT / "overleaf").mkdir(exist_ok=True) if overleaf else None
    cmd = ["pandoc", "-f", "markdown-smart+tex_math_dollars+pipe_tables", "-t", "latex", "-s", "--template", "latex/template_overleaf.tex" if overleaf else "latex/template.tex", "--lua-filter", "latex/filter.lua",
           "--metadata-file", "latex/meta.yaml", "--wrap=preserve", "--no-highlight", "--shift-heading-level-by=-1", "--resource-path", ".", "latex/body.md", "-o", out]
    subprocess.run(cmd, cwd=REPORT, check=True)
    if overleaf:
        import zipfile
        main = REPORT / "overleaf" / "main.tex"
        main.write_text("% !TEX program = xelatex\n% !TEX encoding = UTF-8\n" + main.read_text(encoding="utf-8"), encoding="utf-8")   # Overleaf сам выберет XeLaTeX
        (REPORT / "overleaf" / "latexmkrc").write_text("$pdf_mode = 5;\n$dvi_mode = 0;\n$postscript_mode = 0;\n", encoding="utf-8")        # запасной способ выбора XeLaTeX
        shutil.copytree(REPORT / "figures", REPORT / "overleaf" / "figures", dirs_exist_ok=True)
        subprocess.run(["tectonic", "--keep-logs", "main.tex"], cwd=REPORT / "overleaf", check=True)       # проверка, что проект собирается без системных шрифтов
        with zipfile.ZipFile(REPORT / "overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(REPORT / "overleaf" / "main.tex", "main.tex"); z.write(REPORT / "overleaf" / "latexmkrc", "latexmkrc")
            for f in sorted((REPORT / "figures").glob("*.png")): z.write(f, "figures/" + f.name)
        print("готово:", REPORT / "overleaf.zip")
        return
    subprocess.run(["tectonic", "--keep-logs", "paper_local.tex"], cwd=REPORT, check=True)
    print("готово:", REPORT / "paper_local.pdf")


if __name__ == "__main__":
    main()
