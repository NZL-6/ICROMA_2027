"""Compile the supplementary document only; never train or solve experiments."""
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    source = root / "source"
    build = source / "build"
    build.mkdir(exist_ok=True)
    compiler = shutil.which("xelatex")
    if compiler is None:
        raise SystemExit("Install a TeX distribution with XeLaTeX before compiling.")
    for index in range(3):
        with (build / f"xelatex_{index + 1}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [compiler, "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory=build", "main.tex"],
                cwd=source, stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise SystemExit(f"XeLaTeX failed; inspect source/build/xelatex_{index + 1}.log")
    log = (build / "main.log").read_text(encoding="utf-8", errors="replace")
    if "undefined references" in log or "There were undefined" in log:
        raise SystemExit("Unresolved references remain; inspect source/build/main.log")
    shutil.copy2(build / "main.pdf", root / "supplementary.pdf")
    print(root / "supplementary.pdf")


if __name__ == "__main__":
    main()
