import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def executable_path(name):
    local_executable = Path(sys.prefix) / "bin" / name
    if local_executable.is_file():
        return str(local_executable)
    return shutil.which(name)


def main():
    parser = argparse.ArgumentParser(description="Add searchable OCR text to PDFs.")
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="PDF files or directories to search recursively",
    )
    parser.add_argument("--language", default="eng")
    args = parser.parse_args()

    ocrmypdf = executable_path("ocrmypdf")
    if not ocrmypdf:
        parser.error("OCRmyPDF is missing; install requirements.txt first")
    missing_tools = [tool for tool in ("tesseract", "gs") if not shutil.which(tool)]
    if missing_tools:
        parser.error(f"Missing system tools: {', '.join(missing_tools)}")

    pdf_paths = set()
    for source in args.sources:
        if source.is_file():
            if source.suffix.lower() != ".pdf":
                parser.error(f"Source file is not a PDF: {source}")
            pdf_paths.add(source.resolve())
        elif source.is_dir():
            pdf_paths.update(
                path.resolve()
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() == ".pdf"
            )
        else:
            parser.error(f"Source does not exist: {source}")

    pdfs = sorted(pdf_paths)
    if not pdfs:
        parser.error("No PDFs found in the supplied sources")

    failures = []
    for pdf in pdfs:
        print(f"OCR: {pdf}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix=".ocr-", dir=pdf.parent) as temp_dir:
                output = Path(temp_dir) / pdf.name
                subprocess.run(
                    [
                        ocrmypdf,
                        "--skip-text",
                        "--output-type",
                        "pdf",
                        "--optimize",
                        "0",
                        "--language",
                        args.language,
                        str(pdf),
                        str(output),
                    ],
                    check=True,
                )
                if not output.is_file():
                    raise RuntimeError("OCRmyPDF did not produce an output PDF")
                os.replace(output, pdf)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            failures.append((pdf, exc))
            print(f"FAILED: {pdf}: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(f"OCR failed for {len(failures)} of {len(pdfs)} PDFs", file=sys.stderr)
        return 1

    print(f"OCR complete: {len(pdfs)} PDFs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())