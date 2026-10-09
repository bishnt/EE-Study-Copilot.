import sys
from pathlib import Path

import pandas as pd

REQUIRED = {
    "file",
    "subject",
    "category",
    "title",
    "year",
    "license",
    "source_url",
    "shareable",
}
SUBJECTS = {"dsp", "dcs", "swp", "emd", "tht", "pe"}


def main(root="corpus", manifest="corpus/manifest.csv", min_per_subject=5):
    root_path = Path(root)
    manifest_path = Path(manifest)

    if not manifest_path.is_file():
        sys.exit(f"Manifest not found: {manifest_path}")

    try:
        df = pd.read_csv(manifest_path)
    except (OSError, pd.errors.ParserError) as exc:
        sys.exit(f"Could not read manifest {manifest_path}: {exc}")

    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        sys.exit(f"Missing columns: {missing}")

    errors = []
    manifest_files = set()
    for i, row in df.iterrows():
        subject = row["subject"]
        filename = row["file"]

        if pd.isna(subject) or subject not in SUBJECTS:
            errors.append(f"Row {i + 2}: bad subject {subject}")
            continue
        if pd.isna(filename) or not str(filename).strip():
            errors.append(f"Row {i + 2}: file is missing")
            continue

        filename = str(filename)
        manifest_files.add((subject, filename))
        path = root_path / subject / filename
        if not path.is_file():
            errors.append(f"Row {i + 2}: file not found: {path}")
        if row["shareable"] not in ("yes", "no"):
            errors.append(f"Row {i + 2}: shareable must be yes or no")

    on_disk = {
        (pdf.parent.relative_to(root_path).as_posix(), pdf.name)
        for pdf in root_path.rglob("*.pdf")
        if pdf.is_file()
    }
    for subject, filename in sorted(on_disk - manifest_files):
        errors.append(f"File not in manifest: {subject}/{filename}")

    counts = df[df["subject"].isin(SUBJECTS)].groupby("subject").size()
    for subject in sorted(SUBJECTS):
        count = counts.get(subject, 0)
        if count < min_per_subject:
            errors.append(
                f"Subject {subject} has {count} documents; need {min_per_subject}"
            )

    if errors:
        sys.exit("\n".join(errors))

    print(f"OK: {len(df)} documents")
    print(counts)


if __name__ == "__main__":
    main()
