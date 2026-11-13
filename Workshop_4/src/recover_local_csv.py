"""Offline CSV recovery with exact disk-backed deduplication and staged outputs.

Run from Workshop_4: python -m src.recover_local_csv --csv <local-file>
No network access, partition fitting, vocabulary fitting, or model training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from .data import DataPipeline, ROOT, SPLITS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recover(csv_path, pipeline=None, expected_raw=4_226_158, expected_duplicates=1_445):
    import pyarrow as pa
    import pyarrow.csv as pc
    import pyarrow.parquet as pq

    pipeline = DataPipeline() if pipeline is None else pipeline
    csv_path = Path(csv_path).resolve(strict=True)
    config = pipeline.config
    raw = Path(config["raw_path"])
    processed = Path(config["processed_dir"])
    targets = {"raw": raw, **{name: processed / f"{name}.parquet" for name in SPLITS},
               "raw_metadata": raw.parent / "dataset_meta.json",
               "raw_checksums": raw.parent / "CHECKSUMS.txt",
               "processed_metadata": processed / "recovery_report.json",
               "processed_checksums": processed / "CHECKSUMS.txt"}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Recovery stopped; existing outputs require separate verification: {existing}")
    if pipeline.splits.get("cleaning") != {"drop_malformed": True, "drop_duplicates": True}:
        raise ValueError("unexpected Workshop_1 cleaning policy")
    if config["window_size"] != 80:
        raise ValueError("expected Workshop_1 window size 80")
    protected = {Path(config[key]): sha256(config[key]) for key in ("vocab_path", "splits_path")}
    for workshop in (ROOT.parent / "Workshop_1", ROOT.parent / "Workshop_2"):
        protected.update({p: sha256(p) for p in workshop.rglob("*.py")})
    pdf = ROOT / "Workshop_4.pdf"
    if pdf.exists():
        protected[pdf] = sha256(pdf)
    source_stat = csv_path.stat()
    owner = {speaker: name for name, speakers in pipeline.groups.items() for speaker in speakers}
    schema = pa.schema([(name, pa.string()) for name in ("character_id", "x", "y")])
    output_schema = schema.append(pa.field("y_id", pa.int16()))
    counts = Counter()
    speakers = {name: set() for name in SPLITS}
    unknown_targets = Counter()
    raw_count = malformed = duplicates = 0

    with tempfile.TemporaryDirectory(prefix="csv-recovery-", dir=ROOT) as temporary:
        stage = Path(temporary)
        staged = {key: stage / key for key in targets}
        with ExitStack() as stack:
            database = sqlite3.connect(stage / "dedup.sqlite")
            stack.callback(database.close)
            # Scratch database only: limits RAM while retaining exact, collision-free keys.
            database.execute("PRAGMA journal_mode=OFF")
            database.execute("PRAGMA synchronous=OFF")
            database.execute("PRAGMA cache_size=-32768")
            database.execute("CREATE TABLE seen (speaker TEXT, x TEXT, y TEXT, PRIMARY KEY(speaker,x,y)) WITHOUT ROWID")
            cursor = database.cursor()
            writers = {key: stack.enter_context(pq.ParquetWriter(staged[key], schema if key == "raw" else output_schema,
                                                               compression="snappy")) for key in ("raw", *SPLITS)}
            reader = stack.enter_context(pc.open_csv(csv_path,
                read_options=pc.ReadOptions(block_size=8 * 1024 * 1024, use_threads=False),
                convert_options=pc.ConvertOptions(column_types={name: pa.string() for name in schema.names},
                                                  strings_can_be_null=False)))
            if reader.schema.names != schema.names:
                raise ValueError(f"CSV columns differ from Workshop_1: {reader.schema.names}")
            for batch in reader:
                if batch.schema != schema:
                    raise ValueError(f"unexpected CSV schema: {batch.schema}")
                writers["raw"].write_batch(batch)
                partitions = {name: [] for name in SPLITS}
                for row in batch.to_pylist():
                    raw_count += 1
                    speaker, x, y = (row[name] for name in schema.names)
                    if not isinstance(x, str) or not isinstance(y, str) or len(x) != 80 or len(y) != 1:
                        malformed += 1
                        continue
                    if speaker not in owner:
                        raise ValueError(f"CSV speaker absent from splits.json: {speaker!r}")
                    cursor.execute("INSERT OR IGNORE INTO seen VALUES (?,?,?)", (speaker, x, y))
                    if cursor.rowcount == 0:
                        duplicates += 1
                        continue
                    split = owner[speaker]
                    row["y_id"] = pipeline.vocab["stoi"].get(y, 0)
                    unknown_targets[split] += int(row["y_id"] == 0)
                    partitions[split].append(row)
                    speakers[split].add(speaker)
                database.commit()
                for name, rows in partitions.items():
                    if rows:
                        writers[name].write_table(pa.Table.from_pylist(rows, schema=output_schema))
                        counts[name] += len(rows)
                print(f"read={raw_count:,} kept={sum(counts.values()):,} duplicates={duplicates:,}", flush=True)

        if raw_count != expected_raw or duplicates != expected_duplicates or malformed != 0:
            raise ValueError(f"Workshop_1 baseline mismatch: raw={raw_count}, malformed={malformed}, duplicates={duplicates}")
        if sum(counts.values()) != pipeline.splits["n_rows_total"]:
            raise ValueError("total cleaned row count mismatch")
        for name in SPLITS:
            if counts[name] != pipeline.splits["counts"][name]["n_rows"] or speakers[name] != pipeline.groups[name]:
                raise ValueError(f"row/speaker count mismatch in {name}")
        overlaps = {f"{a}&{b}": len(speakers[a] & speakers[b])
                    for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
        if any(overlaps.values()):
            raise ValueError(f"shared speakers: {overlaps}")

        # Independently read every processed row using the Workshop_4 loader.
        staged_config = dict(config, raw_path=staged["raw"], processed_dir=stage)
        for name in SPLITS:
            os.link(staged[name], stage / f"{name}.parquet")
        staged_pipeline = DataPipeline(staged_config)
        verified, sample_shapes = {}, {}
        for name in SPLITS:
            total = 0
            for batch in staged_pipeline.iter_batches(name, batch_size=8192):
                sample_shapes.setdefault(name, [len(batch["x"]), len(batch["x"][0])])
                total += len(batch["x"])
            if total != counts[name]:
                raise ValueError(f"loader row count mismatch in {name}")
            verified[name] = total
            print(f"Workshop_4 loader verified {name}: {total:,}", flush=True)

        if csv_path.stat().st_size != source_stat.st_size or csv_path.stat().st_mtime_ns != source_stat.st_mtime_ns:
            raise ValueError("source CSV changed during reconstruction")
        hashes = {key: sha256(staged[key]) for key in ("raw", *SPLITS)}
        for path, original_hash in protected.items():
            if sha256(path) != original_hash:
                raise ValueError(f"protected file changed: {path}")
        generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        common = {"generated_utc": generated, "dataset": config["dataset"], "revision": config["revision"],
                  "source": {"path": str(csv_path), "size_bytes": source_stat.st_size, "sha256": sha256(csv_path)},
                  "downloaded": False, "data_contract": pipeline.contract()}
        report = {**common, "raw_rows": raw_count, "malformed_rows_removed": malformed,
                  "duplicate_rows_removed": duplicates, "processed_rows": dict(counts),
                  "speaker_counts": {name: len(ids) for name, ids in speakers.items()},
                  "pairwise_speaker_overlaps": overlaps, "unknown_target_rows": dict(unknown_targets),
                  "loader_verified_rows": verified, "sample_shapes": sample_shapes,
                  "cleaning": pipeline.splits["cleaning"],
                  "protected_sha256": {str(p): digest for p, digest in protected.items()},
                  "artifacts": {key: {"path": str(targets[key]), "sha256": hashes[key],
                                      "size_bytes": staged[key].stat().st_size} for key in hashes}}
        raw_metadata = {**common, "n_rows": raw_count, "columns": {name: "string" for name in schema.names},
                        "parquet_size_bytes": staged["raw"].stat().st_size,
                        "checksums": {raw.name: hashes["raw"]}}
        for key, value in (("raw_metadata", raw_metadata), ("processed_metadata", report)):
            staged[key].write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staged["raw_checksums"].write_text(f"{hashes['raw']}  {raw.name}\n{sha256(staged['raw_metadata'])}  dataset_meta.json\n", encoding="utf-8")
        staged["processed_checksums"].write_text("".join(f"{hashes[name]}  {name}.parquet\n" for name in SPLITS)
            + f"{sha256(staged['processed_metadata'])}  recovery_report.json\n", encoding="utf-8")
        # Hard-link publication is atomic per file and refuses any existing destination.
        for key, destination in targets.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(staged[key], destination)
        print(json.dumps({"generated": [str(p) for p in targets.values()], "rows": dict(counts),
                          "raw_rows": raw_count, "duplicates_removed": duplicates,
                          "speaker_overlaps": overlaps}, indent=2), flush=True)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    recover(args.csv)
