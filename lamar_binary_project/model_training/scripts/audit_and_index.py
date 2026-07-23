#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, gzip, json, platform, sqlite3, sys, time
from collections import Counter
from pathlib import Path

from common import SAMPLES, load_yaml, normalized_record, sequence_hash, sha256_file, write_json


REQUIRED = {"sequence_context", "label", "gene_id", "transcript_ids", "chrom", "position", "genomic_key", "leakage_group"}


def validate_row(row, expected_split, counters):
    sequence = row["sequence_context"]
    if len(sequence) != 101: counters["bad_length"] += 1
    if sequence[50] != "C": counters["bad_center"] += 1
    if row.get("split", expected_split) != expected_split: counters["bad_split_field"] += 1
    counters[f"label_{row['label']}"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config); run = Path(cfg["run_dir"]); data = Path(cfg["dataset_dir"])
    if not (data / "SUCCESS").is_file(): raise RuntimeError("Dataset SUCCESS missing")
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    if manifest["status"] != "complete": raise RuntimeError(manifest["status"])
    db_path = run / "work/train_pool.sqlite"
    if db_path.exists(): raise FileExistsError(db_path)
    connection = sqlite3.connect(db_path)
    connection.executescript("""
    pragma journal_mode=WAL; pragma synchronous=NORMAL;
    create table negatives(id integer primary key, chrom text, position integer, genomic_key text unique,
      gene_id text, gene_name text, transcript_ids text, seq text, sequence_hash text,
      leakage_group text, gc real, entropy real, c_count integer, median_depth real,
      gene_coverage real, negative_type text, matched integer, hard integer);
    create table sequences(sequence_hash text primary key, split text not null);
    """)
    pool_manifest = list(csv.DictReader((data / "train_negative_pool/manifest.tsv").open(), delimiter="\t"))
    random_shards = [Path(row["path"]) for row in pool_manifest if row["pool"] == "random_strict_all"]
    expected_pool = sum(int(row["rows"]) for row in pool_manifest if row["pool"] == "random_strict_all")
    counters = Counter(); sequence_cross_split = 0; start = time.time(); batch=[]; sequence_batch=[]
    for shard in random_shards:
        with gzip.open(shard, "rt", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = REQUIRED - set(reader.fieldnames or [])
            if missing: raise RuntimeError(f"{shard} missing {sorted(missing)}")
            for row in reader:
                validate_row(row, "train", counters)
                value = normalized_record(row); h=value["sequence_hash"]
                sequence_batch.append((h,"train"))
                batch.append((value["chrom"],value["position"],value["genomic_key"],value["gene_id"],value["gene_name"],value["transcript_ids"],value["sequence_context"],h,value["leakage_group"],value["gc_fraction"],value["entropy"],value["c_count"],value["median_depth"],value["gene_coverage"],value["negative_type"],int(row.get("sampling_pool_matched","0")),int(row.get("sampling_pool_hard","0"))))
                if len(batch) >= 5000:
                    connection.executemany("insert into negatives(chrom,position,genomic_key,gene_id,gene_name,transcript_ids,seq,sequence_hash,leakage_group,gc,entropy,c_count,median_depth,gene_coverage,negative_type,matched,hard) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",batch); connection.executemany("insert or ignore into sequences values(?,?)",sequence_batch); connection.commit(); batch=[]; sequence_batch=[]
    if batch:
        connection.executemany("insert into negatives(chrom,position,genomic_key,gene_id,gene_name,transcript_ids,seq,sequence_hash,leakage_group,gc,entropy,c_count,median_depth,gene_coverage,negative_type,matched,hard) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",batch); connection.executemany("insert or ignore into sequences values(?,?)",sequence_batch); connection.commit()
    actual_pool = connection.execute("select count(*) from negatives").fetchone()[0]
    if actual_pool != expected_pool: raise AssertionError((actual_pool,expected_pool))
    connection.executescript("create index idx_matched on negatives(matched); create index idx_hard on negatives(hard); create index idx_group on negatives(leakage_group);")
    file_counts={}; schemas={}
    files=[("train",data/"train_positives.tsv.gz"),("dev",data/"dev_1to10.tsv.gz"),("calibration",data/"calibration_1to1000.tsv.gz"),("test",data/"test_1to1000.tsv.gz")]
    for split,path in files:
        local=Counter()
        with gzip.open(path,"rt",newline="") as handle:
            reader=csv.DictReader(handle,delimiter="\t"); schemas[path.name]=reader.fieldnames
            missing=REQUIRED-set(reader.fieldnames or [])
            if missing: raise RuntimeError(f"{path} missing {sorted(missing)}")
            for row in reader:
                validate_row(row,split,local); h=sequence_hash(row["sequence_context"])
                previous=connection.execute("select split from sequences where sequence_hash=?",(h,)).fetchone()
                if previous and previous[0] != split: sequence_cross_split += 1
                connection.execute("insert or ignore into sequences values(?,?)",(h,split))
        file_counts[split]=dict(local); connection.commit()
    group_split={}; leakage_violations=0; universe=Counter()
    with gzip.open(data/"site_to_group.tsv.gz","rt",newline="") as handle:
        for row in csv.DictReader(handle,delimiter="\t"):
            group=row["leakage_group"]; split=row["split"]
            if group in group_split and group_split[group] != split: leakage_violations += 1
            group_split[group]=split; universe[(split,row["label"])]+=1
    qc=list(csv.DictReader((data/"qc_assertions.tsv").open(),delimiter="\t"))
    failures=[row for row in qc if row["status"] != "PASS"]
    result={
      "status":"PASS" if not failures and not sequence_cross_split and not leakage_violations and not counters["bad_length"] and not counters["bad_center"] else "FAIL",
      "dataset":str(data),"manifest_status":manifest["status"],"train_negative_pool_rows":actual_pool,
      "train_negative_matched":connection.execute("select count(*) from negatives where matched=1").fetchone()[0],
      "train_negative_hard":connection.execute("select count(*) from negatives where hard=1").fetchone()[0],
      "file_counts":file_counts,"universe_counts":{f"{k[0]}_label{k[1]}":v for k,v in universe.items()},
      "sequence_hash_cross_split_violations":sequence_cross_split,"leakage_group_cross_split_violations":leakage_violations,
      "dataset_qc_failures":failures,"schema_aliases":{"transcript_id":"transcript_ids","genomic_coordinate":"chrom+position/genomic_key","sequence_hash":"computed_sha256(sequence_context)","negative_type":"negative_difficulty"},
      "schemas":schemas,"index_seconds":time.time()-start,"python":platform.python_version(),
      "input_checksums":{"dataset_manifest.json":sha256_file(data/"dataset_manifest.json"),"train_positives.tsv.gz":sha256_file(data/"train_positives.tsv.gz"),"dev_1to10.tsv.gz":sha256_file(data/"dev_1to10.tsv.gz"),"calibration_1to1000.tsv.gz":sha256_file(data/"calibration_1to1000.tsv.gz"),"test_1to1000.tsv.gz":sha256_file(data/"test_1to1000.tsv.gz")},
      "test_access_scope":"integrity/schema/hash audit only; no model predictions or metrics"
    }
    connection.close(); write_json(run/"data_audit.json",result)
    if result["status"] != "PASS": raise RuntimeError(result)
    (run/"DATA_AUDIT_OK").write_text("PASS\n")
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__ == "__main__": main()
