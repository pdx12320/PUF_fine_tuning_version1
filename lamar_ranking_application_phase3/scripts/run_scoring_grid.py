#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from ranking_common import load_yaml, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    master_path = Path(args.master)
    master = load_yaml(master_path)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    scripts = Path(master["run_dir"]) / "scripts"
    python = str(master["python"])
    gpu_queue = deque(
        value.strip()
        for value in args.gpus.split(",")
        if value.strip()
    )
    if not gpu_queue:
        raise ValueError("At least one GPU is required")
    lock = threading.Lock()
    started = time.time()

    def run_job(job):
        output = Path(job["output"])
        output_manifest = output.with_suffix(".manifest.json")
        if output.is_file() and output_manifest.is_file():
            value = json.loads(output_manifest.read_text())
            if value.get("status") == "PASS":
                return {
                    "model_id": job["model_id"],
                    "status": "REUSED",
                    "seconds": 0.0,
                }
        if output.exists() or output_manifest.exists():
            raise FileExistsError(output)
        log_path = Path(job["log"])
        if log_path.exists():
            raise FileExistsError(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            gpu = gpu_queue.popleft()
        local_started = time.time()
        try:
            command = [
                python,
                str(scripts / "score_candidates.py"),
                "--master",
                str(master_path),
                "--model-type",
                job["model_type"],
                "--input",
                job["input"],
                "--output",
                job["output"],
                "--model-id",
                job["model_id"],
                "--seed",
                str(job.get("seed", 42)),
                "--batch-size",
                str(job.get("batch_size", 256)),
            ]
            if job.get("model_config"):
                command.extend(
                    ["--model-config", job["model_config"]]
                )
            if job.get("checkpoint"):
                command.extend(
                    ["--checkpoint", job["checkpoint"]]
                )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONPATH"] = str(scripts)
            with log_path.open("x") as log:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    check=False,
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"{job['model_id']} failed with exit "
                    f"{process.returncode}; see {log_path}"
                )
            return {
                "model_id": job["model_id"],
                "status": "SUCCESS",
                "gpu": gpu,
                "seconds": time.time() - local_started,
            }
        finally:
            with lock:
                gpu_queue.append(gpu)

    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(gpu_queue)
    ) as executor:
        futures = {
            executor.submit(run_job, job): job
            for job in manifest["jobs"]
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    run_result = {
        "status": "SUCCESS",
        "manifest": str(manifest_path),
        "jobs": sorted(results, key=lambda row: row["model_id"]),
        "seconds": time.time() - started,
    }
    write_json_new(
        manifest_path.with_name(
            manifest_path.stem + ".run.json"
        ),
        run_result,
    )
    print(json.dumps(run_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
