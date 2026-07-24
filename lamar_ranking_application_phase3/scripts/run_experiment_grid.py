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
    python = str(master["python"])
    scripts = Path(master["run_dir"]) / "scripts"
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
        output = Path(job["output_dir"])
        summary = output / "summary.json"
        if summary.is_file():
            value = json.loads(summary.read_text())
            if value.get("status") == "SUCCESS":
                return {
                    "experiment_id": job["experiment_id"],
                    "status": "REUSED",
                    "seconds": 0.0,
                }
        if output.exists():
            raise FileExistsError(output)
        log_path = Path(job["log"])
        if log_path.exists():
            raise FileExistsError(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            gpu = gpu_queue.popleft()
        local_started = time.time()
        try:
            if job["model_type"] == "kmer":
                command = [
                    python,
                    str(scripts / "train_kmer_ranker.py"),
                    "--master",
                    str(master_path),
                    "--run-config",
                    job["config"],
                    "--output-dir",
                    job["output_dir"],
                ]
            else:
                command = [
                    python,
                    str(scripts / "train_pairwise.py"),
                    "--master",
                    str(master_path),
                    "--run-config",
                    job["config"],
                    "--output-dir",
                    job["output_dir"],
                ]
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
                    f"{job['experiment_id']} failed with "
                    f"exit code {process.returncode}; see {log_path}"
                )
            value = json.loads(summary.read_text())
            if value.get("status") != "SUCCESS":
                raise RuntimeError(value)
            return {
                "experiment_id": job["experiment_id"],
                "status": "SUCCESS",
                "gpu": gpu,
                "seconds": time.time() - local_started,
            }
        finally:
            with lock:
                gpu_queue.append(gpu)

    jobs = manifest["jobs"]
    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(gpu_queue)
    ) as executor:
        futures = {
            executor.submit(run_job, job): job for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    run_result = {
        "status": "SUCCESS",
        "manifest": str(manifest_path),
        "jobs": sorted(
            results, key=lambda value: value["experiment_id"]
        ),
        "seconds": time.time() - started,
    }
    status_path = manifest_path.with_name(
        manifest_path.stem + ".run.json"
    )
    write_json_new(status_path, run_result)
    print(json.dumps(run_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
