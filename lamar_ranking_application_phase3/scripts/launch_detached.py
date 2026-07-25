#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from ranking_common import write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--launch-record", required=True)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("No command supplied after --")
    log_path = Path(args.log)
    pid_path = Path(args.pid_file)
    record_path = Path(args.launch_record)
    for path in (log_path, pid_path, record_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for value in args.env:
        if "=" not in value:
            raise ValueError(value)
        key, item = value.split("=", 1)
        environment[key] = item
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            close_fds=True,
        )
    pid_path.write_text(f"{process.pid}\n")
    record = {
        "status": "LAUNCHED",
        "name": args.name,
        "pid": process.pid,
        "command": command,
        "environment_overrides": args.env,
        "log": str(log_path),
        "pid_file": str(pid_path),
    }
    write_json_new(record_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
