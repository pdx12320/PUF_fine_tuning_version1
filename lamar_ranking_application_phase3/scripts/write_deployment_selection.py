#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_common import write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-selection", required=True)
    parser.add_argument("--lora-selection", required=True)
    parser.add_argument("--lambda-selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lr = json.loads(Path(args.lr_selection).read_text())["selected"]
    lora = json.loads(Path(args.lora_selection).read_text())["selected"]
    hybrid = json.loads(Path(args.lambda_selection).read_text())["selected"]
    selected = {
        "kmer": lr["kmer"],
        "cnn": lr["cnn"],
        "frozen_lamar": lr["frozen_lamar"],
        "lora_lamar": lora,
        "hybrid_lamar": hybrid,
    }
    result = {
        "status": "PASS",
        "deployment_seed": 42,
        "selection_basis": (
            "pre-registered complete-dev budget metric; "
            "three-seed mean is reported, seed42 is deployed"
        ),
        "selected": selected,
        "test_access": False,
    }
    write_json_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
