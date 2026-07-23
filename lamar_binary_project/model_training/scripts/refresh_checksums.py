#!/usr/bin/env python3
from pathlib import Path
from common import sha256_file

RUN = Path("${LAMAR_WORK_ROOT}/lamar_binary_models/run_20260722T203752Z")
rows = []
for path in sorted(RUN.rglob("*")):
    if path.is_file() and path.name not in {"checksums.sha256", "checksums.sha256.tmp", "SUCCESS"}:
        rows.append(f"{sha256_file(path)}  {path.relative_to(RUN)}")
temporary = RUN / "checksums.sha256.tmp"
temporary.write_text("\n".join(rows) + "\n")
temporary.replace(RUN / "checksums.sha256")
print(f"refreshed {len(rows)} checksums")
