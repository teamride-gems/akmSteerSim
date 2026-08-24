#!/usr/bin/env python3
"""Run the synthetic stands or ground pilot; never a scientific bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.adapters import make_adapter
from hardware_study.execution import execute_run
from hardware_study.integrity import sha256_file, write_json
from hardware_study.pilot import engineering_pilot_bundle
from hardware_study.validation import validate_and_write
from scripts.run_hardware_study import (
    assert_no_placeholders,
    start_rosbag,
    stop_rosbag,
    verify_freeze,
    verify_preflight,
    verify_prepared,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stands", "ground"), required=True)
    parser.add_argument("--adapter", choices=("mock", "ros2"), required=True)
    parser.add_argument("--config", default="configs/hardware_study_v1.yaml")
    parser.add_argument("--site")
    parser.add_argument(
        "--prepared-dir",
        default="reproducibility/hardware_validation/study_v1/prepared",
    )
    parser.add_argument(
        "--freeze", default="reproducibility/hardware_validation/study_v1/FREEZE.json"
    )
    parser.add_argument("--preflight")
    parser.add_argument("--arm", default="")
    parser.add_argument("--operator-confirmation", default="")
    parser.add_argument("--output-root", default="hardware_runs/engineering_pilot")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    prepared_dir = (ROOT / args.prepared_dir).resolve()
    freeze_path = (ROOT / args.freeze).resolve()
    site_path = (
        (ROOT / args.site).resolve()
        if args.site
        else ROOT
        / ("configs/hardware_site_mock.yaml" if args.adapter == "mock" else "local_hardware_site.yaml")
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    site = yaml.safe_load(site_path.read_text(encoding="utf-8"))
    assert_no_placeholders(site)
    verify_prepared(prepared_dir)
    verify_freeze(freeze_path)
    expected = f"RUN ENGINEERING PILOT ON {args.mode.upper()}"
    if args.adapter == "ros2":
        if args.arm != config["safety"]["real_adapter_requires_explicit_arm_phrase"]:
            raise RuntimeError("exact frozen arm phrase was not supplied")
        if args.operator_confirmation != expected:
            raise RuntimeError(f"operator confirmation must be exactly: {expected}")
        if not args.preflight:
            raise RuntimeError("a recent passed ROS 2 preflight is required")
        verify_preflight(
            (ROOT / args.preflight).resolve(), config, config_path, site_path
        )

    bundle = engineering_pilot_bundle(args.mode, config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"PILOT_{args.mode.upper()}_{stamp}"
    schedule_row = {
        "run_id": run_id,
        "block_id": f"ENGINEERING_PILOT_{args.mode.upper()}",
        "condition_code": "P",
        "condition": "engineering_pilot",
        "bundle_id": bundle["bundle_id"],
    }
    run_dir = (ROOT / args.output_root / run_id).resolve()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    archive = run_dir / "frozen_inputs"
    archive.mkdir()
    shutil.copy2(config_path, archive / "hardware_study_v1.yaml")
    shutil.copy2(site_path, archive / "hardware_site.yaml")
    shutil.copy2(prepared_dir / "PREPARED_MANIFEST.json", archive / "PREPARED_MANIFEST.json")
    shutil.copy2(freeze_path, archive / "FREEZE.json")
    write_json(archive / "schedule_row.json", schedule_row)
    write_json(archive / "command_bundle.json", bundle)
    os.environ["ROS_DOMAIN_ID"] = str(site.get("ros_domain_id", 0))
    static_hashes = {
        "config_sha256": sha256_file(config_path),
        "site_sha256": sha256_file(site_path),
        "prepared_manifest_sha256": sha256_file(prepared_dir / "PREPARED_MANIFEST.json"),
        "machine_schedule_sha256": sha256_file(prepared_dir / "machine_schedule.json"),
        "bundle_sha256": sha256_file(archive / "command_bundle.json"),
        "freeze_sha256": sha256_file(freeze_path),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    process = None
    log_handle = None
    adapter = None
    bag_metadata = None
    try:
        if args.adapter == "ros2":
            process, log_handle, bag_metadata = start_rosbag(run_dir, site)
        adapter = make_adapter(args.adapter, site)
        manifest = execute_run(
            adapter,
            bundle,
            schedule_row,
            config,
            site,
            run_dir,
            static_hashes,
            rosbag_metadata=bag_metadata,
        )
    finally:
        if adapter is not None:
            adapter.close()
        if process is not None:
            stop_rosbag(process, log_handle)
    validation = validate_and_write(run_dir, config, bundle, schedule_row)
    write_json(
        run_dir / "ENGINEERING_ONLY.json",
        {
            "engineering_only": True,
            "exclude_from_scientific_analysis": True,
            "mode": args.mode,
        },
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "mode": args.mode,
                "completed": manifest["completed"],
                "technical_valid": validation["technical_valid"],
                "archive": str(run_dir),
            },
            indent=2,
        )
    )
    return 0 if validation["technical_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
