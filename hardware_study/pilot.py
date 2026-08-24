"""Synthetic command bundle used only to qualify the physical command path."""

from __future__ import annotations

import math

import numpy as np


def engineering_pilot_bundle(mode: str, config: dict) -> dict:
    if mode not in {"stands", "ground"}:
        raise ValueError("pilot mode must be 'stands' or 'ground'")
    command = config["command_bundle"]
    preamble_count = int(command["preamble_packets"])
    main_count = int(command["main_packets"])
    postamble_count = int(command["postamble_packets"])
    speed_cap = 0.20 if mode == "stands" else 0.50
    preamble_speed = np.linspace(0.0, speed_cap, preamble_count)
    main_speed = np.full(main_count, speed_cap)
    postamble_speed = np.linspace(speed_cap, 0.0, postamble_count)
    if mode == "stands":
        main_steering = np.zeros(main_count)
    else:
        # Smooth, low-amplitude, zero-mean S-turn unrelated to any study source.
        main_steering = np.asarray(
            [0.05 * math.sin(4.0 * math.pi * index / (main_count - 1)) for index in range(main_count)]
        )
    steering = np.concatenate(
        [np.zeros(preamble_count), main_steering, np.zeros(postamble_count)]
    )
    speed = np.concatenate([preamble_speed, main_speed, postamble_speed])
    phases = (
        ["preamble"] * preamble_count
        + ["main"] * main_count
        + ["postamble"] * postamble_count
    )
    phase_counts = {"preamble": 0, "main": 0, "postamble": 0}
    packets = []
    for index, phase in enumerate(phases):
        packets.append(
            {
                "packet_index": index,
                "phase": phase,
                "phase_packet_index": phase_counts[phase],
                "target_steering_rad": float(steering[index]),
                "target_speed_mps": float(speed[index]),
            }
        )
        phase_counts[phase] += 1
    return {
        "schema_version": 1,
        "bundle_id": f"synthetic_engineering_pilot_{mode}_v1",
        "engineering_only": True,
        "warning": "Never include this run in the frozen scientific analysis.",
        "speed_cap_mps": speed_cap,
        "packet_dt_seconds": float(command["packet_dt_seconds"]),
        "conditions": {"engineering_pilot": packets},
    }
