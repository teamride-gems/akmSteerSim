#!/usr/bin/env python3
"""
Run a trained SB3 SAC model interactively or headlessly.

Captures per-step trajectory data for paper figures (steering profiles,
tracking error timeseries, trajectory overlays).

Example:
  python scripts/run_trained_policy.py \
    --model checkpoints/steer_speed_full_s0/sac_final.zip \
    --track Sakhir --steps 5000 --record

  # Then replay:
  python scripts/replay_rollout_html.py --rollout rollouts/policy_run.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC

from rl.common import normalize_track_name, make_env_for_track
from scripts.metrics_logger import EpisodeLogger, StepLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to SB3 SAC model (.zip)")
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    ap.add_argument("--track", default=None, help="Track name (default: from vehicle config)")
    ap.add_argument("--steps", type=int, default=5000, help="Max total timesteps to run")
    ap.add_argument("--sleep", type=float, default=0.0, help="Slow down render (sec/step)")
    ap.add_argument("--render", action="store_true", help="Try to render a window")
    ap.add_argument("--deterministic", action="store_true", help="Deterministic actions")
    ap.add_argument("--record", action="store_true", help="Record rollout to .npz")
    ap.add_argument("--record_path", default="rollouts/policy_run.npz")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--from_meta", action="store_true",
                     help="Load vehicle config from run_meta.json near the model")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        if model_path.with_suffix(".zip").exists():
            model_path = model_path.with_suffix(".zip")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    # --- load config ---
    if args.from_meta:
        import json
        meta_path = model_path.parent / "run_meta.json"
        if not meta_path.exists():
            meta_path = model_path.parent.parent / "run_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"run_meta.json not found near {model_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        veh_cfg = meta["vehicle_cfg"]
    else:
        veh_cfg_path = ROOT / args.vehicle_cfg
        if not veh_cfg_path.exists():
            raise FileNotFoundError(f"Vehicle config not found: {veh_cfg_path}")
        veh_cfg = yaml.safe_load(veh_cfg_path.read_text())

    # --- track ---
    if args.track:
        track = normalize_track_name(args.track)
    else:
        raw_map = veh_cfg.get("sim", {}).get("map_name", "Sakhir_map")
        track = normalize_track_name(raw_map)

    action_space_name = veh_cfg.get("action_space", "steer_speed")

    print("=== Run setup ===")
    print(f"  model:        {model_path}")
    print(f"  action_space: {action_space_name}")
    print(f"  track:        {track}")
    print(f"  render:       {args.render}")
    print(f"  deterministic:{args.deterministic}")
    print(f"  record:       {args.record}")

    # --- env + model ---
    render_mode = "human" if args.render else None
    env = make_env_for_track(veh_cfg, track, render_mode=render_mode)
    model = SAC.load(str(model_path), device=args.device)

    # --- loggers ---
    ep_logger = EpisodeLogger("metrics/episodes.csv")
    step_logger = StepLogger("metrics/trajectory.csv")

    # --- recording buffers ---
    rec_poses = []
    rec_actions = []
    rec_steer_cmds = []
    rec_speed_cmds = []
    rec_steer_rates = []
    rec_e_lats = []
    rec_e_heads = []
    rec_speeds = []
    rec_rewards = []
    rec_ep_ids = []

    obs, info = env.reset()
    episode_id = 0
    ep_reward = 0.0
    ep_len = 0

    for t in range(args.steps):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        ep_reward += float(reward)
        ep_len += 1

        # --- per-step logging ---
        step_logger.log(episode_id=episode_id, info=info, reward=float(reward))

        # --- recording ---
        if args.record:
            pose = info.get("pose", [0.0, 0.0, 0.0])
            rec_poses.append([float(pose[0]), float(pose[1]), float(pose[2])])
            rec_actions.append([float(a) for a in np.asarray(action).ravel()])
            rec_steer_cmds.append(float(info.get("steer_cmd", 0.0)))
            rec_speed_cmds.append(float(info.get("speed_cmd", 0.0)))
            rec_steer_rates.append(float(info.get("steer_rate", 0.0)))
            rec_e_lats.append(float(info.get("lateral_error", 0.0)))
            rec_e_heads.append(float(info.get("heading_error", 0.0)))
            rec_speeds.append(float(info.get("speed", 0.0)))
            rec_rewards.append(float(reward))
            rec_ep_ids.append(episode_id)

        # --- render ---
        if args.render:
            try:
                env.render()
            except Exception as e:
                print(f"RENDER ERROR: {repr(e)}")
                print("Rerun without --render and use --record + HTML replay.")
                args.render = False

        if args.sleep > 0:
            time.sleep(args.sleep)

        # --- episode boundary ---
        if terminated or truncated:
            term_reason = info.get("term_reason", "unknown")
            progress = info.get("normalized_progress", 0.0)

            ep_logger.log(
                episode_id=episode_id,
                action_space=action_space_name,
                track=track,
                reward=ep_reward,
                length=ep_len,
                term_reason=term_reason,
                normalized_progress=progress,
            )

            print(
                f"[episode {episode_id}] steps={ep_len} "
                f"reward={ep_reward:.1f} progress={progress:.3f} "
                f"term={term_reason}"
            )

            episode_id += 1
            ep_reward = 0.0
            ep_len = 0
            obs, info = env.reset()

    # --- cleanup ---
    ep_logger.close()
    step_logger.close()
    try:
        env.close()
    except Exception:
        pass

    # --- save recording ---
    if args.record:
        out = Path(args.record_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out,
            pose=np.array(rec_poses, dtype=float),
            action=np.array(rec_actions, dtype=float),
            steer_cmd=np.array(rec_steer_cmds, dtype=float),
            speed_cmd=np.array(rec_speed_cmds, dtype=float),
            steer_rate=np.array(rec_steer_rates, dtype=float),
            lateral_error=np.array(rec_e_lats, dtype=float),
            heading_error=np.array(rec_e_heads, dtype=float),
            speed=np.array(rec_speeds, dtype=float),
            reward=np.array(rec_rewards, dtype=float),
            episode_id=np.array(rec_ep_ids, dtype=int),
            track=track,
            action_space=action_space_name,
        )
        print(f"\nSaved rollout: {out}")
        print(f"  {len(rec_poses)} steps across {episode_id + 1} episodes")
        print(f"Replay: python scripts/replay_rollout_html.py --rollout {out}")

    print("Done.")


if __name__ == "__main__":
    main()