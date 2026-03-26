#!/usr/bin/env python3
"""
Run a trained SB3 SAC model in your F1TenthSACEnv.

Supports:
- render (if your WSL2 pyglet window works)
- headless recording to .npz (then replay with replay_rollout_html.py)

Example:
  python scripts/run_trained_policy.py \
    --model checkpoints/20260120-233406/sac_final \
    --vehicle_cfg configs/vehicle.yaml \
    --track Sakhir \
    --steps 5000 \
    --record --record_path rollouts/sac_sakhir.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from metrics_logger import LapMetricsLogger

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict) or not d:
        raise ValueError(f"Invalid/empty YAML: {path}")
    return d


def resolve_track_assets(track: str, veh_cfg: dict) -> tuple[str, Path]:
    """
    Returns (map_name, centerline_csv_path)

    We prefer:
      assets/f1tenth_racetracks/<Track>/<Track>_centerline.csv
    but fall back to any '*centerline*.csv' in that directory.
    """
    track = track.replace("_map", "").strip()

    # base dir for track
    base = ROOT / "assets" / "f1tenth_racetracks" / track
    if not base.exists():
        raise FileNotFoundError(f"Track directory not found: {base}")

    # centerline file
    preferred = base / f"{track}_centerline.csv"
    if preferred.exists():
        cl = preferred
    else:
        # fallback: any csv with "centerline" in name
        candidates = sorted(base.glob("*centerline*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No centerline CSV found in {base}. Expected {preferred.name} or *centerline*.csv"
            )
        cl = candidates[0]

    # map_name format expected by f110_gym often "<Track>_map"
    map_name = f"{track}_map"

    # Also update vehicle cfg sim entries so env sees consistent info
    veh_cfg.setdefault("sim", {})
    veh_cfg["sim"]["map_dir"] = str(base)
    veh_cfg["sim"]["map_name"] = map_name

    return map_name, cl


def try_get_pose(env) -> tuple[float, float, float]:
    """
    Best-effort pose getter.
    Tries:
      - env._last_sim_obs dict with poses_x/y/theta
      - env.sim.* common fields
    Returns (x,y,yaw) or (nan,nan,nan) if not available.
    """
    # 1) if your wrapper caches last sim obs
    sim_obs = getattr(env, "_last_sim_obs", None)
    if isinstance(sim_obs, dict):
        try:
            px = np.asarray(sim_obs.get("poses_x"))[0]
            py = np.asarray(sim_obs.get("poses_y"))[0]
            pt = np.asarray(sim_obs.get("poses_theta"))[0]
            return float(px), float(py), float(pt)
        except Exception:
            pass

    # 2) query underlying sim object if present
    sim = getattr(env, "sim", None)
    if sim is not None:
        # common in f110_gym
        for xk, yk, tk in [
            ("poses_x", "poses_y", "poses_theta"),
            ("pose_x", "pose_y", "pose_theta"),
        ]:
            if hasattr(sim, xk) and hasattr(sim, yk) and hasattr(sim, tk):
                try:
                    px = getattr(sim, xk)
                    py = getattr(sim, yk)
                    pt = getattr(sim, tk)
                    px = np.asarray(px).reshape(-1)[0]
                    py = np.asarray(py).reshape(-1)[0]
                    pt = np.asarray(pt).reshape(-1)[0]
                    return float(px), float(py), float(pt)
                except Exception:
                    pass

    return float("nan"), float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to SB3 SAC model (sac_final or sac_final.zip)")
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml", help="Vehicle/env yaml")
    ap.add_argument("--track", default=None, help="Track folder name under assets/f1tenth_racetracks (e.g., Sakhir)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--sleep", type=float, default=0.0, help="Slow down render (sec/step)")
    ap.add_argument("--render", action="store_true", help="Try to render a window")
    ap.add_argument("--deterministic", action="store_true", help="Deterministic actions (recommended for eval)")
    ap.add_argument("--record", action="store_true", help="Record rollout to .npz")
    ap.add_argument("--record_path", default="rollouts/policy_run.npz", help="Where to save .npz if --record")
    ap.add_argument("--record_lidar", action="store_true", help="Also store lidar sectors from obs[7:]")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists() and not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Model not found: {model_path} (or {model_path}.zip)")

    veh_cfg_path = ROOT / args.vehicle_cfg
    if not veh_cfg_path.exists():
        raise FileNotFoundError(f"Vehicle cfg not found: {veh_cfg_path}")
    veh_cfg = load_yaml(veh_cfg_path)

    # Track override (optional)
    if args.track is None:
        # use whatever is in vehicle.yaml
        sim_cfg = veh_cfg.get("sim", {})
        raw_map = sim_cfg.get("map_name", "Sakhir_map")
        track = str(raw_map).replace("_map", "").strip()
    else:
        track = args.track.strip()

    map_name, centerline_csv = resolve_track_assets(track, veh_cfg)

    print("=== Run setup ===")
    print("model:", str(model_path))
    print("track:", track)
    print("map_name:", map_name)
    print("map_dir:", veh_cfg["sim"]["map_dir"])
    print("centerline:", str(centerline_csv))
    print("render:", args.render)
    print("record:", args.record, "->", args.record_path if args.record else "")

    # Import after config is ready
    from stable_baselines3 import SAC
    from envs.f1tenth_sb3_env import F1TenthSACEnv

    env = F1TenthSACEnv(
        vehicle_cfg=veh_cfg,
        track_centerline_csv=str(centerline_csv),
        render_mode="human" if args.render else None,
    )

    # Load model
    # NOTE: SB3 expects env/action_space compatibility; this is your same env used in training.
    model = SAC.load(str(model_path), device="auto")

    obs, info = env.reset()
    ep = 0
    
    lap_id = 0
    lap_start_time = time.time()
    logger = LapMetricsLogger("metrics/lap_metrics.csv")

    # recording buffers
    poses = []
    actions = []
    e_heads = []
    e_lats = []
    lidar_sectors = []

    for t in range(args.steps):
        action, _ = model.predict(obs, deterministic=args.deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        if args.render:
            try:
                env.render()
            except Exception as e:
                print("RENDER ERROR:", repr(e))
                print("Tip: rerun without --render and use --record + HTML replay.")
                args.render = False

        if args.sleep > 0:
            time.sleep(args.sleep)

        if args.record:
            # pose
            x, y, yaw = try_get_pose(env)
            poses.append([x, y, yaw])

            # action
            actions.append([float(action[0]), float(action[1])])

            # errors (your obs layout used earlier: obs[4]=e_head, obs[5]=e_lat)
            eh = float(obs[4]) if np.asarray(obs).size > 5 else float("nan")
            el = float(obs[5]) if np.asarray(obs).size > 5 else float("nan")
            e_heads.append(eh)
            e_lats.append(el)

            # lidar sectors from obs[7:] if you want
            if args.record_lidar:
                o = np.asarray(obs, dtype=float)
                lidar_sectors.append(o[7:].tolist())

        if terminated or truncated:
            ep += 1

            lap_time_sec = time.time() - lap_start_time
            logger.log_lap(lap_id, lap_time_sec)

            # Reset lap
            lap_id += 1
            lap_start_time = time.time()
            crashed = False
            sim_done = False
            if isinstance(info, dict):
                crashed = bool(info.get("crash", False) or info.get("collision", False) or info.get("is_crash", False))
                sim_done = bool(info.get("sim_done", False))
            print(f"[episode {ep}] t={t} crash={crashed} sim_done={sim_done}")
            obs, info = env.reset()

    try:
        logger.close()
        env.close()
    except Exception:
        pass

    if args.record:
        out = Path(args.record_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        poses = np.asarray(poses, dtype=float)
        actions = np.asarray(actions, dtype=float)
        e_heads = np.asarray(e_heads, dtype=float)
        e_lats = np.asarray(e_lats, dtype=float)

        save_kwargs = dict(
            pose=poses,
            action=actions,
            e_head=e_heads,
            e_lat=e_lats,
            track_centerline_csv=str(centerline_csv),
        )

        if args.record_lidar and len(lidar_sectors) > 0:
            save_kwargs["lidar_sectors"] = np.asarray(lidar_sectors, dtype=float)

        np.savez_compressed(out, **save_kwargs)
        print(f"Saved rollout: {out}")

        print("\nReplay it with your HTML tool, e.g.:")
        print(f"  python scripts/replay_rollout_html.py --rollout {out} --out {out.with_suffix('.html')} --stride 1 --lidar")

    print("Done.")


if __name__ == "__main__":
    main()
