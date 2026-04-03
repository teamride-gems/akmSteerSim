import json
import os
import time
import numpy as np
import multiprocessing as mp
from typing import List, Dict, Any
from pathlib import Path
import torch
import random
from stable_baselines3 import SAC


from rl.train import make_env_for_track

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def _run_single_episode(ckpt_path: str, vehicle_cfg: Dict[str,Any], track: str, pose: Dict[str,Any], deterministic=True, timeout_s: float = 120.0):
    """
    Run a single episode at the given start pose.
    Returns a dict with metrics and raw trajectory (optional).
    """
    # Use local seeds:
    set_all_seeds(int(pose.get("pose_seed", 0)))
    env = make_env_for_track(vehicle_cfg, track, render_mode=None)
    try:
        # reset with explicit start pose if env supports it via reset(params)
        # Many envs accept an initial_state dict; if not, you may need to set env.world state programmatically.
        reset_kwargs = {"init_x": pose["x"], "init_y": pose["y"], "init_yaw": pose["yaw"]}
        try:
            obs, info = env.reset(**reset_kwargs)
        except TypeError:
            # backwards-compatible fallback: env.reset() then env.set_pose(...) if exists
            obs, info = env.reset()
            if hasattr(env, "set_pose"):
                env.set_pose(pose["x"], pose["y"], pose["yaw"])
        # load policy model
        model = SAC.load(ckpt_path, device="cpu")
        done = False
        t0 = time.time()
        steps = 0
        ep_reward = 0.0
        traj = []
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            steps += 1
            if hasattr(info, "get") and isinstance(info, dict):
                pass
            # collect minimal trajectory if needed
            if steps % 5 == 0:
                try:
                    state = {"x": info.get("x", np.nan), "y": info.get("y", np.nan)}
                except Exception:
                    state = {}
                traj.append(state)
            if terminated or truncated:
                done = True
            # timeout safety
            if (time.time() - t0) > timeout_s:
                break
        end_time = time.time()
        duration = end_time - t0
        metrics = dict(
            success = bool(info.get("success", False)) if isinstance(info, dict) else False,
            ep_reward = float(ep_reward),
            steps = int(steps),
            duration_s = float(duration),
            pose_id = int(pose["pose_id"]),
            pose_seed = int(pose.get("pose_seed", 0)),
            track = track
        )
        # add optional keys from info (collisions, feasibility violations) if present
        if isinstance(info, dict):
            metrics.update({k:info.get(k) for k in ["collisions", "feasibility_violations", "max_lateral_accel"] if k in info})
    finally:
        try:
            env.close()
        except Exception:
            pass
    return metrics

def _worker(args):
    return _run_single_episode(*args)

def run_episodes_parallel(ckpt_path: str, vehicle_cfg: Dict[str,Any], track: str, poses: List[Dict[str,Any]], deterministic: bool, timeout_s: float, workers: int):
    work_items = []
    for pose in poses:
        work_items.append((ckpt_path, vehicle_cfg, track, pose, deterministic, timeout_s))
    pool = mp.Pool(processes=min(workers, len(work_items)))
    results = pool.map(_worker, work_items)
    pool.close()
    pool.join()
    return results

def aggregate_results_list(results: List[Dict[str,Any]]):
    # Build summary metrics for a set of episodes
    N = len(results)
    if N == 0:
        return {}
    success_rate = sum(1 for r in results if r.get("success")) / N
    mean_reward = float(np.mean([r["ep_reward"] for r in results]))
    mean_duration = float(np.mean([r["duration_s"] for r in results]))
    mean_steps = float(np.mean([r["steps"] for r in results]))
    # feasibility violations if present
    fv = [r.get("feasibility_violations", 0) for r in results]
    mean_fv = float(np.mean(fv)) if fv else 0.0
    return {
        "n_episodes": N,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "mean_duration_s": mean_duration,
        "mean_steps": mean_steps,
        "mean_feasibility_violations": mean_fv
    }
