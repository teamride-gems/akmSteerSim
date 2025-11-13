# rl/train.py
from pathlib import Path
import yaml, os, time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from envs.f1tenth_sb3_env import F1TenthSACEnv

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    sac_cfg = yaml.safe_load((ROOT / "configs" / "sac.yaml").read_text())

    veh_cfg_path = ROOT / "configs" / "vehicle.yaml"
    if not veh_cfg_path.exists():
        raise FileNotFoundError(f"Missing vehicle config: {veh_cfg_path}")
    veh_cfg = yaml.safe_load(veh_cfg_path.read_text())
    if not veh_cfg:
        raise ValueError(f"Vehicle config at {veh_cfg_path} is empty or invalid YAML.")

    cl_csv = ROOT / "assets" / "track_centerline.csv"
    if not cl_csv.exists():
        raise FileNotFoundError(f"Missing centerline CSV: {cl_csv}")

    env = F1TenthSACEnv(vehicle_cfg=veh_cfg, track_centerline_csv=str(cl_csv))

    model = SAC(
        policy=sac_cfg["policy"],
        env=env,
        learning_rate=sac_cfg["learning_rate"],
        batch_size=sac_cfg["batch_size"],
        tau=sac_cfg["tau"],
        gamma=sac_cfg["gamma"],
        seed=sac_cfg["seed"],
        policy_kwargs=sac_cfg.get("policy_kwargs", None),
        verbose=1,
        tensorboard_log=str(ROOT / "runs"),
        device="auto",
        ent_coef=sac_cfg.get("ent_coef", "auto"),
    )

    save_dir = ROOT / "checkpoints" / time.strftime("%Y%m%d-%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    eval_env = F1TenthSACEnv(vehicle_cfg=veh_cfg, track_centerline_csv=str(cl_csv))
    callbacks = [
        CheckpointCallback(save_freq=sac_cfg["save_every_steps"], save_path=str(save_dir), name_prefix="sac_f1"),
        EvalCallback(eval_env, best_model_save_path=str(save_dir), eval_freq=sac_cfg["eval_interval_steps"], n_eval_episodes=3),
    ]

    model.learn(total_timesteps=sac_cfg["train_steps"], callback=callbacks)
    model.save(str(save_dir / "sac_final"))
