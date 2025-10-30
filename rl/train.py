import os, yaml, time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from envs.f1tenth_sb3_env import F1TenthSACEnv


ROOT = os.path.dirname(os.path.dirname(__file__))


if __name__ == "__main__":
sac_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "sac.yaml")).read())
veh_cfg = os.path.join(ROOT, "configs", "vehicle.yaml")
cl_csv = os.path.join(ROOT, "assets", "track_centerline.csv")


env = F1TenthSACEnv(vehicle_cfg=veh_cfg, track_centerline_csv=cl_csv)


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
tensorboard_log=os.path.join(ROOT, "runs"),
device="auto",
ent_coef=sac_cfg.get("ent_coef", "auto"),
)


save_dir = os.path.join(ROOT, "checkpoints", time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(save_dir, exist_ok=True)


eval_env = F1TenthSACEnv(vehicle_cfg=veh_cfg, track_centerline_csv=cl_csv)
callbacks = [
CheckpointCallback(save_freq=sac_cfg["save_every_steps"], save_path=save_dir, name_prefix="sac_f1"),
EvalCallback(eval_env, best_model_save_path=save_dir, eval_freq=sac_cfg["eval_interval_steps"], n_eval_episodes=3)
]


model.learn(total_timesteps=sac_cfg["train_steps"], callback=callbacks)
model.save(os.path.join(save_dir, "sac_final"))