import os
import sys
import numpy as np

# --- add project root to sys.path ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../akmSteerSim/rl
PROJECT_ROOT = os.path.dirname(THIS_DIR)                   # .../akmSteerSim
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.f1tenth_sb3_env import F1TenthSACEnv


# def handcrafted_policy(obs):
#     """
#     obs: normalized state vector [v, a_long, delta, yaw_rate, e_head, e_lat, a_lat, lidar...]

#     - obs[4] ≈ e_head / pi    (heading error, sign just debugged in step 1)
#     - obs[5] ≈ e_lat / 2.0    (lateral error: left/right of centerline)
#     """
#     e_head_n = float(obs[4])
#     e_lat_n  = float(obs[5])

#     # --- Gains: keep them small ---
#     k_head = 0.5   # whatever sign you confirmed in step 1
#     k_lat  = 0.2   # much smaller than k_head

#     # Use the SAME sign for the heading part that worked in step 1:
#     steer_cmd = -k_head * e_head_n   # or +k_head * e_head_n if that was the good one

#     # Lateral term usually wants the same sign logic:
#     # if e_lat_n > 0 means "I'm to one side", the minus sign tends to pull you back.
#     steer_cmd += -k_lat * e_lat_n

#     # Clip steering to be gentle
#     steer_cmd = float(np.clip(steer_cmd, -0.3, 0.3))

#     # Slow speed; optional slight speed boost if errors small
#     err_mag = abs(e_head_n) + 0.5 * abs(e_lat_n)
#     base_speed_level = -0.4
#     speed_boost = 0.1 * np.exp(-3.0 * err_mag)
#     speed_cmd = base_speed_level + speed_boost
#     speed_cmd = float(np.clip(speed_cmd, -0.6, -0.1))

#     return np.array([speed_cmd, steer_cmd], dtype=float)




def handcrafted_policy(obs):
    """
    obs: normalized state vector
         [v, a_long, delta, yaw_rate, e_head, e_lat, a_lat, lidar_0, ..., lidar_{N-1}]

    We'll use ONLY the LiDAR sectors (obs[7:]) to try to stay centered in the "corridor":
    - If there's more free space on the left, steer left a bit.
    - If there's more free space on the right, steer right a bit.
    """

    # ---------------- LiDAR extraction ----------------
    lidar = np.asarray(obs[7:], dtype=float)
    n = lidar.shape[0]

    if n < 5:
        # Fallback: if something is weird, just go straight & slow
        return np.array([-0.4, 0.0], dtype=float)

    # Split LiDAR into rough regions: [right | center | left]
    third = n // 3
    right_slice = lidar[:third]
    center_slice = lidar[third : 2 * third]
    left_slice  = lidar[2 * third :]

    right_mean = float(np.mean(right_slice))
    left_mean  = float(np.mean(left_slice))
    center_mean = float(np.mean(center_slice))

    # Note: LiDAR values are normalized distances (≈0 = near, ≈1 = far).
    # If left_mean > right_mean → more space on the left → steer left (positive).
    # If right_mean > left_mean → more space on the right → steer right (negative).
    side_imbalance = left_mean - right_mean  # >0 → steer left, <0 → steer right

    # ---------------- Steering command ----------------
    k_side = 0.8   # how strongly we react to left/right imbalance
    k_center = 0.4 # small bias to keep moving toward open center corridor

    # Encourage steering toward the more open side, but also away from a very close center obstacle
    steer_cmd = k_side * side_imbalance

    # If center is very "closed", gently steer toward whichever side has more space
    # (this is already encoded in side_imbalance; this term just amplifies)
    congestion = max(0.0, 0.6 - center_mean)   # bigger if center distances are small
    steer_cmd += k_center * congestion * np.sign(side_imbalance + 1e-6)

    # Keep steering gentle
    steer_cmd = float(np.clip(steer_cmd, -0.4, 0.4))

    # ---------------- Speed command ----------------
    # Go slow, speed up only if things are open all around.
    openness = (left_mean + right_mean + center_mean) / 3.0
    # openness ~ 1.0 in big open area, smaller near walls
    base_speed_level = -0.5                   # near v_min (slow)
    speed_boost = 0.3 * np.clip(openness - 0.5, 0.0, 0.5)  # small boost if environment is open

    speed_cmd = base_speed_level + speed_boost
    speed_cmd = float(np.clip(speed_cmd, -0.7, -0.2))

    return np.array([speed_cmd, steer_cmd], dtype=float)


def main():
    env = F1TenthSACEnv(render_mode="human")

    obs, info = env.reset()
    done = False

    for t in range(500):  # ~500 steps
        action = handcrafted_policy(obs)
        obs, r, terminated, truncated, info = env.step(action)

        # This calls down into f110-gym's pyglet renderer
        env.render()

        if terminated or truncated:
            print(f"Episode ended at step {t}, reward={r:.3f}")
            break

    env.close()


if __name__ == "__main__":
    main()
