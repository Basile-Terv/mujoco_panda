import os
import time
import mujoco_py
import quaternion
import numpy as np
from mujoco_panda import PandaArm
from mujoco_panda.utils.tf import quatdiff_in_euler
import imageio
import pathlib
import torch
from evals.simu_env_planning.envs.mujoco_panda_wrap import PandaArmEnv
from evals.simu_env_planning.envs.mujoco_panda.examples.constant_actions_video import print_attrs_robot

# Task-space controller parameters
P_pos = 1500.  # Position stiffness gain
P_ori = 200.   # Orientation stiffness gain
D_pos = 2. * np.sqrt(P_pos)  # Position damping gain
D_ori = 1.     # Orientation damping gain

# length of traj to travel
z_range = 0.3
nb_steps_per_unit_z = 250

# ----------------------Video Utils---------------------------
def make_video(images, fps, output_path):
    output_directory = pathlib.Path(output_path).parent
    output_directory.mkdir(parents=True, exist_ok=True)
    make_video_mp4(images, fps, output_path + ".mp4")
    make_video_gif(images, fps, output_path + ".gif")


def make_video_mp4(images, fps, output_path):
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")

    for img in images:
        img = img[-1].numpy() if isinstance(img, torch.Tensor) else img
        # img = img.transpose(1, 2, 0)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        writer.append_data(img)

    writer.close()
    print(f"Video saved to {output_path}")


def make_video_gif(images, fps, output_path):
    writer = imageio.get_writer(output_path, fps=fps, format="GIF")

    for img in images:
        img = img[-1].numpy() if isinstance(img, torch.Tensor) else img
        # img = img.transpose(1, 2, 0)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        writer.append_data(img)

    writer.close()
    print(f"GIF saved to {output_path}")

MODEL_PATH = os.environ["MJ_PANDA_PATH"] + "/mujoco_panda/models/"
FILE_NAME = pathlib.Path(__file__).stem
OUTPUT_BASE_PATH = pathlib.Path(__file__).parent.parent.resolve() / "local" / "plots" / FILE_NAME

# ----------------------------------------------------------------------

def compute_ts_force(curr_pos, curr_ori, goal_pos, goal_ori, curr_vel, curr_omg):
    delta_pos = (goal_pos - curr_pos).reshape([3, 1])
    delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3, 1])
    F = np.vstack([P_pos * delta_pos, P_ori * delta_ori]) - \
        np.vstack([D_pos * curr_vel.reshape([3, 1]), D_ori * curr_omg.reshape([3, 1])])
    error = np.linalg.norm(delta_pos) + np.linalg.norm(delta_ori)
    return F, error

if __name__ == "__main__":
    p = PandaArm.fullRobotWithTorqueActuators(render=False, compensate_gravity=True)
    print_attrs_robot(p)
    env = PandaArmEnv(robot = p)
    ctrl_rate = 1 / p.model.opt.timestep
    render_rate = 100

    p.set_neutral_pose()
    p.step()
    time.sleep(0.01)

    curr_ee, original_ori = p.ee_pose()
    
    target_z_traj = np.linspace(curr_ee[2], curr_ee[2] + z_range, int(z_range * nb_steps_per_unit_z)).tolist()
    z_target = curr_ee[2]
    target_pos = curr_ee.copy()

    frames = []

    i = 0
    compute_ts_force_counter = 0
    while i < len(target_z_traj):
        z_target = target_z_traj[i]
        curr_pos, curr_ori = p.ee_pose()
        curr_vel, curr_omg = p.ee_velocity()
        target_pos[2] = z_target
        F, error = compute_ts_force(curr_pos, curr_ori, target_pos, original_ori, curr_vel, curr_omg)
        compute_ts_force_counter +=1
        # print(f"{error=}")
        if error <= 0.005:
            print("\n error <= 0.005, Next target_z_traj[i] to attain")
            i += 1

        impedance_acc_des = np.dot(p.jacobian().T, F).flatten().tolist()
        p.set_joint_commands(impedance_acc_des, compensate_dynamics=True)
        p.step(render=False)

        frame = p._sim.render(224, 224)[::-1, :, :]  # flip image vertically
        frames.append(frame)
    print(f"Finished at {i=}")
    print(f"{compute_ts_force_counter=}")
    output_path = str(OUTPUT_BASE_PATH / "simulation_video")
    make_video(frames, fps=30, output_path=output_path)