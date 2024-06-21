import os
import time
import threading
import mujoco_py
import quaternion
import numpy as np
from mujoco_panda import PandaArm
from mujoco_panda.utils.tf import quatdiff_in_euler
from mujoco_panda.utils.viewer_utils import render_frame
import matplotlib.pyplot as plt
import pathlib
import imageio
import torch

"""
Simplified demo of task-space control using joint torque actuation.

Robot moves its end-effector 10cm upwards (+ve Z axis) from starting position.
"""

# --------- Modify as required ------------
# Task-space controller parameters
# stiffness gains
P_pos = 1500.
P_ori = 200.
# damping gains
D_pos = 2.*np.sqrt(P_pos)
D_ori = 1.
# ----------------------Video Utils-------------------
def make_video(images, fps, output_path):
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
# --------------------------------------------------------------------


def compute_ts_force(curr_pos, curr_ori, goal_pos, goal_ori, curr_vel, curr_omg):
    delta_pos = (goal_pos - curr_pos).reshape([3, 1])
    delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3, 1])
    # print
    F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
        np.vstack([D_pos*(curr_vel).reshape([3, 1]),
                   D_ori*(curr_omg).reshape([3, 1])])

    error = np.linalg.norm(delta_pos) + np.linalg.norm(delta_ori)

    return F, error


def controller_thread(ctrl_rate):
    global p, target_pos

    threshold = 0.005

    target_pos = curr_ee.copy()
    while run_controller:

        error = 100.
        while error > threshold:
            now_c = time.time()
            curr_pos, curr_ori = p.ee_pose()
            curr_vel, curr_omg = p.ee_velocity()

            target_pos[2] = z_target
            F, error = compute_ts_force(
                curr_pos, curr_ori, target_pos, original_ori, curr_vel, curr_omg)

            if error <= threshold:
                break

            impedance_acc_des = np.dot(p.jacobian().T, F).flatten().tolist()

            p.set_joint_commands(impedance_acc_des, compensate_dynamics=True)

            p.step(render=False)

            elapsed_c = time.time() - now_c
            sleep_time_c = (1./ctrl_rate) - elapsed_c
            if sleep_time_c > 0.0:
                time.sleep(sleep_time_c)

MODEL_PATH = os.environ["MJ_PANDA_PATH"] + "/mujoco_panda/models/"
FILE_NAME = pathlib.Path(__file__).stem
OUTPUT_BASE_PATH = pathlib.Path(__file__).parent.parent.resolve() / "local" / "plots" / FILE_NAME

if __name__ == "__main__":
    # p = PandaArm.withTorqueActuators(render=True, compensate_gravity=True)
    p = PandaArm.fullRobotWithPositionActuators(render=False, compensate_gravity=True)
    # p = PandaArm(
    #     # model_path=MODEL_PATH + "panda_block_table.xml",
    #     model_path=MODEL_PATH + "franka_panda_pos.xml",
    #     render=False,
    #     compensate_gravity=False,
    #     smooth_ft_sensor=True,
    # )
    image = p._sim.render(224, 224)
    image = image[::-1, :, :]  # flip image vertically
    plt.imsave(OUTPUT_BASE_PATH/"init_state.png", image)

    ctrl_rate = 1/p.model.opt.timestep
    render_rate = 100

    p.set_neutral_pose()
    p.step()
    time.sleep(0.01)

    new_pose = p._sim.data.qpos.copy()[:7]

    curr_ee, original_ori = p.ee_pose()

    target_z_traj = np.linspace(curr_ee[2], curr_ee[2]+0.1, 25).tolist()
    z_target = curr_ee[2]

    target_pos = curr_ee
    run_controller = True
    ctrl_thread = threading.Thread(target=controller_thread, args=[ctrl_rate])
    ctrl_thread.start()

    now_r = time.time()
    i = 0
    frames = []
    while i < len(target_z_traj):
        z_target = target_z_traj[i]

        robot_pos, robot_ori = p.ee_pose()
        elapsed_r = time.time() - now_r

        if elapsed_r >= 0.1:
            i += 1
            now_r = time.time()
        
        frame = p._sim.render(224, 224)[::-1, :, :]  # flip image vertically

        frames.append(frame)
        # render_frame(p.viewer, robot_pos, robot_ori)
        # render_frame(p.viewer, target_pos, original_ori, alpha=0.2)
        # p.render()
        # Save video
    output_path = str(OUTPUT_BASE_PATH / "simulation_video")
    make_video(frames, fps=30, output_path=output_path)
    print("Done controlling. Press Ctrl+C to quit.")

    # while True:
    #     robot_pos, robot_ori = p.ee_pose()
    #     render_frame(p.viewer, robot_pos, robot_ori)
    #     render_frame(p.viewer, target_pos, original_ori, alpha=0.2)
    #     p.render()

    run_controller = False
    ctrl_thread.join()
