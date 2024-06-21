import os
import time
import numpy as np
from mujoco_panda import PandaArm
from mujoco_panda.utils.tf import quatdiff_in_euler
import pathlib
import imageio

# Task-space controller parameters
P_pos = 1500.
P_ori = 200.
D_pos = 2.*np.sqrt(P_pos)
D_ori = 1.

def compute_ts_force(curr_pos, curr_ori, goal_pos, goal_ori, curr_vel, curr_omg):
    delta_pos = (goal_pos - curr_pos).reshape([3, 1])
    delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3, 1])
    F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
        np.vstack([D_pos*(curr_vel).reshape([3, 1]), D_ori*(curr_omg).reshape([3, 1])])
    error = np.linalg.norm(delta_pos) + np.linalg.norm(delta_ori)
    return F, error

def make_video(images, fps, output_path):
    output_directory = pathlib.Path(output_path).parent
    output_directory.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    writer = imageio.get_writer(output_path + ".mp4", fps=fps, codec="libx264")
    for img in images:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        writer.append_data(img)
    writer.close()
    print(f"Video saved to {output_path}.mp4")

if __name__ == "__main__":
    # p = PandaArm.fullRobotWithPositionActuators(render=False, compensate_gravity=True)
    p = PandaArm.withTorqueActuators(render=False, compensate_gravity=True)
    ctrl_rate = 1/p.model.opt.timestep
    p.set_neutral_pose()
    p.step()
    time.sleep(0.01)

    curr_ee, original_ori = p.ee_pose()
    target_z_traj = np.linspace(curr_ee[2], curr_ee[2]+0.1, 25).tolist()
    z_target = curr_ee[2]
    target_pos = curr_ee.copy()
    frames = []

    for z_target in target_z_traj:
        target_pos[2] = z_target
        def controller_thread(ctrl_rate):
            threshold = 0.005
            error = 100.
            while error > threshold:
                now_c = time.time()
                curr_pos, curr_ori = p.ee_pose()
                curr_vel, curr_omg = p.ee_velocity()
                F, error = compute_ts_force(curr_pos, curr_ori, target_pos, original_ori, curr_vel, curr_omg)
                impedance_acc_des = np.dot(p.jacobian().T, F).flatten().tolist()

                p.set_joint_commands(impedance_acc_des, compensate_dynamics=True)
                
                breakpoint()
                p.step(render=False)
                frame = p._sim.render(224, 224)
                frames.append(frame)

                breakpoint()
                elapsed_c = time.time() - now_c
                sleep_time_c = (1./ctrl_rate) - elapsed_c
                if sleep_time_c > 0.0:
                    time.sleep(sleep_time_c)
                breakpoint()

        controller_thread(ctrl_rate)

    output_path = str(pathlib.Path(__file__).parent.parent.resolve() / "local" / "plots" / pathlib.Path(__file__).stem / "simulation_video")
    breakpoint()
    make_video(frames, fps=30, output_path=output_path)
    print("Done controlling. Press Ctrl+C to quit.")