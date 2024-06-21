import numpy as np
import imageio
import pathlib
from mujoco_panda import PandaArm
import quaternion

def print_attrs_robot(p):
    print(f"{p.actuated_arm_joint_names=}")
    print(f"{p.actuated_gripper_joint_names=}")
    # Print Joint to Actuator Mapping
    print("Joint to Actuator Mapping:")
    for joint_id in p._model.jnt_qposadr:
        joint_name = p._model.joint_id2name(joint_id)
        try:
            actuator_id = p.get_actuator_ids([joint_id])[0]
            actuator_name = p._model.actuator_id2name(actuator_id)
            print(f"Joint '{joint_name}' is actuated by '{actuator_name}'")
        except ValueError as e:
            print(f"Joint '{joint_name}' does not have a valid actuator")
    # Print information about all bodies
    print("\nBody Information:")
    for body_id in range(p._model.nbody):
        body_name = p._model.body_id2name(body_id)
        body_pos, body_quat = p.body_pose(body_name)
        print(f"Body '{body_name}': Position = {body_pos}, Quaternion = {body_quat}")
    # Print information about all sites
    print("\nSite Information:")
    for site_id in range(p._model.nsite):
        site_name = p._model.site_id2name(site_id)
        site_pos, site_quat = p.site_pose(site_name)
        print(f"Site '{site_name}': Position = {site_pos}, Quaternion = {site_quat}")
    # Print information about joint positions, velocities, and accelerations
    print("\nJoint Dynamics:")
    joint_positions = p.joint_positions()
    joint_velocities = p.joint_velocities()
    joint_accelerations = p.joint_accelerations()
    print(f"Joint Positions: {joint_positions}")
    print(f"Joint Velocities: {joint_velocities}")
    print(f"Joint Accelerations: {joint_accelerations}")
    # Print information about contact points
    print("\nContact Information:")
    contacts = p.get_contact_info()
    for contact in contacts:
        print(contact)
    # Print end-effector position and orientation
    ee_pos, ee_quat = p.ee_pose()
    ee_euler = quaternion.as_euler_angles(quaternion.from_float_array(ee_quat))  # Convert quaternion to Euler angles
    print("\nEnd-Effector Information:")
    print(f"Position: {ee_pos}")
    print(f"Quaternion: {ee_quat}")
    print(f"Euler Angles (roll, pitch, yaw): {ee_euler}")

# Function to save videos
def save_video(frames, fps, output_path, format='mp4'):
    output_directory = pathlib.Path(output_path).parent
    output_directory.mkdir(parents=True, exist_ok=True)
    if format == 'mp4':
        writer = imageio.get_writer(f"{output_path}.mp4", fps=fps, codec='libx264')
    else:
        writer = imageio.get_writer(f"{output_path}.gif", fps=fps, format='GIF')
    
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"{format.upper()} video saved to {output_path}.{format}")

if __name__=="main":
    # Define the number of steps and the constant action
    num_steps = 500
    constant_action_joints = np.array([0.04, 0, -0.08, 0, 0, 0, 0.04]) # 7-dim
    # positive will open gripper
    constant_action_gripper = np.array([0.08, 0.0]) # 2-dim

    # Initialize the PandaArm
    p = PandaArm.fullRobotWithPositionActuators(compensate_gravity=True, render=False)
    print_attrs_robot(p)

    # Main simulation loop
    frames = []
    for _ in range(num_steps):
        # Set joint commands and compensate for dynamics
        # p.set_joint_commands(constant_action_joints, compensate_dynamics=True)
        p.set_actuator_ctrl(constant_action_joints, actuator_ids=[0, 1, 2, 3, 4, 5, 6])
        p.set_actuator_ctrl(constant_action_gripper, actuator_ids=[7, 8])
        p.step(render=False)  # Step the simulation
        frame = p._sim.render(224, 224)[::-1, :, :]
        frames.append(frame)

    # Define output path
    FILE_NAME = pathlib.Path(__file__).stem
    OUTPUT_BASE_PATH = pathlib.Path(__file__).parent.parent.resolve() / "local" / "plots" / FILE_NAME
    output_base_path = OUTPUT_BASE_PATH / "panda_sim_vid_set_actuator_ctrl"

    # Save as MP4 and GIF
    save_video(frames, fps=30, output_path=str(output_base_path), format='mp4')
    save_video(frames, fps=30, output_path=str(output_base_path), format='gif')

    print("Simulation and video saving completed.")