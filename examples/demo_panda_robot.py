# import os
# os.environ["MUJOCO_GL"] = "egl"
# # os.environ["MUJOCO_GL"] = "glfw"
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu/nvidia-opengl/:'+'/private/home/basileterv/.mujoco/mujoco210/bin:' + '/usr/lib/nvidia:' + os.environ.get('LD_LIBRARY_PATH', '')
from mujoco_panda import PandaArm
from mujoco_panda.utils.debug_utils import ParallelPythonCmd
import time
import os

from matplotlib import pyplot as plt

"""
Testing PandaArm instance and parallel command utility

# in the parallel debug interface run PandaArm arm commands to control and monitor the robot.
# eg: p.set_neutral_pose(), p.joint_angles(), p.ee_pose(), etc.

"""

def exec_func(cmd):
    if cmd == '':
        return None
    a = eval(cmd)
    print(cmd)
    print(a)
    if a is not None:
        return str(a)

MODEL_PATH = os.environ["MJ_PANDA_PATH"] + "/mujoco_panda/models/"

if __name__ == "__main__":

# -----------Vlad's---------------
    p = PandaArm(
        # model_path=MODEL_PATH + "panda_block_table.xml",
        model_path=MODEL_PATH + "franka_panda_pos.xml",
        render=False,
        compensate_gravity=False,
        smooth_ft_sensor=True,
    )

    image = p._sim.render(500, 500)

    image = image[::-1, :, :]  # flip image vertically

    # save image to file
    plt.imsave("image.png", image)

# --------------------------
    # p = PandaArm.fullRobotWithTorqueActuators(render=True,compensate_gravity=True)
    # p.start_asynchronous_run() # run simulation in separate independent thread

    # _cth = ParallelPythonCmd(exec_func)

    # while True:
    #     p.render()
# ---------------------------
# in the parallel debug interface run PandaArm arm commands to control and monitor the robot.
# eg: p.set_neutral_pose(), p.joint_angles(), p.ee_pose(), etc.
