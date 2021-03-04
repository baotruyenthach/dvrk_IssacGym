"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Kuka bin perfromance test
-------------------------------
Test simulation perfromance and stability of the robotic arm dealing with a set of complex objects in a bin.
"""

from __future__ import print_function, division, absolute_import

import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from copy import copy
import matplotlib.pyplot as plt
import cv2
import pptk
import random



# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Kuka Bin Test",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 16, "help": "Number of environments to create"},
        {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
        {"name": "--object_type", "type": int, "default": 0, "help": "Type of bjects to place in the bin: 0 - box, 1 - meat can, 2 - banana, 3 - mug, 4 - brick, 5 - random"}])

num_envs = args.num_envs
num_envs = 1

# configure sim
sim_type = args.physics_engine
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
# sim_params.dt = 1.0 / 1500
# sim_params.substeps = 1
if sim_type is gymapi.SIM_FLEX:
    sim_params.substeps = 4
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 40
    sim_params.flex.relaxation = 0.2
    sim_params.flex.warm_start = 0.1
elif sim_type is gymapi.SIM_PHYSX:
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 25
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.rest_offset = 0.001

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)

# enable Von-Mises stress visualization
sim_params.stress_visualization = True
sim_params.stress_visualization_min = 0.0
sim_params.stress_visualization_max = 1.e+5

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up ground
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load assets
asset_root = "../../assets"

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.1)
#pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

asset_options = gymapi.AssetOptions()
asset_options.armature = 0.001
asset_options.fix_base_link = True
asset_options.thickness = 0.002


asset_root = "../../assets"
# kuka_asset_file = "urdf/kuka_allegro_description/kuka_allegro.urdf"
# kuka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
#kuka_asset_file = "urdf/daVinci_description/robots/psm_from_WPI_test_2.urdf"  # change
kuka_asset_file = "urdf/daVinci_description/robots/new_psm.urdf"
# kuka_asset_file = "urdf/daVinci_description/robots/new_psm_new_gripper.urdf"

asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
if sim_type is gymapi.SIM_FLEX:
    asset_options.max_angular_velocity = 40.

print("Loading asset '%s' from '%s'" % (kuka_asset_file, asset_root))
kuka_asset = gym.load_asset(sim, asset_root, kuka_asset_file, asset_options)

# Create soft icosphere asset
asset_root = "./" # Current directory
soft_asset_file = "deformable_object_grasping/examples/rectangle/test_soft_body.urdf"

soft_pose = gymapi.Transform()
soft_pose.p = gymapi.Vec3(0, 0.395, 0.03)
soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
soft_thickness = 0.05    # important to add some thickness to the soft body to avoid interpenetrations

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.thickness = soft_thickness
asset_options.disable_gravity = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)

asset_soft_body_count = gym.get_asset_soft_body_count(soft_asset)
asset_soft_materials = gym.get_asset_soft_materials(soft_asset)

# Print asset soft material properties
print('Soft Material Properties:')
for i in range(asset_soft_body_count):
    mat = asset_soft_materials[i]
    print(f'(Body {i}) youngs: {mat.youngs} poissons: {mat.poissons} damping: {mat.damping}')



# create box asset
box_size = 0.045
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)
box_pose = gymapi.Transform()

# set up the env grid
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# use position drive for all dofs; override default stiffness and damping values
dof_props = gym.get_asset_dof_properties(kuka_asset)
dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
dof_props["stiffness"][:8].fill(1000.0)
dof_props["damping"][:8].fill(200.0)
# dof_props["stiffness"][2:8].fill(1)
# dof_props["damping"][2:8].fill(20)

# dof_props["stiffness"][8:].fill(20.0)
# dof_props["damping"][8:].fill(4.0)


# get joint limits and ranges for kuka
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']
ranges = upper_limits - lower_limits
mids = 0.5 * (upper_limits + lower_limits)
#num_dofs = len(kuka_dof_props)


# default dof states and position targets
num_dofs = gym.get_asset_dof_count(kuka_asset)
default_dof_pos = np.zeros(num_dofs, dtype=np.float32)
default_dof_pos = lower_limits

default_dof_state = np.zeros(num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# cache some common handles for later use
envs = []
kuka_handles = []




print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))
base_poses = []

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)


    # add kuka
    kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", i, 1)
    



    kuka_handles.append(kuka_handle)

# Camera setup
cam_pos = gymapi.Vec3(0.5, 0.5, 1)
cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)



frame_count = 0
start_time = 1
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    t = gym.get_sim_time(sim)
    if (t <= start_time):
        gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_yaw_joint"), 0)
        gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_pitch_back_joint"), 0)
        gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.2)
        gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_roll_joint"), np.pi/3)
        gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_pitch_joint"), 0)
        gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_yaw_joint"), 0.0)

        # gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_pitch_bottom_joint"), 0.0)
        # gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_pitch_end_joint"), 0.0)
        gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper1_joint"), 1.0)
        gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper2_joint"), 1.0)  
    

    
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)


    frame_count = frame_count + 1



# Output img with window name as 'image' 
# cv2.imshow('image', color_image)  
# cv2.waitKey(0)         
# cv2.destroyAllWindows()  



print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


