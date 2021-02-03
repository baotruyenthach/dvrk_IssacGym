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


# configure sim
sim_type = args.physics_engine
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
if sim_type is gymapi.SIM_FLEX:
    sim_params.substeps = 4
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 20
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
elif sim_type is gymapi.SIM_PHYSX:
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 25
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.rest_offset = 0.001

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)



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
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
#pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

asset_options = gymapi.AssetOptions()
asset_options.armature = 0.001
asset_options.fix_base_link = True
asset_options.thickness = 0.002


asset_root = "../../assets"
#kuka_asset_file = "urdf/kuka_allegro_description/kuka_allegro.urdf"
#kuka_asset_file = "urdf/daVinci_description/robots/psm_from_WPI_test_2.urdf"  # change
kuka_asset_file = "urdf/daVinci_description/robots/new_psm.urdf"

asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = True

if sim_type is gymapi.SIM_FLEX:
    asset_options.max_angular_velocity = 40.

print("Loading asset '%s' from '%s'" % (kuka_asset_file, asset_root))
kuka_asset = gym.load_asset(sim, asset_root, kuka_asset_file, asset_options)

# set up the env grid
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# use position drive for all dofs; override default stiffness and damping values
dof_props = gym.get_asset_dof_properties(kuka_asset)
dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
dof_props["stiffness"].fill(1000.0)
dof_props["damping"].fill(200.0)


# get joint limits and ranges for kuka
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']
ranges = upper_limits - lower_limits
mids = 0.5 * (upper_limits + lower_limits)
#num_dofs = len(kuka_dof_props)


# default dof states and position targets
num_dofs = gym.get_asset_dof_count(kuka_asset)
default_dof_pos = np.zeros(num_dofs, dtype=np.float32)
default_dof_pos = upper_limits

default_dof_state = np.zeros(num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# cache some common handles for later use
envs = []
kuka_handles = []
attractor_handles = {}

# Attractors setup
kuka_attractors = ["psm_tool_yaw_link"]  # , "thumb_link_3", "index_link_3", "middle_link_3", "ring_link_3"]
attractors_offsets = [gymapi.Transform(), gymapi.Transform(), gymapi.Transform(), gymapi.Transform(), gymapi.Transform()]



print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))
base_poses = []

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)


    # add kuka
    kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", i, 1)
    
    
    # Initialize attractors:
    attractor_handles[i] = []
    body_dict = gym.get_actor_rigid_body_dict(env, kuka_handle)
    props = gym.get_actor_rigid_body_states(env, kuka_handle, gymapi.STATE_POS)
    for j, body in enumerate(kuka_attractors):
        attractor_properties = gymapi.AttractorProperties()
        attractor_properties.stiffness = 1e6
        attractor_properties.damping = 5e2
        body_handle = gym.find_actor_rigid_body_handle(env, kuka_handle, body)
        attractor_properties.target = props['pose'][:][body_dict[body]]
        attractor_properties.target.p.y -= 0.15


        # By Default, offset pose is set to origin, so no need to set it
        if j > 0:
            attractor_properties.offset = attractors_offsets[j]
        base_poses.append(attractor_properties.target)
        if j == 0:
            # make attractor in all axes
            attractor_properties.axes = gymapi.AXIS_ALL
        else:
            # make attractor in Translation only
            attractor_properties.axes = gymapi.AXIS_TRANSLATION

        # attractor_properties.target.p.z=0.1
        attractor_properties.rigid_handle = body_handle

#        gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
#        gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

        attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)
        attractor_handles[i].append(attractor_handle)    


    kuka_handles.append(kuka_handle)

# Camera setup
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)



# set dof properties
for env in envs:
    gym.set_actor_dof_properties(env, kuka_handles[i], dof_props)



# a helper function to initialize all envs; set initial dof states and initial position targets
def init():
    for i in range(num_envs):
        # set updated stiffness and damping properties
        gym.set_actor_dof_properties(envs[i], kuka_handles[i], dof_props)

        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_NONE)
        for j in range(num_dofs):
            davinci_dof_states['pos'][j] = mids[j] - mids[j] * .5
        gym.set_actor_dof_states(envs[i], kuka_handles[i], davinci_dof_states, gymapi.STATE_POS)


# Let's move the robot's effector (tool yaw link) toward the attractors!
def move_daVinci_end_effector(x,y,z,rotation,index):
    gym.clear_lines(viewer)
    i = index
    j = 0
    attractor_properties = gym.get_attractor_properties(envs[i], attractor_handles[i][j])
    attr_pose = copy(base_poses[j])
    attr_pose.p.x = x 
    attr_pose.p.y = y 
    attr_pose.p.z = z            
    attr_pose.r = rotation
    gym.set_attractor_target(envs[i], attractor_handles[i][j], attr_pose)
#            gymutil.draw_lines(axes_geom, gym, viewer, envs[i], attr_pose)
#            gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], attr_pose)


# Open/close gripper
def move_gripper(i, open_gripper = True):
    pos_targets = np.zeros(num_dofs, dtype=np.float32)       
    current_position = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)        
    for j in range(num_dofs):
        pos_targets[j] = current_position[j][0]
    
    if open_gripper:
        pos_targets[-1] = 0.8
        pos_targets[-2] = 0.8
    else:
        pos_targets[-1] = 0.1
        pos_targets[-2] = 0.1       
    
    gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets)    


#def move_robot(i):

init()

start_time = 1
z_target = 0.5
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # check if we should start
    t = gym.get_sim_time(sim)
    if (t >= start_time):
        for i in range(num_envs):
#            # Move end effector to the desired pose
#            move_daVinci_end_effector(0.5, 0.5, 0.5, gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107),i)
#            
#            # Open gripper
#            move_gripper(i)
            
            if t >= 2:
                z_target -= 0.01
                move_daVinci_end_effector(0.5, 0.5, z_target, gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107),i)
                move_gripper(i, open_gripper = False)
                
            else:
                # Move end effector to the desired pose
                move_daVinci_end_effector(0.5, 0.5, 0.5, gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107),i)
            
            # Open gripper
            move_gripper(i)
        # step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)



print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
