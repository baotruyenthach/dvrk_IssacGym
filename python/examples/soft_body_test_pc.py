"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Soft Body
---------
Simple import of a URDF with a soft body link and rigid body press mechanism
"""

import math
import random
from isaacgym import gymapi
from isaacgym import gymutil
import pptk
import numpy as np

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="FEM Soft Body Example")
if args.physics_engine != gymapi.SIM_FLEX:
    print("*** Soft body example only supports FleX")
    print("*** Run example with --flex flag")
    quit()

random.seed(7)

# simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 3
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.75
sim_params.flex.shape_collision_margin = 0.1

# enable Von-Mises stress visualization
sim_params.stress_visualization = True
sim_params.stress_visualization_min = 0.0
sim_params.stress_visualization_max = 1.e+5

sim = gym.create_sim(args.compute_device_id, args.compute_device_id, args.physics_engine, sim_params)

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load urdf for sphere asset used to create softbody
asset_root = "../../assets"
soft_asset_file = "urdf/new_icosphere_2.urdf"

soft_thickness = 0.1    # important to add some thickness to the soft body to avoid interpenetrations

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.thickness = soft_thickness
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

# set up the env grid
num_envs = 1
spacing = 3.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# cache some common handles for later use
envs = []
soft_actors = []

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))
for i in range(num_envs):

    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 2.0, 0.0)

    # add soft body + rail actor
    soft_actor = gym.create_actor(env, soft_asset, pose, "soft", i, 1, segmentationId=1)
    soft_actors.append(soft_actor)

    # set soft material within a range of default
    actor_default_soft_materials = gym.get_actor_soft_materials(env, soft_actor)
    actor_soft_materials = gym.get_actor_soft_materials(env, soft_actor)
    for j in range(asset_soft_body_count):
        youngs = actor_soft_materials[j].youngs
        actor_soft_materials[j].youngs = random.uniform(youngs * 0.2, youngs * 2.4)

        poissons = actor_soft_materials[j].poissons
        actor_soft_materials[j].poissons = random.uniform(poissons * 0.8, poissons * 1.2)

        damping = actor_soft_materials[j].damping
        # damping is 0, instead we just randomize from scratch
        actor_soft_materials[j].damping = random.uniform(0.0, 0.08)**2

        gym.set_actor_soft_materials(env, soft_actor, actor_soft_materials)

    # enable pd-control on rail joint to allow
    # control of the press using the GUI
    gym.set_joint_target_position(env, gym.get_joint_handle(env, "soft", "rail"), 0.0)

# Point camera at environments
cam_pos = gymapi.Vec3(-4.0, 2.8, -1.2)
cam_target = gymapi.Vec3(0.0, 1.4, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# options
flag_draw_contacts = False
flag_compute_pressure = False

# Camera properties
cam_positions = []
cam_targets = []
cam_handles = []
cam_width = 480
cam_height = 320
cam_props = gymapi.CameraProperties()
cam_props.width = cam_width
cam_props.height = cam_height

# Camera 0 Position and Target
cam_positions.append(gymapi.Vec3(2, 0.5, 1.5))
cam_targets.append(gymapi.Vec3(0.0, 2.0, 0.5))

# Camera 1 Position and Target
cam_positions.append(gymapi.Vec3(-0.5, 0.5, 2))
cam_targets.append(gymapi.Vec3(0, 0, 0))

# Camera 2 Position and Target
cam_positions.append(gymapi.Vec3(2.333, 2.5, -2))
cam_targets.append(gymapi.Vec3(0, 0, 0))

# Camera 3 Position and Target
cam_positions.append(gymapi.Vec3(2.2, 1.5, -2))
cam_targets.append(gymapi.Vec3(0, 0, 0))

# Camera 4 Position and Target
cam_positions.append(gymapi.Vec3(2, 2.5, -0.5))
cam_targets.append(gymapi.Vec3(0, 0, 0))

# Create cameras in environment zero and set their locations
# to the above
env = envs[0]
for c in range(len(cam_positions)):
    cam_handles.append(gym.create_camera_sensor(env, cam_props))
    gym.set_camera_location(cam_handles[c], env, cam_positions[c], cam_targets[c])



gym.viewer_camera_look_at(viewer, envs[0], gymapi.Vec3(3, 2, 3), gymapi.Vec3(0, 0, 0))

frame_count = 0

while True:
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update graphics
    gym.step_graphics(sim)

    # Update viewer and check for exit conditions
    
    gym.draw_viewer(viewer, sim, False)

    # deprojection is expensive, so do it only once on the 2nd frame
    if frame_count == 200:
        state = gym.get_actor_rigid_body_states(env, soft_actor, gymapi.STATE_POS)
        print(state)
        state['pose']['p']['x'] -= 10
        # state['pose']['r'].fill((object_pose_stamped.pose.orientation.x,object_pose_stamped.pose.orientation.y,\
        #                         object_pose_stamped.pose.orientation.z,object_pose_stamped.pose.orientation.w))
        gym.set_actor_rigid_body_states(envs[i], soft_actor, state, gymapi.STATE_ALL)         

 

    frame_count = frame_count + 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
