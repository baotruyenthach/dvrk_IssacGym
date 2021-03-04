"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Depth Camera to Point Cloud Exmaple
-----------------------------------
An example which shows how to deproject the depth camera ground truth image
from gym into a 3D point cloud.

Requires pptk toolkit for viewing the resulting point cloud (pip install pptk)

Note: If pptk viewer stalls on Ubuntu, refer to https://github.com/heremaps/pptk/issues/3 (remove libz from package so it uses system libz)

"""
import open3d
import numpy as np
import pptk
from isaacgym import gymutil
from isaacgym import gymapi
import pickle
import math

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
sim_params.dt = 1.0 / 60
# sim_params.substeps = 1
if sim_type is gymapi.SIM_FLEX:
    # # Set FleX-specific parameters
    # sim_params.flex.solver_type = 5
    # sim_params.flex.num_outer_iterations = 10
    # sim_params.flex.num_inner_iterations = 200
    # sim_params.flex.relaxation = 0.75
    # sim_params.flex.warm_start = 0.8

    sim_params.flex.deterministic_mode = True

    # Set contact parameters
    sim_params.flex.shape_collision_distance = 5e-4
    sim_params.flex.contact_regularization = 1.0e-6
    sim_params.flex.shape_collision_margin = 1.0e-4
    sim_params.flex.dynamic_friction = 0.7
    # sim_params.flex.static_friction = 100
    
    sim_params.substeps = 4
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 6
    sim_params.flex.num_inner_iterations = 50
    sim_params.flex.relaxation = 0.7
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
# kuka_asset_file = "urdf/daVinci_description/robots/new_psm.urdf"
kuka_asset_file = "urdf/daVinci_description/robots/new_psm_new_gripper.urdf"

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
soft_pose.p = gymapi.Vec3(0, 0.39, 0.03)
soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
soft_thickness = 0.005    # important to add some thickness to the soft body to avoid interpenetrations

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.thickness = soft_thickness
asset_options.disable_gravity = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)







# create box asset
box_size = 0.1
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
attractor_handles = {}
soft_actors = []





print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))
base_poses = []

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)


    # add kuka
    kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", i, 1, segmentationId=3 )
    
    # # add box
    # box_pose.p.x = 0.0
    # box_pose.p.y = 0.4
    # box_pose.p.z = 0.5 * box_size
    # # box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    # box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0, segmentationId=1)
    # color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    # gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)   
    
    # add soft body + rail actor
    soft_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 1, segmentationId=2)
    soft_actors.append(soft_actor)



    kuka_handles.append(kuka_handle)



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
cam_positions.append(gymapi.Vec3(1, 1, 1))
cam_targets.append(gymapi.Vec3(0.0, 0.4, 0.0))

# Camera 1 Position and Target
cam_positions.append(gymapi.Vec3(-0.5, 0.5, 2))
cam_targets.append(gymapi.Vec3(0.0, 0.0, 0.5))

# Camera 2 Position and Target
cam_positions.append(gymapi.Vec3(2.333, 2.5, -2))
cam_targets.append(gymapi.Vec3(0.0, 0.0, 0.5))

# Camera 3 Position and Target
cam_positions.append(gymapi.Vec3(2.2, 1.5, -2))
cam_targets.append(gymapi.Vec3(0.0, 0.0, 0.5))

# Camera 4 Position and Target
cam_positions.append(gymapi.Vec3(2, 2.5, -0.5))
cam_targets.append(gymapi.Vec3(0.0, 0.0, 0.5))

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
    if frame_count == 5:
        # Array of RGB Colors, one per camera, for dots in the resulting
        # point cloud. Points will have a color which indicates which camera's
        # depth image created the point.
        color_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]])

        # Render all of the image sensors only when we need their output here
        # rather than every frame.
        gym.render_all_camera_sensors(sim)

        points = []
        color = []
        print("Converting Depth images to point clouds. Have patience...")
        for c in range(len(cam_handles)):
            print("Deprojecting from camera %d" % c)
            # Retrieve depth and segmentation buffer
            depth_buffer = gym.get_camera_image(sim, env, cam_handles[c], gymapi.IMAGE_DEPTH)
            seg_buffer = gym.get_camera_image(sim, env, cam_handles[c], gymapi.IMAGE_SEGMENTATION)
            # for stuff in depth_buffer:
            #     print(stuff)
            # print(depth_buffer.min())
            # print(depth_buffer.max())
            # print(np.where(seg_buffer > 0))
            # Get the camera view matrix and invert it to transform points from camera to world
            # space
            vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, cam_handles[c])))

            # Get the camera projection matrix and get the necessary scaling
            # coefficients for deprojection
            proj = gym.get_camera_proj_matrix(sim, env, cam_handles[c])
            fu = 2/proj[0, 0]
            fv = 2/proj[1, 1]

            # Ignore any points which originate from ground plane or empty space
            depth_buffer[seg_buffer == 1] = -10001

            centerU = cam_width/2
            centerV = cam_height/2
            for i in range(cam_width):
                for j in range(cam_height):
                    if depth_buffer[j, i] < -3:
                        continue
                    if seg_buffer[j, i] == 0:
                    # if True:
                        u = -(i-centerU)/(cam_width)  # image-space coordinate
                        v = (j-centerV)/(cam_height)  # image-space coordinate
                        d = depth_buffer[j, i]  # depth buffer value
                        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                        p2 = X2*vinv  # Inverse camera view to get world coordinates
                        if p2[0, 2] > 0.005:
                            points.append([p2[0, 0], p2[0, 1], p2[0, 2]])
                            color.append(c)
        # print(points)
        # item = [x[2] for x in points]
        # print(max(item))
        # with open("stuff/point_cloud_box.txt", 'wb') as f:
            # pickle.dump(points, f)


        
        # # use pptk to visualize the 3d point cloud created above
        v = pptk.viewer(points, color)
        v.color_map(color_map)
        # Sets a similar view to the gym viewer in the PPTK viewer
        v.set(lookat=[0, 0, 0], r=5, theta=0.4, phi=0)
        print("Point Cloud Complete")

    #     # In headless mode, quit after the deprojection is complete
    #     # The pptk viewer will remain valid until its window is closed
    #     if args.headless:
    #         break
        
    #     # Calculate bounding box:
    # #     obb = open3d.geometry.OrientedBoundingBox()
    # #     pcd = open3d.geometry.PointCloud()
    # #     pcd.points = open3d.utility.Vector3dVector(np.array(points))
    # #     obb = pcd.get_oriented_bounding_box()

    # #     points = np.asarray(obb.get_box_points())
    # #     lines = [
    # #         [0, 1],
    # #         [0, 2],
    # #         [0, 3],
    # #         [1, 6],
    # #         [1, 7],
    # #         [2, 5], 
    # #         [2, 7],
    # #         [3, 5],
    # #         [3, 6],
    # #         [4, 5],
    # #         [4, 6],
    # #         [4, 7],
    # #     ]
    # #     colors = [[1, 0, 0] for i in range(len(lines))]
    # #     line_set = open3d.geometry.LineSet(
    # #         points=open3d.utility.Vector3dVector(points),
    # #         lines=open3d.utility.Vector2iVector(lines),
    # #     )
    # #     line_set.colors = open3d.utility.Vector3dVector(colors)
    # #     open3d.visualization.draw_geometries([pcd, line_set])    

        gym.write_camera_image_to_file(sim,env,cam_handles[0], gymapi.IMAGE_COLOR, "test_images/test_rgb.png")
        gym.write_camera_image_to_file(sim,env,cam_handles[0], gymapi.IMAGE_DEPTH, "test_images/test_depth.png")
        gym.write_camera_image_to_file(sim,env,cam_handles[0], gymapi.IMAGE_SEGMENTATION, "test_images/test_segment.png")
    frame_count = frame_count + 1

    

gym.destroy_viewer(viewer)

gym.destroy_sim(sim)