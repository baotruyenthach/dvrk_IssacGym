import numpy as np
from math import *
import pickle
import open3d

with open("stuff/point_cloud_box.txt", 'rb') as f:
    points = pickle.load(f)


obb = open3d.geometry.OrientedBoundingBox()
pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(np.array(points))
obb = pcd.get_oriented_bounding_box()

# points = np.asarray(obb.get_box_points())
# lines = [
#     [0, 1],
#     [0, 2],
#     [0, 3],
#     [1, 6],
#     [1, 7],
#     [2, 5], 
#     [2, 7],
#     [3, 5],
#     [3, 6],
#     [4, 5],
#     [4, 6],
#     [4, 7],
# ]
# colors = [[1, 0, 0] for i in range(len(lines))]
# line_set = open3d.geometry.LineSet(
#     points=open3d.utility.Vector3dVector(points),
#     lines=open3d.utility.Vector2iVector(lines),
# )
# line_set.colors = open3d.utility.Vector3dVector(colors)
# open3d.visualization.draw_geometries([pcd, line_set]) 

print(obb.R)
