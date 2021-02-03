import open3d
import numpy as np
obb = open3d.geometry.OrientedBoundingBox()
pcd = open3d.geometry.PointCloud()
np_points = np.random.rand(100, 3)*100
# print(np_points)

# From numpy to Open3D
pcd.points = open3d.utility.Vector3dVector(np_points)

obb = obb.create_from_points(pcd.points)

# open3d.visualization.draw_geometries([pcd])

#for point in bb.get_box_points:
#    np_points = np.asarray(point)
#    print(np_points)
# print(np.asarray(abc.get_box_points()))


points = np.asarray(obb.get_box_points())
lines = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 6],
    [1, 7],
    [2, 5], 
    [2, 7],
    [3, 5],
    [3, 6],
    [4, 5],
    [4, 6],
    [4, 7],
]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = open3d.geometry.LineSet(
    points=open3d.utility.Vector3dVector(points),
    lines=open3d.utility.Vector2iVector(lines),
)
line_set.colors = open3d.utility.Vector3dVector(colors)
open3d.visualization.draw_geometries([pcd, line_set])


