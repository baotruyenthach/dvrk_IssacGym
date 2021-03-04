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

# for point in obb.get_box_points:
#    np_points = np.asarray(point)
#    print(np_points)

points = np.asarray(obb.get_box_points())
# print(points)
print(obb.R)
print(np.transpose(obb.R))
x_axis = (points[1]-points[0])/2
y_axis = (points[2]-points[0])/2
z_axis = (points[3]-points[0])/2
print("x", x_axis/np.linalg.norm(x_axis))
print("y", y_axis/np.linalg.norm(y_axis))
print("z", z_axis/np.linalg.norm(z_axis))
print("det oob.R", np.linalg.det(obb.R))
print("det x,y,z manual", np.linalg.det(obb.R))



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


