import open3d


def make_point_cloud(pts):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pts)
    return pcd
