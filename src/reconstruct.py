
import numpy as np
import open3d as o3d
import argparse
from sklearn.neighbors import NearestNeighbors
import glob
import time
from PIL import Image 


def depth_image_to_point_cloud(rgb, depth,count):
    # camera intrinsic parameters
    fov = 90
    height, width = depth.shape
    focal_length = -(height/2)/np.tan(np.deg2rad(fov/2))
    cx ,cy = 256,256
    #matrix
    k_matrix = np.array([[focal_length, 0, cx],
                         [0, focal_length, cy],
                         [0, 0, 1]])
    #create coordinate point
    i, j = np.indices(depth.shape)
    coordinates_point = np.column_stack((j.flatten(), i.flatten(), np.ones_like(i).flatten()))

    # 2d to 3d 
    colors = (rgb /255).reshape(-1,3) #shape is (512*512,3)
    pcd  =  np.dot(np.linalg.inv(k_matrix), coordinates_point.T).T #shape is (512*512,3)
    pcd *= depth.flatten()[:, np.newaxis] / 1000.0

    # creat point cloud
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

    print(f"Point cloud {count+1} done")
    return pcd_o3d

def preprocess_point_cloud(pcd, voxel_size, counter):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    print(f"Processed Point{counter+1} Done")
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def local_icp_algorithm(source, target, transformation, voxel_size):
    distance_threshold = voxel_size * 0.4 #0.4 original

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    '''
    assert len(A) == len(B)
    # TODOs

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    W = np.dot(BB.T, AA)
    U, s, VT = np.linalg.svd(W)
    R = np.dot(U, VT)
    # special reflection case
    if np.linalg.det(R) < 0:
        VT[2,:] *= -1
        R = np.dot(U, VT)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points
    Output:
        distances: Euclidean distances (errors) of the nearest neighbor
        indecies: dst indecies of the nearest neighbor
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    
    distances, indices = neigh.kneighbors(src, return_distance=True)
    valid=distances < np.median(distances)*0.8
    return distances[valid].ravel(), indices[valid].ravel(),valid.ravel()

def my_local_icp_algorithm(A, B, init_pose=None,  max_iterations=1000000 ,tolerance=0.000002): #tuning the parameter
    # floor 1 max_iterations=1000000 ,tolerance=0.000002 

    '''
    The Iterative Closest Point method
    Input:
        A: Nx3 numpy array of source 3D points
        B: Nx3 numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
    
        T: final homogeneous transformation
        distances: Euclidean distances (errors) of the nearest neighbor
    '''
    A = np.asarray(A.points)
    B = np.asarray(B.points)
    # make points homogeneous, copy them so as to maintain the originals
    src = np.ones((4,A.shape[0]))
    dst = np.ones((4,B.shape[0]))
    src[0:3,:] = np.copy(A.T)
    dst[0:3,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0

    # main part
    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        distances, indices,valid = nearest_neighbor(src[0:3,:].T, dst[0:3,:].T)
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[0:3,valid].T, dst[0:3,indices].T)
        # update the current source
    # refer to "Introduction to Robotics" Chapter2 P28. Spatial description and transformations
        src = np.dot(T, src)

        # check error
        mean_error = np.sum(distances) / distances.size
        if abs(prev_error-mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculcate final tranformation
    T,_,_ = best_fit_transform(A, src[0:3,:].T)

    return T 

def reconstruct(args,pcd_down,pcd_fpfh):
    # TODO: Return results
    if args.version == 'open3d':
        mode = "open3d"
    elif args.version == 'my_icp':
        mode = "my_icp"
    
    #recreate
    point_cloud = o3d.geometry.PointCloud()  # create point cloud object
    point_final=[]
    point_final.append(pcd_down[0])
    
    #create camera point cloud

    pred_cam_pos = o3d.geometry.LineSet()
    camera_cloud_list = []
    for i in range(len(pcd_down)):
        camera_cloud = o3d.geometry.PointCloud()
        camera_cloud.points.append([0 ,0, 0])
        camera_cloud.colors.append([1 ,0, 0])
        camera_cloud_list.append(camera_cloud)

    # algorimn for recreate
    for i in range(1,len(pcd_down)):
        result_ransac = execute_global_registration(pcd_down[i], point_final[i-1], pcd_fpfh[i], pcd_fpfh[i-1], voxel_size)
        transformation = result_ransac.transformation
        if mode == 'open3d':
            result_icp = local_icp_algorithm(pcd_down[i], point_final[i-1], transformation,voxel_size)
            transformation = result_icp.transformation
        elif mode == 'my_icp':
            result_icp = my_local_icp_algorithm(pcd_down[i], point_final[i-1], transformation)
            transformation = result_icp

        point_final.append(pcd_down[i].transform(transformation)) 
        camera_cloud_list[i]=camera_cloud_list[i].transform(transformation)
        print(f'recreate {i} done')
    
    #camera point to line 
    camera_points = []
    for point in camera_cloud_list:
        point_coordinates = np.asarray(point.points)
        camera_points.append(point_coordinates)
    points_o3d = o3d.utility.Vector3dVector(np.concatenate(camera_points, axis=0))

    pred_cam_pos.points = points_o3d
    lines = []
    for i in range(len(camera_points) - 1):
        lines.append([i, i+1])

    pred_cam_pos.lines = o3d.utility.Vector2iVector(lines)
    line_color = [1, 0, 0]
    line_colors = [line_color] * len(lines)
    pred_cam_pos.colors = o3d.utility.Vector3dVector(line_colors)

    # final point cloud 
    for final in point_final:
        point_cloud += final
    
    #hidden ceiling
    xyz_points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    threshold_y = 0.011
    filtered_xyz_points = xyz_points[xyz_points[:, 1] <= threshold_y]
    filtered_colors = colors[xyz_points[:, 1] <= threshold_y]
    point_cloud.points = o3d.utility.Vector3dVector(filtered_xyz_points)
    point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return point_cloud  , pred_cam_pos , camera_points

def computeL2(goundtruthpoints,camera_points):

    l2 = []
    for i in range(len(goundtruthpoints)):#len(goundtruthpoints)):
        data = abs (goundtruthpoints[i]-camera_points[i])
        distance = np.sqrt(data[0][1]**2 + data[0][1]**2 + data[0][2]**2)
        l2.append(distance )
    l2_average = np.mean(l2)

    return l2_average

def goundtruth(args):
    #ground truth
    data = np.load('./'+str(args.data_root)+'GT_pose.npy')
    if args.floor == 1:#args.floor == 1:
        x = -data[:, 0]/40
        y = data[:, 1]/40
        z = -data[:, 2]/40
        rotation = data[:, 3:]  # rw, rx, ry, rz
    elif args.floor == 2: #
        x = -data[:, 0]/40-0.00582038
        y = data[:, 1]/40-0.07313087
        z = -data[:, 2]/40-0.03052245
        rotation = data[:, 3:]  # rw, rx, ry, rz

    # 創建真實軌跡的3D模型
    real_trajectory_points = np.column_stack((x, y, z))
    real_trajectory_line = o3d.geometry.LineSet()
    real_trajectory_line.points = o3d.utility.Vector3dVector(real_trajectory_points)

    # 創建線的連接
    lines = []
    for i in range(len(real_trajectory_points) - 1):
        lines.append([i, i+1])
    real_trajectory_line.lines = o3d.utility.Vector2iVector(lines)

    return real_trajectory_line ,real_trajectory_points

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"

    #data list 
    rgb_file_list = glob.glob(r'./'+str(args.data_root)+'rgb/*')
    depth_file_list = glob.glob(r'./'+str(args.data_root)+'depth/*')

    rgb_file_list = ["./"+str(args.data_root)+"rgb/{}.png".format(i+1) for i in range(len(rgb_file_list))]
    depth_file_list = ["./"+str(args.data_root)+"depth/{}.png".format(i+1) for i in range(len(depth_file_list))]

    #read image
    rgb_image = np.array([np.array(Image.open(fname)) for fname in rgb_file_list]) # data shape is (number of pic ,width , height,3)
    depth_image = np.array([np.array(Image.open(fname)) for fname in depth_file_list]) # data shape is (number of pic , width , height )

    #original point cloud 
    pcd_list = []
    camera_list = []
    pcd_list = [ depth_image_to_point_cloud(rgb_image[i],depth_image[i],i)  for i in range(len(rgb_image))] #len(rgb_image)

    #processed poind cloud
    voxel_size = 0.0028
    pcd_down = []
    pcd_fpfh = []
    for i in range(len(rgb_image)):
        pcd ,fpfh= preprocess_point_cloud(pcd_list[i], voxel_size, i) 
        pcd_down.append(pcd)
        pcd_fpfh.append(fpfh)


    #reconstruct
    #o3d.visualization.draw_geometries([reconstruct(args,pcd_down,pcd_fpfh)])
    hiddenceiling_pointcloud , camera_pose ,camera_points = reconstruct(args,pcd_down,pcd_fpfh)
    real_trajectory_line , goundtruthpoints = goundtruth(args)
    print(f'The L2 is {computeL2(goundtruthpoints,camera_points)}')
    end = time.time()
    o3d.visualization.draw_geometries([hiddenceiling_pointcloud,real_trajectory_line,camera_pose])

    
    print(f'The construction takes {end-start} second')
   