import copy
import math
import os
import re
import shutil
import sys
import time
from math import sqrt
from multiprocessing import Process
from os import listdir, makedirs
from os.path import exists, isfile, join, splitext, isdir
import warnings
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import open3d as o3d
from cv2 import ROTATE_90_COUNTERCLOCKWISE
from scipy.special import ellipe
from sympy import FractionField, Symbol, Derivative, sympify
from sympy.solvers import solve
import time
import winsound

import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4atypes

sys.path.insert(1, '../')
folder_path = join(os.path.dirname(os.path.abspath(__file__)), "dataset")
dataset_path = join(folder_path, "Demo1")

warnings.filterwarnings("ignore")

DIV = 1 # 0-4 (Division to start running code)

"""""""""""""""""""""""""""""
0 --> DAQ (ONLY WITH DEVICE!)
1 --> Generate Pointcloud
2 --> Pointcloud Processing and Reconstruction
3 --> Surface Reconstruction & Filtering
4 --> Measurement

"""""""""""""""""""""""""""""

MODE = "Playback"  # Device/Playback
IS_VERTICAL_CAMERA = True
FILE = "Video_Test\Rotate1.mkv"

MODEL_COMPLEXITY = 1  # 0, 1, 2
IS_DRAW_SKELETON = False  # False Only in Device Mode
IS_DRAW_EFFECT = False  # True/False
EXPORT_RGB_FOLDER = "RGB_data"
EXPORT_DEPTH_FOLDER = "DEPTH_data"

# For Pointcloud in Depth view --> need live camera (BGRA Format) #
# If create Pointcloud in RGB view (In case of Playback) --> may cause Pointcloud Distortion #
PCD_VIEW = "Depth"  # Depth/RGB 
EXPORT_POINTCLOUD_FOLDER = "pointcloud_data"
EXPORT_POINTCLOUD_SKEL_FOLDER = "skeleton_track"
EXPORT_FULL_BODY_POINTCLOUD_FOLDER = "full_body_pcd"
IS_FINE_SEGMENT = True  # True/False

GUI = True
DEBUG_MSG = False
DEBUG_PLOT = False

Frequency = 1000
Duration_Beep = 250

### File Handling Function  ###

def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        shutil.rmtree(path_folder)
        makedirs(path_folder)


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            join(path, f)
            for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list


def load_pcd_folder_to_list(filename):
    pcd_List = []
    ply_file_names = get_file_list(
        join(dataset_path, filename), ".ply")

    for file in ply_file_names:
        pcd = o3d.io.read_point_cloud(file)
        pcd_List.append(pcd)
    return pcd_List


### Math Function ###

def get_rotation_matrix(x_angle=0, y_angle=0, z_angle=0):
    if x_angle != 0:
        c = round(math.cos(math.radians(x_angle)), 2)
        s = round(math.sin(math.radians(x_angle)), 2)
        Rx = np.array([[1, 0, 0, 0],
                       [0, c, -s, 0],
                       [0, s, c, 0],
                       [0, 0, 0, 1]])
    else:
        Rx = np.identity(4)
    if y_angle != 0:
        c = round(math.cos(math.radians(y_angle)), 2)
        s = round(math.sin(math.radians(y_angle)), 2)
        Ry = np.array([[c, 0, s, 0],
                       [0, 1, 0, 0],
                       [-s, 0, c, 0],
                       [0, 0, 0, 1]])
    else:
        Ry = np.identity(4)
    if z_angle != 0:
        c = round(math.cos(math.radians(z_angle)), 2)
        s = round(math.sin(math.radians(z_angle)), 2)
        Rz = np.array([[c, -s, 0, 0],
                       [s, c, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    else:
        Rz = np.identity(4)
    rotation_matrix = np.matmul(np.matmul(Rz, Ry), Rx)
    return rotation_matrix


def get_translation_matrix(Tx=0, Ty=0, Tz=0):
    rotation_matrix = np.array([[1, 0, 0, Tx],
                                [0, 1, 0, Ty],
                                [0, 0, 1, Tz],
                                [0, 0, 0, 1]])
    return rotation_matrix


def get_projection_xy_matrix():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 1]])


def get_projection_zy_matrix():
    return np.array([[0, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def point_at_y(np_pcd, y, range=10):
    xz = np.array([])
    # ===== X->X, Z->Y =====
    for coord in np_pcd:
        if abs(coord[1] - y) < range:
            # print ("x = {x},\tz = {z} ".format(x=coord[0], z=coord[2]))
            xz = np.append(xz, np.array([coord[0], coord[2]]))  # [x, z]

    xz = xz.reshape(-1, 2)
    xz = xz[xz[:, 0].argsort()]  # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    return xz  # Sorted, from less x-value to most x-value


def point_at_z(np_pcd, z, range=10):
    xy = np.array([])
    for coord in np_pcd:
        if abs(coord[2] - z) < range:
            xy = np.append(xy, np.array([coord[0], coord[1]]))  # [x, y]
    xy = xy.reshape(-1, 2)
    xy = xy[xy[:, 0].argsort()]  # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    # np.savetxt('z.csv',zz,delimiter = ',')

    return xy


def bound_at_y(pcd, y, range=10, prime_axis=0, is_second_axis=False):
    first_iter = True
    min = 0
    max = 0
    second_axis = 2 if prime_axis == 0 else 0
    for coord in pcd:
        if abs(coord[1] - y) < range:
            if not first_iter:
                if coord[prime_axis] < min[prime_axis]: min = coord
                if coord[prime_axis] > max[prime_axis]: max = coord
            elif first_iter:
                min = coord
                max = coord
                first_iter = False
    if is_second_axis:
        return max[prime_axis], min[prime_axis], max[second_axis], min[second_axis]
    return max[prime_axis], min[prime_axis]


def bound_at_x(pcd, x, range=10, prime_axis=1):
    first_iter = True
    min = 0
    max = 0
    for coord in pcd:
        if abs(coord[0] - x) < range:
            if not first_iter:
                if coord[prime_axis] < min[prime_axis]: min = coord
                if coord[prime_axis] > max[prime_axis]: max = coord
            elif first_iter:
                min = coord
                max = coord
                first_iter = False
    return max[prime_axis], min[prime_axis]


### Image Processing Function ###

class poseDetector():

    def __init__(self, mode=False, modelComplexity=MODEL_COMPLEXITY, smooth_Landmark=False, segmentation=True,
                 smooth_segmentation=False, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.modelComplex = modelComplexity
        self.smooth_Landmark = smooth_Landmark
        self.segmentation = segmentation
        self.smooth_seg = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComplex, self.smooth_Landmark,
                                     self.segmentation, self.smooth_seg, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def findPose(self, img, drawSegmen=True, drawSkel=True, plot=False, drawSegmen_Edge=False, Mode="Playback"):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        annotated_image = img.copy()
        if Mode == "Playback":
            BG_COLOR = (0, 0, 0)
            n = 3
        elif Mode == "Device":
            BG_COLOR = (0, 0, 0, 0)
            n = 4

        if self.results.pose_landmarks:
            if drawSegmen:
                if self.results.segmentation_mask.any():
                    condition = np.stack(
                        (self.results.segmentation_mask,) * n, axis=-1) > 0.1
                    bg_image = np.zeros(img.shape, dtype=np.uint8)
                    bg_image[:] = BG_COLOR
                    annotated_image = np.where(
                        condition, annotated_image, bg_image)
            if drawSkel:
                self.mpDraw.draw_landmarks(
                    annotated_image,
                    self.results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

            if plot:
                self.mpDraw.plot_landmarks(
                    self.results.pose_world_landmarks, self.mpPose.POSE_CONNECTIONS)

            if drawSegmen_Edge:
                annotated_image = 255 * np.ones(img.shape, dtype=np.uint8)
                if self.results.segmentation_mask.any():
                    condition = np.stack(
                        (self.results.segmentation_mask,) * n, axis=-1) > 0.1
                    bg_image = np.zeros(img.shape, dtype=np.uint8)
                    bg_image[:] = BG_COLOR
                    annotated_image = np.where(
                        condition, annotated_image, bg_image)

        return annotated_image

    def findPosition(self, img):
        self.lmList = []
        self.segmentation_mask = []
        if self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                if lm.x < 1 and lm.y < 1:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy, w * cy + cx])
            if self.results.segmentation_mask.any():
                self.segmentation_mask = np.round(
                    self.results.segmentation_mask, decimals=1)
        return self.lmList, self.segmentation_mask

    def findAngle(self, img, p1, p2, p3, draw=True):
        try:
            x1, y1 = self.lmList[p1][1:3]
            x2, y2 = self.lmList[p2][1:3]
            x3, y3 = self.lmList[p3][1:3]
        except IndexError:
            return 0

        # Calculate the Angle
        angle = abs(math.degrees(-math.atan2(y3 - y2, x3 - x2) +
                                 math.atan2(y1 - y2, x1 - x2)))

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)

            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)

            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

            cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50),
                        cv2.cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return angle

    def findDistance(self, img, p1, p2, draw=False):
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]

        distance = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.circle(img, (x1, y1), 7, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 12, (255, 0, 0), 2)

            cv2.circle(img, (x2, y2), 7, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 12, (255, 0, 0), 2)
            cv2.putText(img, str(int(distance)), (x2 - 20, y2 + 50),
                        cv2.cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return distance


def concat_vh(list_2d, dimension, interpolation=cv2.INTER_CUBIC):
    # return final image
    for img_list in range(len(list_2d)):
        list_2d[img_list] = [cv2.resize(img,
                                        dimension, interpolation=interpolation)
                             for img in list_2d[img_list]]
    return cv2.vconcat([cv2.hconcat(list_h)
                        for list_h in list_2d])


### Point Cloud Processing Function ###

def create_xy_table(playback_calibration, depth_height, depth_width):
    p = _k4atypes.k4a_float2_t()
    table_x_data = []
    table_y_data = []
    t = 0 if PCD_VIEW == "Depth" else 1
    for y in range(depth_height):
        p.xy.y = float(y)
        for x in range(depth_width):
            p.xy.x = float(x)
            target_point3d = playback_calibration.convert_2d_to_3d(
                p, 1.0, t, t)  # 0 -> enumeration in C = K4A_CALIBRATION_TYPE_DEPTH 1-> for TYPE_RGB
            table_x_data.append(target_point3d.xyz.x)
            table_y_data.append(target_point3d.xyz.y)
    return table_x_data, table_y_data


def generate_point_cloud(d_depth_array, d_rgb_array, d_segmentation_mask, x_table_data, y_table_data, skel_position):
    ### Create point cloud from depth array ###
    point_cloud_data_x = x_table_data * d_depth_array
    point_cloud_data_y = y_table_data * d_depth_array
    point_cloud_data_z = d_depth_array

    ### Combine three coordinate array ###
    point_cloud_data = np.array(
        np.stack((point_cloud_data_x, point_cloud_data_y, point_cloud_data_z), axis=1))

    ### Map 2D skeleton joint to 3D point cloud ###
    skel_point_cloud_data = map_skel_to_point_cloud(
        skel_position, point_cloud_data)

    ### Maximum depth of human body skel plus estimated width of human body ###
    thresholds = max(
        skel_point_cloud_data[23][2], skel_point_cloud_data[24][2]) + 240

    if not IS_FINE_SEGMENT:
        thresholds = 10000

    ### Segmentation ###
    point_cloud_rgb = [d_rgb_array[i] for i in range(np.shape(
        d_rgb_array)[0]) if point_cloud_data[i][2] != 0 and d_segmentation_mask[i] != 0.0
                       and point_cloud_data[i][2] < thresholds]
    point_cloud_data = [point_cloud_data[i] for i in range(np.shape(point_cloud_data)[
                                                               0]) if
                        point_cloud_data[i][2] != 0 and d_segmentation_mask[i] != 0.0
                        and point_cloud_data[i][2] < thresholds]

    return point_cloud_data, point_cloud_rgb, skel_point_cloud_data


def map_skel_to_point_cloud(skel_position, point_cloud_data_org):
    skel_list = []
    for lm in skel_position:
        skel_list.append(point_cloud_data_org[lm[3]])
    return (skel_list)


def point_cloud_worker(element, depth_width, depth_height, x_table_data, y_table_data, file_index):
    depth_image, modified_rgb_image, \
    skel_position, segmentation_mask = element
    ### CV use BGR format / Open3D need RGB ###
    modified_rgb_image = cv2.cvtColor(modified_rgb_image, cv2.COLOR_BGR2RGB)

    #### Array to 1D array ####
    d_depth_array = np.reshape(
        np.array(depth_image), depth_width * depth_height)

    d_rgb_array = modified_rgb_image.reshape(depth_width * depth_height, 3)

    d_segmentation_mask = np.reshape(
        np.array(segmentation_mask), depth_width * depth_height)

    ## Generate Point Cloud ###
    point_cloud_data, point_cloud_rgb, skel_point_cloud_data = generate_point_cloud(
        d_depth_array, d_rgb_array, d_segmentation_mask, x_table_data, y_table_data, skel_position)

    ### Use Open3D Library ###
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

    pcd_skel = o3d.geometry.PointCloud()
    pcd_skel.points = o3d.utility.Vector3dVector(skel_point_cloud_data)

    ### Rotate point cloud coordinate frame ###
    Rotation_Matrix = get_rotation_matrix(x_angle=180)
    pcd.transform(Rotation_Matrix)
    pcd_skel.transform(Rotation_Matrix)

    ### Add RGB to point cloud ###
    pcd.colors = o3d.utility.Vector3dVector(
        np.divide(point_cloud_rgb, 255))

    ### Create file for point cloud ###
    filename_ply = join(dataset_path, EXPORT_POINTCLOUD_FOLDER) + \
                   "/data" + str(file_index) + ".ply"
    o3d.io.write_point_cloud(filename_ply, pcd, write_ascii=True)

    filename_skel_ply = join(
        dataset_path, EXPORT_POINTCLOUD_SKEL_FOLDER) + "/data" + str(file_index) + ".ply"
    o3d.io.write_point_cloud(filename_skel_ply, pcd_skel, write_ascii=True)


### Point Cloud Reconstruction Function ###


def init_pointcloud_visualizer():
    vis = o3d.visualization.VisualizerWithKeyCallback()
    global rotating
    rotating = False

    def change_background_to_black(vis, action, mods):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def change_background_to_white(vis, action, mods):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([255, 255, 255])
        return False

    def key_action_callback(vis, action, mods):
        global rotating
        if action == 1:  # key down
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
        elif action == 0:  # key up
            pass
        elif action == 2:  # key repeat
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
        return True

    def animation_callback(vis):
        global rotating
        if rotating:
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)

    # key_action_callback will be triggered when there's a keyboard press
    vis.register_key_action_callback(ord("B"), change_background_to_black)
    vis.register_key_action_callback(ord("W"), change_background_to_white)
    vis.register_key_action_callback(32, key_action_callback)
    # animation_callback is always repeatedly called by the visualizer
    vis.register_animation_callback(animation_callback)

    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])

    return vis


def preprocess_point_cloud(pcd, voxel_size, scale):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2 * scale
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5 * scale
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def fast_global_registration(target, source, voxel_size=0.05, scale=1, is_projected=False):
    front_part_temp = copy.deepcopy(target)
    back_part_temp = copy.deepcopy(source)

    if is_projected:
        projection_matrix = get_projection_zy_matrix()
        front_part_temp.transform(projection_matrix)
        back_part_temp.transform(projection_matrix)

    source_down, source_fpfh = preprocess_point_cloud(
        back_part_temp, voxel_size, scale)
    target_down, target_fpfh = preprocess_point_cloud(
        front_part_temp, voxel_size, scale)

    distance_threshold = voxel_size * 1.4
    # print(":: Apply fast global registration with distance threshold %.3f" \
    # % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))

    return result


def get_hip_dimension(pcd_front_np, pcd_back_np, pcd_side1_np, pcd_side2_np, pcd_front_skel_np, scaling_width=1,
                      scaling_thick=1):
    hip_y_avg = (pcd_front_skel_np[23][1] + pcd_front_skel_np[24][1]) / 2
    # print (hip_y_avg)

    # THICKNESS
    max_x_one, min_x_one = bound_at_y(pcd_side1_np, hip_y_avg, range=10)
    thick_side_one = max_x_one - min_x_one

    max_x_two, min_x_two = bound_at_y(pcd_side2_np, hip_y_avg, range=10)
    thick_side_two = max_x_two - min_x_two

    avg_thick = (thick_side_one + thick_side_two) / 2 * scaling_thick
    # print(avg_thick)

    # WIDTH
    max_x_front, min_x_front = bound_at_y(pcd_front_np, hip_y_avg, range=10)
    width_front = max_x_front - min_x_front

    max_x_back, min_x_back = bound_at_y(pcd_back_np, hip_y_avg, range=10)
    width_back = max_x_back - min_x_back

    avg_width = (width_front + width_back) / 2 * scaling_width

    if DEBUG_MSG:
        print(
            "\n============================ get_hip_dimension ============================")
        print("DIM@HIP - thick_side_one is {x} mm.".format(x=thick_side_one))
        print("DIM@HIP - thick_side_two is {x} mm.".format(x=thick_side_two))
        print("DIM@HIP - width_front is {x} mm.".format(x=width_front))
        print("DIM@HIP - width_back is {x} mm.".format(x=width_back))
        print("===========================================================================")
        print("DIM@HIP - avg_width is {x} mm.".format(x=avg_width))
        print("DIM@HIP - avg_thick is {x} mm.".format(x=avg_thick))
        print("===========================================================================")
        print("HIP_MEASUREMENT : THICKNESS = {x:.1f} cm".format(x=avg_thick/10))
        
    return hip_y_avg, avg_width, avg_thick


class pcd_Pose():

    def __init__(self, pcd, pcd_skel):
        self.pcd = pcd
        self.pcd_points_np = np.asarray(pcd.points)
        self.pcd_colors_np = np.asarray(pcd.colors)
        self.pcd_skel = pcd_skel
        self.pcd_skel_np = np.asarray(pcd_skel.points)
        self.pcd_upper_left_part = None
        self.pcd_upper_right_part = None
        self.pcd_torso_part = None
        self.pcd_head_part = None

    def get_human_part_decomposition(self, mode=0):
        pcd_count = np.shape(self.pcd_colors_np)[0]
        if mode is 0:
            ### Get Upper Left ###
            y_threshold1 = (self.pcd_skel_np[24][1] + 2 * self.pcd_skel_np[12][1]) / 3
            x_threshold1 = (3 * self.pcd_skel_np[12][0] + self.pcd_skel_np[14][0]) / 4

            upper_left_part_color_np = [self.pcd_colors_np[i] for i in range(pcd_count) if
                                        self.pcd_points_np[i][0] < x_threshold1
                                        and self.pcd_points_np[i][1] > y_threshold1]

            upper_left_part_np = [self.pcd_points_np[i] for i in range(pcd_count) if
                                  self.pcd_points_np[i][0] < x_threshold1
                                  and self.pcd_points_np[i][1] > y_threshold1]

            self.pcd_upper_left_part = o3d.geometry.PointCloud()
            self.pcd_upper_left_part.points = o3d.utility.Vector3dVector(upper_left_part_np)
            self.pcd_upper_left_part.colors = o3d.utility.Vector3dVector(
                upper_left_part_color_np)

            ### Get Upper Right ###
            y_threshold2 = (self.pcd_skel_np[23][1] + 2 * self.pcd_skel_np[11][1]) / 3
            x_threshold2 = (3 * self.pcd_skel_np[11][0] + self.pcd_skel_np[13][0]) / 4
            upper_right_part_color_np = [self.pcd_colors_np[i] for i in range(pcd_count) if
                                         self.pcd_points_np[i][0] > x_threshold2
                                         and self.pcd_points_np[i][1] > y_threshold2]

            upper_right_part_np = [self.pcd_points_np[i] for i in range(pcd_count) if
                                   self.pcd_points_np[i][0] > x_threshold2
                                   and self.pcd_points_np[i][1] > y_threshold2]

            self.pcd_upper_right_part = o3d.geometry.PointCloud()
            self.pcd_upper_right_part.points = o3d.utility.Vector3dVector(upper_right_part_np)
            self.pcd_upper_right_part.colors = o3d.utility.Vector3dVector(upper_right_part_color_np)

            ### Get Torso ###
            x_threshold1 = self.pcd_skel_np[12][0]
            x_threshold2 = self.pcd_skel_np[11][0]
            y_threshold3 = self.pcd_skel_np[12][1] + 30

            torso_part_color_np = [self.pcd_colors_np[i] for i in range(pcd_count) if (
                    self.pcd_points_np[i][0] > x_threshold1 and self.pcd_points_np[i][0] < x_threshold2 and
                    self.pcd_points_np[i][1] < y_threshold3) or (self.pcd_points_np[i][1] < y_threshold1)]
            torso_part_np = [self.pcd_points_np[i] for i in range(pcd_count) if (
                    self.pcd_points_np[i][0] > x_threshold1 and self.pcd_points_np[i][0] < x_threshold2 and
                    self.pcd_points_np[i][1] < y_threshold3) or (self.pcd_points_np[i][1] < y_threshold1)]

            self.pcd_torso_part = o3d.geometry.PointCloud()
            self.pcd_torso_part.points = o3d.utility.Vector3dVector(torso_part_np)
            self.pcd_torso_part.colors = o3d.utility.Vector3dVector(torso_part_color_np)

            ### Get Head ###
            y_threshold3 = (self.pcd_skel_np[12][1] + self.pcd_skel_np[0][1])/2
            head_part_color_np = [self.pcd_colors_np[i] for i in range(pcd_count) if
                                  (self.pcd_points_np[i][1] > y_threshold3)]

            head_part_np = [self.pcd_points_np[i] for i in range(pcd_count) if
                            (self.pcd_points_np[i][1] > y_threshold3)]

            self.pcd_head_part = o3d.geometry.PointCloud()
            self.pcd_head_part.points = o3d.utility.Vector3dVector(head_part_np)
            self.pcd_head_part.colors = o3d.utility.Vector3dVector(head_part_color_np)

        elif mode is 1:
            ### Get Torso ###
            z_threshold = self.pcd_skel_np[24][2]
            y_threshold = (self.pcd_skel_np[24][1] + 2 * self.pcd_skel_np[12][1]) / 3
            # and self.pcd_points_np[i][1] < y_threshold2 in case want headcut off
            y_threshold2 = self.pcd_skel_np[12][1] + 30
            torso_part_color_np = [self.pcd_colors_np[i] for i in range(pcd_count) if (
                    self.pcd_points_np[i][1] > y_threshold and self.pcd_points_np[i][2] < z_threshold) or (
                                           self.pcd_points_np[i][1] < y_threshold)]
            torso_part_np = [self.pcd_points_np[i] for i in range(pcd_count) if (
                    self.pcd_points_np[i][1] > y_threshold and self.pcd_points_np[i][2] < z_threshold) or (
                                     self.pcd_points_np[i][1] < y_threshold)]
            self.pcd_torso_part = o3d.geometry.PointCloud()
            self.pcd_torso_part.points = o3d.utility.Vector3dVector(torso_part_np)
            self.pcd_torso_part.colors = o3d.utility.Vector3dVector(torso_part_color_np)

    def transform_pcd(self, matrix):
        self.pcd.transform(matrix)
        self.pcd_skel.transform(matrix)
        self.pcd_points_np = np.asarray(self.pcd.points)
        self.pcd_skel_np = np.asarray(self.pcd_skel.points)
        if self.pcd_torso_part:  self.pcd_torso_part.transform(matrix)
        if self.pcd_upper_left_part: self.pcd_upper_left_part.transform(matrix)
        if self.pcd_upper_right_part: self.pcd_upper_right_part.transform(matrix)

    def add_pcd(self, new_pcd):
        self.pcd = self.pcd + new_pcd
        self.pcd_points_np = np.asarray(self.pcd.points)
        self.pcd_colors_np = np.asarray(self.pcd.colors)

    @classmethod
    def merge_front_and_back_pcd(cls, front_Pose, back_Pose, ref_avg_thick, ref_y):

        ### Find boundary of front and back point cloud ###
        max_z_front, min_z_front = bound_at_y(
            front_Pose.pcd_points_np, ref_y, prime_axis=2)

        max_z_back, min_z_back = bound_at_y(
            back_Pose.pcd_points_np, ref_y, prime_axis=2)

        ### Calculate translation distance ###
        translation_dist = abs(max_z_back - min_z_front)
        merged_body_dist = abs(max_z_front - min_z_front) + abs(max_z_back - min_z_back)
        added_translation_dist = translation_dist + (ref_avg_thick - merged_body_dist)

        back_Pose.transform_pcd(get_translation_matrix(Tz=-added_translation_dist))

        pcd_full_body = front_Pose.pcd + back_Pose.pcd_torso_part
        pcd_full_body_skel = front_Pose.pcd_skel + back_Pose.pcd_skel

        return cls(pcd_full_body, pcd_full_body_skel)

    ### Fix this if we have CCW rotation ###
    @classmethod
    def merge_side_one_and_two_pcd(cls, full_Pose, side1_Pose, side2_Pose, ref_avg_width, ref_y):

        max_x_full, min_x_full = bound_at_y(full_Pose.pcd_points_np, ref_y, prime_axis=0)
        max_z_full, min_z_full = bound_at_y(full_Pose.pcd_points_np, ref_y, prime_axis=2)

        max_x_side1, min_x_side1 = bound_at_y(side1_Pose.pcd_points_np, ref_y, prime_axis=0)
        max_z_side1, min_z_side1 = bound_at_y(side1_Pose.pcd_points_np, ref_y, prime_axis=2)

        z_translation_side1 = ((max_z_full - max_z_side1) + (min_z_full - min_z_side1)) / 2
        x_translation_side1 = max_x_full - max_x_side1

        max_x_side2, min_x_side2 = bound_at_y(side2_Pose.pcd_points_np, ref_y, prime_axis=0)
        max_z_side2, min_z_side2 = bound_at_y(side2_Pose.pcd_points_np, ref_y, prime_axis=2)

        z_translation_side2 = ((max_z_full - max_z_side2) + (min_z_full - min_z_side2)) / 2
        x_translation_side2 = min_x_full - min_x_side2

        ### Adjust translation to be equal to calculated width ###
        width_to_adjust = ((max_x_side1 + x_translation_side1) - (
                min_x_side2 + x_translation_side2) - ref_avg_width) / 2
        x_translation_side1 -= width_to_adjust
        x_translation_side2 += width_to_adjust

        side1_Pose.transform_pcd(get_translation_matrix(Tx=x_translation_side1, Tz=z_translation_side1))
        side2_Pose.transform_pcd(get_translation_matrix(Tx=x_translation_side2, Tz=z_translation_side2))

        # max_x_side1, min_x_side1  = bound_at_y(side1_Pose.pcd_points_np, ref_y, prime_axis=0)
        # max_x_side2, min_x_side2  = bound_at_y(side2_Pose.pcd_points_np, ref_y, prime_axis=0)
        # print(max_x_side1-min_x_side2, ref_avg_width,width_to_adjust)

        full_Pose.add_pcd(side1_Pose.pcd_torso_part)
        full_Pose.add_pcd(side2_Pose.pcd_torso_part)

        return full_Pose

    def reflect_arm(self):
        if self.pcd_upper_left_part is None:
            return

        upper_left_part = copy.deepcopy(self.pcd_upper_left_part)
        upper_right_part = copy.deepcopy(self.pcd_upper_right_part)


### Measurement Function ###
def pcd_ellipse_plot(X, Y, x):
    fig, ax = plt.subplots()
    plt.scatter(X, Y, label='Data Points')
    bound = 50
    x_coord = np.linspace(X[0] - bound, X[-1] + bound, 1000)
    y_coord = np.linspace(np.min(Y) - bound, np.max(Y + bound), 1000)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord ** 2 + x[3] * X_coord + x[4] * Y_coord
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2).legend_elements(
        "Least Square Ellipse")

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_aspect('equal')
    ax.autoscale()
    plt.show()
    
    
def ellipse_finder(np_pcd, offset, range=10,
                   plane_to_project=1, is_plot = False):  # https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
    if plane_to_project == 1: projected_point = point_at_y(np_pcd, offset, range)
    if plane_to_project == 2: projected_point = point_at_z(np_pcd, offset, range)

    X = projected_point[:, 0].reshape(-1, 1)
    Y = projected_point[:, 1].reshape(-1, 1)

    # Formulate and solve the least squares problem ||Ax - b||^2
    A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

    # Print the equation of the ellipse in standard form
    if DEBUG_MSG: print(
        '\nThe ellipse is given by {0:.3}x^2 + {1:.3}xy + {2:.3}y^2 + {3:.3}x + {4:.3}y = 1'.format(x[0], x[1], x[2],
                                                                                                    x[3], x[4]))

    # Plot the noisy data
    # Plot the least squares ellipse
    if is_plot:
        pcd_ellipse_plot(X, Y, x)

    return x  # coeff


def measure_curve_human_part(pcd, y, range=10,is_plot = False):
    a, b, c, d, e = ellipse_finder(pcd, y, range, plane_to_project=1 ,is_plot = is_plot) #coeff: f = -1
    # This shit is a real savior - https://math.stackexchange.com/questions/616645/determining-the-major-minor-axes-of-an-ellipse-from-general-form
    # The coefficient normalizing factor is given by:
    q = 64*((-1*(4*a*c - b**2) - a*e**2 + b*d*e - c*d**2)/(4*a*c - b**2)**2)
    # The distance between center and focal point is given by:
    s = 0.25*sqrt(abs(q)*sqrt(b**2 + (a - c)**2))
  
    # The semi-major axis length is given by:
    major = float(0.125*sqrt(2*abs(q)*sqrt(b**2 + (a - c)**2) - 2*q*(a + c)))

    # The semi-minor axis length is given by:
    minor = float(sqrt(major**2 - s**2))

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html
    e_sq = 1.0 - (minor**2/major**2)
    p = 4*major*ellipe(e_sq)

    if DEBUG_MSG:
        print("\nMEASURE_CURVE")
        print("=============================================")
        print("MEASURE_CURVE : focal dist = {x} m".format(x=s / 1000))
        print("MEASURE_CURVE : major = {x} m".format(x=major / 1000))
        print("MEASURE_CURVE : minor = {x} m".format(x=minor / 1000))
        print("MEASURE_CURVE : perimeter = {x} m".format(x=p / 1000))
        print("MEASURE_CURVE : perimeter = {x} in".format(x=p / 25.4))

    return p


def waist_measurement(pcd_np, pcd_skel_np, range=10):
    y_shoulder = (pcd_skel_np[11][1] + pcd_skel_np[12][1]) / 2
    y_hip = (pcd_skel_np[23][1] + pcd_skel_np[24][1]) / 2
    y_start = y_hip + 0.1 * (y_shoulder - y_hip)
    y_stop = y_hip + 0.6 * (y_shoulder - y_hip)

    step = 5
    y = y_start
    yp = np.array([])

    while (y < y_stop):
        yp = np.append(yp, np.array([y, measure_curve_human_part(pcd_np, y, range)]))  # [y, p]
        y = y + step

    yp = yp.reshape(-1, 2)

    curve_poly = np.polyfit(yp[:, 0], yp[:, 1], 2)

    x = Symbol('x')
    eq = sympify(curve_poly[0] * x ** 2 + curve_poly[1] * x + curve_poly[2])

    deriv = Derivative(eq, x).doit()
    critical = solve(deriv, x)[0]
    peri = eq.subs(x, critical).evalf() # milimeter
    
    if DEBUG_MSG: print("\nWAIST_MEAS : eq = {x}".format(x=eq))
    if DEBUG_MSG: print("WAIST_MEAS : deriv = {x}".format(x=deriv))
    if DEBUG_MSG: print("WAIST_MEAS : critical = {x}".format(x=critical))
    if DEBUG_MSG: print("WAIST_MEASUREMENT : Perimeter = {x:.1f} in".format(x=peri / 25.4))
    
    if DEBUG_PLOT: p  = measure_curve_human_part(pcd_np, critical, range, is_plot=True)

    return peri/25.4


def hip_measurement(pcd_np, pcd_skel_np, range=10):
    if DEBUG_MSG: print("\n========== HIP_MEAS ==========")

    y_hip = (pcd_skel_np[23][1] + pcd_skel_np[24][1]) / 2
    y_knee = (pcd_skel_np[25][1] + pcd_skel_np[26][1]) / 2
    y_start = y_knee + 0.8 * (y_hip - y_knee)
    y_stop = y_hip

    step = 5
    y = y_start
    peri = 0

    while (y < y_stop):
        p = measure_curve_human_part(pcd_np, y, range)
        if p>peri :
            peri =  p
            y_pmax = y

        if DEBUG_MSG: print("HIP_MEAS : p = {x:} in".format(x=p / 25.4))

        y = y + step

    if DEBUG_MSG: print("HIP_MEASUREMENT : Perimeter = {x:.1f} in".format(x=peri / 25.4))

    if DEBUG_PLOT: p  = measure_curve_human_part(pcd_np, y_pmax, range, is_plot=True)
    
    return peri/25.4


def measure_head_part(pcd, pcd_skel, range=10):

    Head_z_avg = (pcd_skel[11][2] + pcd_skel[12][2]) / 2 - 30

    a, b, c, d, e = ellipse_finder(pcd, Head_z_avg, range, plane_to_project=2)  # coeff: f = -1

    # https://math.stackexchange.com/questions/616645/determining-the-major-minor-axes-of-an-ellipse-from-general-form
    # The coefficient normalizing factor is given by:
    q = 64 * ((-1 * (4 * a * c - b ** 2) - a * e ** 2 + b * d * e - c * d ** 2) / (4 * a * c - b ** 2) ** 2)
    # The distance between center and focal point is given by:
    s = 0.25 * sqrt(abs(q) * sqrt(b ** 2 + (a - c) ** 2))

    # The semi-major axis length is given by:
    major = float(0.125 * sqrt(2 * abs(q) * sqrt(b ** 2 + (a - c) ** 2) - 2 * q * (a + c)))
    if major**2 - s**2 < 0:
         return None
    # The semi-minor axis length is given by:
    minor = float(sqrt(major ** 2 - s ** 2))

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html
    e_sq = 1.0 - (minor ** 2 / major ** 2)
    p = 4 * major * ellipe(e_sq)
    if DEBUG_MSG:
        print("\nMEASURE_HEAD_CURVE")
        print("=============================================")
        print("MEASURE_CURVE : focal dist = {x} m".format(x=s / 1000))
        print("MEASURE_CURVE : major = {x} m".format(x=major / 1000))
        print("MEASURE_CURVE : minor = {x} m".format(x=minor / 1000))
        print("MEASURE_CURVE : perimeter = {x} m".format(x=p / 1000))
        print("MEASURE_CURVE : perimeter = {x} in".format(x=p / 25.4))

    return major * 2


def measure_human_height(pcd, pcd_skel, head_length):
    full_body_Pose = pcd_Pose(pcd, pcd_skel)
    landmark_xyz = full_body_Pose.pcd_skel_np
    
    highest_pc,lowest_pc = bound_at_x(full_body_Pose.pcd_points_np, (landmark_xyz[11][0]+landmark_xyz[12][0])/2,range = 5000)
    height_highestwithlowest = highest_pc - lowest_pc
    
    if head_length != None:
        # height_fromhead = head_length*(1/2)+(landmark_xyz[2][1]+landmark_xyz[5][1])/2- lowest_pc
        height_fromhead = head_length+(landmark_xyz[12][1] + landmark_xyz[0][1])/2- lowest_pc
        
        final_height = max(height_highestwithlowest, height_fromhead)
        if DEBUG_MSG: print("HEIGHT_MEASUREMENT : Height = {x:.1f} cm".format(x=final_height / 10))
        if DEBUG_MSG: print("HEIGHT_MEASUREMENT(MAX-MIN) : Height = {x:.1f} cm".format(x=height_highestwithlowest / 10))
        # print(height_fromhead, height_highestwithlowest )
    else:
        final_height = height_highestwithlowest
        if DEBUG_MSG: print("HEIGHT_MEASUREMENT : Height = {x:.1f} cm".format(x=final_height / 10))
        
    final_height = height_highestwithlowest
    return final_height/10

### Main Function ###

def init_k4a():
    ##### Initialize the library, if the library is not found, add the library path as argument ####
    pykinect.initialize_libraries()

    ### Device camera or Playback mode ###
    mode = MODE

    if mode == "Playback":
        video_filename = FILE

        ##### Start playback ####
        device = pykinect.start_playback(video_filename)

        device_length = device.get_recording_length() / 1000000
        # print("Recording is {:0.2f} seconds long".format(device_length))

        device_config = device.get_record_configuration()
        # print(device_config)

        device_calibration = device.get_calibration()

    elif mode == "Device":
        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15

        # Start device
        video_filename = "Video_Test\Rotate1.mkv"
        device = pykinect.start_device(config=device_config, record=False)
        device_calibration = device.get_calibration(
            pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED, pykinect.K4A_COLOR_RESOLUTION_720P)
    return device, device_calibration


def get_key_poses_extraction(device):
    ### Create dataset_path folder ###
    if not isdir(dataset_path):
        makedirs(dataset_path)

    make_clean_folder(join(dataset_path, EXPORT_DEPTH_FOLDER))
    make_clean_folder(join(dataset_path, EXPORT_RGB_FOLDER))

    detector = poseDetector()

    key_poses_list = np.array([0, 0, 0, 0], dtype=object)
    start_clock = False
    key_pose_mode = 1
    cv2.namedWindow("Captured Image", cv2.WINDOW_AUTOSIZE)
    blank_img = np.zeros((300,300,4), dtype=np.uint8) 
    white_img = np.ones((300,300,4), dtype=np.uint8) * 255
    list2d = [[blank_img, white_img], [white_img, blank_img]]
    result_img = concat_vh(list2d, (300,300))
    cv2.imshow("Captured Image", result_img )

    while True:
        ### Check if the playback is EOF ###
        if MODE == "Playback":
            if not device.isOpened():
                break

        ##### Get camera capture ####
        capture = device.update()

        #### Get Color Depth image ####
        ret1, depth_color_image = capture.get_colored_depth_image()

        if PCD_VIEW == "RGB":
            #### Get Rgb Image ####
            ret2, rgb_image = capture.get_color_image()  # (720,1280)

            #### Get Transformed Depth to Color Camera ###
            ret3, depth_image = capture.get_transformed_depth_image()  # (720,1280)

        elif PCD_VIEW == "Depth":
            ret2, rgb_image = capture.get_transformed_color_image()

            ret3, depth_image = capture.get_depth_image()

        #### Check for ret ####
        if not ret1:
            continue
        if not ret2:
            continue
        if not ret3:
            continue

        if IS_VERTICAL_CAMERA:
            depth_color_image = cv2.rotate(
                depth_color_image, ROTATE_90_COUNTERCLOCKWISE)
            rgb_image = cv2.rotate(rgb_image, ROTATE_90_COUNTERCLOCKWISE)
            depth_image = cv2.rotate(
                depth_image, ROTATE_90_COUNTERCLOCKWISE)

        modified_rgb_image = detector.findPose(
            rgb_image, drawSegmen=True, drawSkel=IS_DRAW_SKELETON, Mode=MODE)
        skel_position, segmentation_mask = detector.findPosition(rgb_image)

        if key_pose_mode == 1:
            if np.shape(segmentation_mask) != (0,):
                angle1 = detector.findAngle(
                    modified_rgb_image, 13, 11, 23, draw=IS_DRAW_EFFECT)
                angle2 = detector.findAngle(
                    modified_rgb_image, 14, 12, 24, draw=IS_DRAW_EFFECT)
            else:
                angle1 = 0
                angle2 = 0
            if angle1 >= 80 and angle2 >= 80:
                if not start_clock:
                    skel_start_time = time.time()
                    start_clock = True
                    angle1_ex, angle2_ex = angle1, angle2
                elif start_clock and time.time() - skel_start_time > 1:
                    start_clock = False
                    key_poses_list[0] = [
                        depth_image, modified_rgb_image, skel_position, segmentation_mask]
                    
                    modified_rgb_image1 = cv2.resize(modified_rgb_image,(300,300))
                    list2d = [[modified_rgb_image1, white_img], [white_img, blank_img]]
                    result_img = concat_vh(list2d, (300,300))
                    cv2.imshow("Captured Image", result_img )
                    winsound.Beep(Frequency,Duration_Beep)
                    length_max = detector.findDistance(
                        modified_rgb_image, 11, 12, draw=IS_DRAW_EFFECT)
                    # print(length_max)

                    key_pose_mode += 1
                elif abs(angle2_ex - angle2) > 5 or abs(angle1_ex - angle1) > 5:
                    start_clock = False
                angle1_ex, angle2_ex = angle1, angle2
            else:
                start_clock = False

        elif key_pose_mode == 2:
            if np.shape(segmentation_mask) != (0,):
                ### Check for human body width ###
                length = detector.findDistance(
                    modified_rgb_image, 11, 12, draw=IS_DRAW_EFFECT)
                if length / length_max < 0.5:
                    if not start_clock:
                        side_start_time = time.time()
                        start_clock = True
                        length_ex = length
                    elif start_clock and time.time() - side_start_time > 1:
                        start_clock = False
                        key_poses_list[1] = [
                            depth_image, modified_rgb_image, skel_position, segmentation_mask]
                        modified_rgb_image2 = cv2.resize(modified_rgb_image,(300,300))
                        list2d = [[modified_rgb_image1, modified_rgb_image2], [white_img, blank_img]]
                        result_img = concat_vh(list2d, (300,300))
                        cv2.imshow("Captured Image", result_img )
                        winsound.Beep(Frequency,Duration_Beep)
                        
                        key_pose_mode += 1

                    elif abs(length_ex - length) / 100 > 0.1:
                        start_clock = False
                    length_ex = length

        elif key_pose_mode == 3:
            if np.shape(segmentation_mask) != (0,):
                angle1 = detector.findAngle(
                    modified_rgb_image, 13, 11, 23, draw=IS_DRAW_EFFECT)
                angle2 = detector.findAngle(
                    modified_rgb_image, 14, 12, 24, draw=IS_DRAW_EFFECT)
            else:
                angle1 = 0
                angle2 = 0
            if angle1 >= 80 and angle2 >= 80:
                if not start_clock:
                    skel_start_time = time.time()
                    start_clock = True
                    angle1_ex, angle2_ex = angle1, angle2
                elif start_clock and time.time() - skel_start_time > 1:
                    start_clock = False
                    key_poses_list[2] = [
                        depth_image, modified_rgb_image, skel_position, segmentation_mask]
                    modified_rgb_image3 = cv2.resize(modified_rgb_image,(300,300))
                    list2d = [[modified_rgb_image1, modified_rgb_image2], [modified_rgb_image3, blank_img]]
                    result_img = concat_vh(list2d, (300,300))
                    cv2.imshow("Captured Image", result_img )
                    winsound.Beep(Frequency,Duration_Beep)
                    key_pose_mode += 1
                elif abs(angle2_ex - angle2) > 5 or abs(angle1_ex - angle1) > 5:
                    start_clock = False
                angle1_ex, angle2_ex = angle1, angle2
            else:
                start_clock = False

        elif key_pose_mode == 4:
            if np.shape(segmentation_mask) != (0,):
                length = detector.findDistance(
                    modified_rgb_image, 11, 12, draw=IS_DRAW_EFFECT)
                if length / length_max < 0.5:
                    if not start_clock:
                        side_start_time = time.time()
                        start_clock = True
                        length_ex = length
                    elif start_clock and time.time() - side_start_time > 1:
                        start_clock = False
                        key_poses_list[3] = [
                            depth_image, modified_rgb_image, skel_position, segmentation_mask]
                        modified_rgb_image4 = cv2.resize(modified_rgb_image,(300,300))
                        list2d = [[modified_rgb_image1, modified_rgb_image2], [modified_rgb_image3, modified_rgb_image4]]
                        result_img = concat_vh(list2d, (300,300))
                        cv2.imshow("Captured Image", result_img )
                        winsound.Beep(Frequency,Duration_Beep*6)
                        
                        key_pose_mode += 1

                    elif abs(length_ex - length) / 100 > 0.1:
                        start_clock = False
                    length_ex = length

        img_tile = concat_vh(
            [[modified_rgb_image[:, :, :3], depth_color_image[:, :, :3]
              ]], (480, 480))

        # # # # ##### Plot the image ####
        cv2.imshow('Image', img_tile)

        ##### Press q key to stop ####
        if (cv2.waitKey(1) & 0xFF == ord('q')) :
            sys.exit()
        if key_pose_mode == 5 :
            break

    device.close()
    cv2.destroyAllWindows()

    for id, element in enumerate(key_poses_list):
        if element != 0:
            rgb_image = element[1]
            depth_image = element[0]

            filename_rgb = join(dataset_path, EXPORT_RGB_FOLDER) + \
                           "/rgb_data" + str(id) + ".png"
            cv2.imwrite(filename_rgb, rgb_image)

            filename_depth = join(dataset_path, EXPORT_DEPTH_FOLDER) + \
                             "/depth_data" + str(id) + ".png"
            cv2.imwrite(filename_depth, depth_image)
    np.save(join(dataset_path, "dataset.npy"), key_poses_list)


def point_cloud_processing(key_poses_list, device_calibration):
    ### Get one capture for depth and width info ###
    depth_image_front = key_poses_list[0][0]
    depth_width = np.shape(depth_image_front)[1]
    depth_height = np.shape(depth_image_front)[0]

    ### Create uv table ###
    if isfile(join(folder_path, "uv_table.npz")):
        data = np.load(join(folder_path, "uv_table.npz"))
        x_table_data, y_table_data = data['name1'], data['name2']
        if np.shape(x_table_data)[0] != depth_height * depth_width:
            x_table_data, y_table_data = create_xy_table(
                device_calibration, depth_height, depth_width)
            np.savez(join(folder_path, "uv_table"),
                     name1=x_table_data, name2=y_table_data)
    else:
        x_table_data, y_table_data = create_xy_table(
            device_calibration, depth_height, depth_width)
        np.savez(join(folder_path, "uv_table"),
                 name1=x_table_data, name2=y_table_data)

    make_clean_folder(join(dataset_path, EXPORT_POINTCLOUD_FOLDER))
    make_clean_folder(join(dataset_path, "skeleton_track"))

    jobs = []
    for i in range(len(key_poses_list)):
        if key_poses_list is not 0:
            p = Process(target=point_cloud_worker, args=(
                key_poses_list[i], depth_width, depth_height, x_table_data, y_table_data, i + 1))
            jobs.append(p)
            p.start()
    for proc in jobs:
        proc.join()


def pcd_human_reconstruction(pcd_List, skel_pcd_List):
    make_clean_folder(join(dataset_path, EXPORT_FULL_BODY_POINTCLOUD_FOLDER))

    ### Create axis coordinate visualizer ###
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=300, origin=[0, 0, 0])

    ### Get all pointcloud pose ###
    front_Pose = pcd_Pose(pcd_List[0], skel_pcd_List[0])
    back_Pose = pcd_Pose(pcd_List[2], skel_pcd_List[2])
    side1_Pose = pcd_Pose(pcd_List[1], skel_pcd_List[1])
    side2_Pose = pcd_Pose(pcd_List[3], skel_pcd_List[3])

    ### Rotate Back Pose by 180 ###
    back_Pose.transform_pcd(get_rotation_matrix(y_angle=180))

    ### Decompose Front Human Part ###
    front_Pose.get_human_part_decomposition()

    ### Decompose Back Human Part ###
    back_Pose.get_human_part_decomposition()

    ### 3D Global Registration ###
    torso_result = fast_global_registration(
        front_Pose.pcd_torso_part, back_Pose.pcd_torso_part, scale=1, is_projected=False)

    ### Transform 3D Point cloud according to result ###
    back_Pose.transform_pcd(torso_result.transformation)

    hip_y_avg, hip_avg_width, hip_avg_thick = get_hip_dimension(front_Pose.pcd_points_np, back_Pose.pcd_points_np,
                                                                side1_Pose.pcd_points_np, side2_Pose.pcd_points_np,
                                                                front_Pose.pcd_skel_np)

    ### Merge Back Point Cloud to Front ###
    full_Pose = pcd_Pose.merge_front_and_back_pcd(front_Pose, back_Pose, hip_avg_thick, hip_y_avg)

    full_Pose.get_human_part_decomposition()

    ### Decompose Side Human Part into torso-only(exclude arm) part ###
    side1_Pose.get_human_part_decomposition(mode=1)
    side2_Pose.get_human_part_decomposition(mode=1)

    side1_Pose.transform_pcd(get_rotation_matrix(y_angle=90))
    side2_Pose.transform_pcd(get_rotation_matrix(y_angle=-90))

    full_Pose = pcd_Pose.merge_side_one_and_two_pcd(full_Pose, side1_Pose, side2_Pose, hip_avg_width, hip_y_avg)

    ### Save Reconstructed PointCloud ###
    filename = join(
        dataset_path, EXPORT_FULL_BODY_POINTCLOUD_FOLDER) + "/0_data.ply"
    o3d.io.write_point_cloud(filename, full_Pose.pcd, write_ascii=True)

    filename = join(
        dataset_path, EXPORT_FULL_BODY_POINTCLOUD_FOLDER) + "/1_skel_data.ply"
    o3d.io.write_point_cloud(filename, full_Pose.pcd_skel, write_ascii=True)

    if DEBUG_PLOT:
        ### Initialize vis with callback func ###
        vis = init_pointcloud_visualizer()
        # full_Pose.pcd.paint_uniform_color([1, 0.706, 0])

        ### Add Geometry ###
        vis.add_geometry(full_Pose.pcd)

        vis.add_geometry(mesh_frame)

        vis.run()
        vis.destroy_window()


def poisson_surface_reconstruction(pcd):
    pcd_org = copy.deepcopy(pcd)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, scale=2, linear_fit=True)

    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    # print(densities.min(), densities.max()) # min 6.0 max 9.66
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)

    vertices_to_remove = densities < 7
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # mesh.paint_uniform_color([1, 0.706, 0])
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)

    pcd = mesh.sample_points_uniformly(number_of_points=30000)
    # pcd = mesh.sample_points_poisson_disk(number_of_points=15000, pcl=pcd) 

    filename = join(
        dataset_path, EXPORT_FULL_BODY_POINTCLOUD_FOLDER) + "/2_modified_data.ply"
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

    pcd.transform(get_translation_matrix(Tx=1500))
    pcd_org.transform(get_translation_matrix(Tx=-1500))

    if DEBUG_PLOT:
        vis = init_pointcloud_visualizer()
        vis.add_geometry(pcd)
        vis.add_geometry(mesh)
        vis.add_geometry(pcd_org)
        vis.run()
        vis.destroy_window()


def gui_result(waist, hip, height):
    img = cv2.imread("GUI_NEW.png")

    cv2.imshow("Measurement Result", img)
    cv2.putText(img, "{:.1f}".format(waist)+" in", (500, 418), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), thickness=1) 
    cv2.putText(img, "{:.1f}".format(hip)+" in", (500, 478), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), thickness=1) 
    cv2.putText(img, "{:.1f}".format(height)+" cm", (485, 548), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), thickness=1) 

    cv2.imshow("Measurement Result", img)
    cv2.waitKey(0)


def measure_human_part(full_body_pcd):
    full_body_Pose = pcd_Pose(full_body_pcd[2], full_body_pcd[1]) 

    full_body_Pose.get_human_part_decomposition()

    hip_perimeter = hip_measurement(full_body_Pose.pcd_points_np, full_body_Pose.pcd_skel_np, range=10)
    waist_perimeter = waist_measurement(full_body_Pose.pcd_points_np, full_body_Pose.pcd_skel_np, range=10)

    head_length= measure_head_part(np.asarray(full_body_Pose.pcd_head_part.points), full_body_Pose.pcd_skel_np, range=25)
    height = measure_human_height(full_body_pcd[2], full_body_pcd[1], head_length)

    vis = init_pointcloud_visualizer()
    vis.add_geometry(full_body_Pose.pcd)
    vis.run()
    vis.destroy_window()
    
    if GUI: gui_result(waist_perimeter, hip_perimeter, height)


def vis_text(text):
    img = np.zeros((100,300,3), dtype=np.uint8)
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), thickness=1) 

    cv2.imshow("Display",img)
    cv2.waitKey(2000)

    img = np.zeros((100,300,3), dtype=np.uint8)
    cv2.imshow("Display",img)

    cv2.waitKey(2000)

 
if __name__ == "__main__":

    k4a_device, device_calibration = init_k4a()

    if DIV < 1 & MODE == "Device":
        vis_text("Process Start")
        get_key_poses_extraction(k4a_device)
        os.system('cls')
        vis_text("Capture Complete")
        
    if DIV < 2:
        key_poses_list = np.load(
            join(dataset_path, "dataset.npy"), allow_pickle=True)

        point_cloud_processing(key_poses_list, device_calibration)
        os.system('cls')
        vis_text("PCD Complete")

    if DIV < 3:
        pcd_List = load_pcd_folder_to_list("pointcloud_data")
        skel_pcd_List = load_pcd_folder_to_list("skeleton_track")

        pcd_human_reconstruction(pcd_List, skel_pcd_List)
        os.system('cls')
        print
        vis_text("Rcnstrt Complete")

    if DIV < 4:
        full_body_pcd = load_pcd_folder_to_list("full_body_pcd")
        poisson_surface_reconstruction(full_body_pcd[0])
        os.system('cls')
        vis_text("Poisson Complete")

    if DIV < 5:
        full_body_pcd = load_pcd_folder_to_list("full_body_pcd")
        measure_human_part(full_body_pcd)
        os.system('cls')
        vis_text("Process Complete")
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
