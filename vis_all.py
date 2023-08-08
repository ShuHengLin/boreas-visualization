import cv2
import rospy
import numpy as np
from tqdm import tqdm
from lib.utils_bbox import draw_box_3d, draw_box_2d
from lib.utils_pointcloud import new_marker_array, box_to_marker
from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train
from pyboreas.utils.utils import get_T_bev_metric

import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

bd = BoreasDataset(root='/data_1TB_2/boreas/', split=None, verbose=False, labelFolder="labels_detection")

if bd.split is None:
  num_split = 1
else:
  num_split = len(bd.split)

resolution = 0.25
width = int(150 / resolution)
T_bev_metric = get_T_bev_metric(resolution, width)

# ==================================================================================================================

pointcloud_pub = rospy.Publisher('/pointcloud',   PointCloud2, queue_size=10)
marker_pub     = rospy.Publisher('/detect_box3d', MarkerArray, queue_size=10)
rospy.init_node('talker', anonymous=True)

for seq_i in range(num_split):

  if rospy.is_shutdown():
    break

  seq = bd.sequences[seq_i]
  seq.filter_frames_gt()
  seq.synchronize_frames('lidar')

  for i in tqdm(range(0, len(seq.lidar_frames))):

    if rospy.is_shutdown():
      break

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'

    # loading pointcloud
    lidar_frame = seq.get_lidar(i)
    pointcloud_msg = pcl2.create_cloud_xyz32(header, lidar_frame.points[:, 0:3])

    # loading lidar label
    marker_array = new_marker_array()
    boxes = lidar_frame.get_bounding_boxes()
    for j, box in enumerate(boxes.bbs):
      marker = box_to_marker(box.pc.points, cls=box.label, index=j)
      marker_array.markers.append(marker)


    # loading image
    camera_frame = seq.get_camera(i)
    img = camera_frame.img

    # loading camera label
    boxes = camera_frame.get_bounding_boxes(seq.labelFiles, seq.labelTimes, seq.labelPoses)
    for box in boxes.bbs:
      if not box.pos[2] < 0:
        uv = box.project(seq.calib.P0, checkdims=False)
        img = draw_box_3d(img, uv, box.label)


    # loading radar
    radar_frame = seq.get_radar(i)
    radar_img = radar_frame.polar_to_cart(resolution, width)
    radar_img = cv2.addWeighted(radar_img, 1, radar_img, 1, 0)  # use to enhance the brightness
    radar_img = cv2.cvtColor(radar_img, cv2.COLOR_GRAY2RGB)
    radar_img *= 255

    # loading radar label
    boxes = radar_frame.get_bounding_boxes(seq.labelFiles, seq.labelTimes, seq.labelPoses)
    bounds = [-75, 75, -75, 75, -5, 10] # xmin, xmax, ymin, ymax, zmin, zmax
    boxes.passthrough(bounds)
    boxes.transform(T_bev_metric)
    for box in boxes.bbs:
      draw_box_2d(radar_img, box.pc.points.T.astype(int), box.label)


    # visualize
    pointcloud_pub.publish(pointcloud_msg)
    marker_pub.publish(marker_array)

    img = cv2.resize(img, (1434, 1200))
    radar_img = cv2.resize(radar_img, (1200, 1200))
    img = cv2.copyMakeBorder(img.copy(), 0, 0, 0, 1200, cv2.BORDER_CONSTANT)
    img[0:1200, 1434:2634] = radar_img

    cv2.namedWindow('img', 0)
    cv2.resizeWindow('img', 1800, 800)
    cv2.imshow('img', img)
    cv2.waitKey(500)
    lidar_frame.unload_data()
    camera_frame.unload_data()
    radar_frame.unload_data()

