import cv2
import rospy
import numpy as np
from tqdm import tqdm
from lib.utils_bbox import *
from lib.utils_pointcloud import *

from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train, obj_test
from pyboreas.utils.utils import get_T_bev_metric

import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

bd = BoreasDataset(root='/data_1TB_2/boreas/', split=obj_train, verbose=False, labelFolder="labels_detection")

if bd.split is None:
  num_split = 1
else:
  num_split = len(bd.split)

# ==================================================================================================================

header = std_msgs.msg.Header()
header.frame_id = 'map'

fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
          PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
          PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
          PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)]

pointcloud_pub = rospy.Publisher('/pointcloud',   PointCloud2, queue_size=10)
marker_pub     = rospy.Publisher('/detect_box3d', MarkerArray, queue_size=10)
rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(1000)

# ==================================================================================================================

resolution = 0.25
width = int(200 / resolution)
T_bev_metric = get_T_bev_metric(resolution, width)

# ==================================================================================================================

for seq_i in range(num_split):

  if rospy.is_shutdown():
    break

  seq = bd.sequences[seq_i]
  seq.filter_frames_gt()
  seq.synchronize_frames('lidar')
  for i in tqdm(range(0, len(seq.lidar_frames))):

    if rospy.is_shutdown():
      break

    # loading pointcloud
    lidar_frame = seq.get_lidar(i)
    pointcloud_msg = pcl2.create_cloud(header, fields, lidar_frame.points[:, 0:4])
    pointcloud_pub.publish(pointcloud_msg)

    # loading lidar label
    marker_array = new_marker_array()
    boxes = lidar_frame.get_bounding_boxes()
    if boxes is not None:
      for j, box in enumerate(boxes.bbs):
        marker = box_to_marker(box.pc.points, cls=box.label, index=j)
        marker_array.markers.append(marker)
      marker_pub.publish(marker_array)


    # loading image
    camera_frame = seq.get_camera(i)
    img = camera_frame.img
    img = jpg_compress(img, 90)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # loading camera label
    boxes = camera_frame.get_bounding_boxes(seq.labelFiles, seq.labelTimes, seq.labelPoses)
    if boxes is not None:
      for box in boxes.bbs:
        if not box.pos[2] < 0:
          uv = box.project(seq.calib.P0, checkdims=False)
          img = draw_box_3d(img, uv, box.label)


    # loading radar
    radar_frame = seq.get_radar(i)
    radar_img = radar_frame.polar_to_cart(resolution, width)
    radar_img = (radar_img * 255.0).astype(np.uint8)
    radar_img = cv2.addWeighted(radar_img, 1, radar_img, 1, 0)  # use to enhance the brightness
    radar_img = cv2.cvtColor(radar_img, cv2.COLOR_GRAY2RGB)

    # loading radar label
    boxes = radar_frame.get_bounding_boxes(seq.labelFiles, seq.labelTimes, seq.labelPoses)
    if boxes is not None:
      boxes.transform(T_bev_metric)
      for box in boxes.bbs:
        draw_box_2d(radar_img, box.pc.points.T.astype(int), box.label)


    # visualize
    img = cv2.resize(img, (956, 800))
    radar_img = cv2.resize(radar_img, (800, 800))
    output = np.zeros((800, 1756, 3), dtype=np.uint8)
    output[:, :956, :] = img
    output[:, 956:, :] = radar_img

    cv2.namedWindow('img', 0)
    cv2.resizeWindow('img', 1800, 800)
    cv2.imshow('img', output)
    cv2.waitKey(1)

    lidar_frame.unload_data()
    camera_frame.unload_data()
    radar_frame.unload_data()
