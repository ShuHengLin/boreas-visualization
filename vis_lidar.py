import rospy
import numpy as np
from tqdm import tqdm
from lib.utils_pointcloud import new_marker_array, box_to_marker
from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train

import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

bd = BoreasDataset(root='/data_1TB_2/boreas/', split=None, verbose=False, labelFolder="labels_detection")

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

    # loading label
    marker_array = new_marker_array()
    boxes = lidar_frame.get_bounding_boxes()
    for j, box in enumerate(boxes.bbs):
      marker = box_to_marker(box.pc.points, cls=box.label, index=j)
      marker_array.markers.append(marker)
    marker_pub.publish(marker_array)

    lidar_frame.unload_data()
    rate.sleep()

