import rospy
import numpy as np

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# ==================================================================================================================

# clearing all markers / view in RVIZ remotely
#https://answers.ros.org/question/53595/clearing-all-markers-view-in-rviz-remotely/

def new_marker_array():
  marker_array_msg = MarkerArray()
  marker = Marker()
  marker.id = 0
  marker.action = Marker.DELETEALL
  marker_array_msg.markers.append(marker)
  return marker_array_msg

# ==================================================================================================================

def color_select(cls, marker):

    if cls == 'Car':
      marker.color.r = 0      # Green
      marker.color.g = 1
      marker.color.b = 0

    elif cls == 'Pedestrian':
      marker.color.r = 1      # Red
      marker.color.g = 0
      marker.color.b = 0

    elif cls == 'Cyclist':
      marker.color.r = 1      # Yellow
      marker.color.g = 1
      marker.color.b = 0

    elif cls == 'Truck':
      marker.color.r = 0      # Cyan
      marker.color.g = 1
      marker.color.b = 1

    elif cls == 'Van':
      marker.color.r = 1      # Purple
      marker.color.g = 0
      marker.color.b = 1

    else:
      marker.color.r = 1      # White
      marker.color.g = 1
      marker.color.b = 1

    return marker

# ==================================================================================================================

lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

def box_to_marker(ob, cls, index):

  detect_points_set = []
  for x in range(8):
    detect_points_set.append(Point(ob[x][0], ob[x][1], ob[x][2]))

  marker = Marker()
  marker.header.frame_id = 'map'
  marker.header.stamp = rospy.Time.now()
  marker.id = index
  marker.action = Marker.ADD
  marker.type = Marker.LINE_LIST
  marker.lifetime = rospy.Duration(0)

  marker = color_select(cls, marker)
  marker.color.a = 1
  marker.scale.x = 0.2
  marker.points = []

  for line in lines:
    marker.points.append(detect_points_set[line[0]])
    marker.points.append(detect_points_set[line[1]])

  return marker

