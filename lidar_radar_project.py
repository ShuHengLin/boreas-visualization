import cv2
import numpy as np
from tqdm import tqdm
from lib.utils_bbox import *
from lib.utils_pointcloud import *

from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train, obj_test
from pyboreas.utils.utils import get_inverse_tf, yawPitchRollToRot

bd = BoreasDataset(root='/data_1TB_2/boreas/', split=None, verbose=False, labelFolder="labels_detection")

if bd.split is None:
  num_split = 1
else:
  num_split = len(bd.split)

# ==================================================================================================================

resolution = 0.1
width = 1400

# ==================================================================================================================

Rot = np.matrix(np.zeros((4, 4)))
Rot[:3, :3] = yawPitchRollToRot(np.radians(0), np.radians(180), np.radians(180))
Rot[3, 3] = 1

def crop_radar(img):

  centre = (width // 2, width // 2)
  x, y = np.meshgrid(np.arange(width) - centre[0], np.arange(width) - centre[1])
  angle = np.degrees(np.arctan2(y, x))
  radar_img_crop = np.where((angle >= -130.5) & (angle <= -49.5), radar_img, 0)

  return radar_img_crop

# ==================================================================================================================

for seq_i in range(num_split):

  seq = bd.sequences[seq_i]
  seq.synchronize_frames('radar')
  for i in tqdm(range(0, len(seq.radar_frames))):

    # loading radar
    radar_frame = seq.get_radar(i)
    radar_img = radar_frame.polar_to_cart(resolution, width)
    radar_img = crop_radar(radar_img)
    radar_img = (radar_img * 255.0).astype(np.uint8)
    radar_img = cv2.addWeighted(radar_img, 1, radar_img, 1, 0)  # use to enhance the brightness
    radar_img = cv2.cvtColor(radar_img, cv2.COLOR_GRAY2RGB)

    # loading Camera
    camera_frame = seq.get_camera(i)
    img = camera_frame.img
    img = jpg_compress(img, 90)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # loading pointcloud & Remove motion distortion from pointcloud
    lidar_frame = seq.get_lidar(i)
    lidar_frame.remove_motion(lidar_frame.body_rate)
    pts = np.copy(lidar_frame.points[:, :5])

    # Filter by image
    image_filter = get_image_filter(camera_frame, lidar_frame, seq.calib.P0)
    pts = pts[image_filter]


    # Get the transform from lidar to radar
    T_enu_radar = radar_frame.pose
    T_enu_lidar = lidar_frame.pose
    T_radar_lidar = np.matmul(get_inverse_tf(T_enu_radar), T_enu_lidar)

    # Voxelize and create LiDAR image
    pointcloud = Pointcloud(pts)
    pointcloud.transform(T_radar_lidar)
    pointcloud.transform(Rot)
    voxel = pointcloud.voxelize((resolution, resolution, resolution),
                                extents=np.array([[-70.0, 70.0],
                                                  [-70.0, 70.0],
                                                  [-2.0,   5.0],]),
                                return_indices=False,
                               )
    lidar_img = np.zeros((width, width, 3)).astype(np.uint8)
    lidar_img[:, :, 2] = np.sum(voxel, axis=2) * 255
    lidar_img = cv2.flip(lidar_img, 1)


    # Output
    output = cv2.addWeighted(lidar_img, 0.4, radar_img, 1.0, 0)
    output = output[0:700, 300:1100]
    cv2.namedWindow ('camera', 0)
    cv2.resizeWindow('camera', [img.shape[1] // 3, img.shape[0] // 3])
    cv2.moveWindow  ('camera', 1200, 0)

    cv2.namedWindow ('output', 0)
    cv2.resizeWindow('output', [900, 900])
    cv2.moveWindow  ('output', 0, 0)

    cv2.imshow('output', output)
    cv2.imshow('camera', img)
    cv2.waitKey(1)

    radar_frame.unload_data()
    lidar_frame.unload_data()
    camera_frame.unload_data()
