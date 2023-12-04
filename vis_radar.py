import cv2 
import numpy as np
from tqdm import tqdm
from lib.utils_bbox import *

from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train, obj_test
from pyboreas.utils.utils import get_T_bev_metric

bd = BoreasDataset(root='/data_1TB_2/boreas/', split=None, verbose=False, labelFolder="labels_detection")

if bd.split is None:
  num_split = 1
else:
  num_split = len(bd.split)

# ==================================================================================================================

resolution = 0.25
width = int(200 / resolution)
T_bev_metric = get_T_bev_metric(resolution, width)

# ==================================================================================================================

for seq_i in range(num_split):

  seq = bd.sequences[seq_i]
  for i in tqdm(range(0, len(seq.radar_frames))):

    # loading radar
    radar_frame = seq.get_radar(i)
    radar_img = radar_frame.polar_to_cart(resolution, width)
    radar_img = (radar_img * 255.0).astype(np.uint8)
    radar_img = cv2.addWeighted(radar_img, 1, radar_img, 1, 0)  # use to enhance the brightness
    radar_img = cv2.cvtColor(radar_img, cv2.COLOR_GRAY2RGB)

    # loading label
    boxes = radar_frame.get_bounding_boxes(seq.labelFiles, seq.labelTimes, seq.labelPoses)
    if boxes is not None:
      boxes.transform(T_bev_metric)
      for box in boxes.bbs:
        draw_box_2d(radar_img, box.pc.points.T.astype(int), box.label)

    cv2.namedWindow('radar_img', 0)
    cv2.imshow('radar_img', radar_img)
    cv2.waitKey(1)
#    cv2.imwrite('outputs/radar/' + radar_frame.frame + '.jpg', radar_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    radar_frame.unload_data()
