import cv2
import numpy as np
from tqdm import tqdm
from lib.utils_bbox import draw_box_3d
from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train

bd = BoreasDataset(root='/data_1TB_2/boreas/', split=None, verbose=False, labelFolder="labels_detection")

if bd.split is None:
  num_split = 1
else:
  num_split = len(bd.split)

# ==================================================================================================================

for seq_i in range(num_split):

  seq = bd.sequences[seq_i]
  seq.filter_frames_gt()
  seq.synchronize_frames('lidar')

  for i in tqdm(range(0, len(seq.lidar_frames))):

    # loading image
    camera_frame = seq.get_camera(i)
    img = camera_frame.img

    # loading label
    boxes = camera_frame.get_bounding_boxes(seq.labelFiles, seq.labelTimes, seq.labelPoses)
    for box in boxes.bbs:
      if not box.pos[2] < 0:
        uv = box.project(seq.calib.P0, checkdims=False)
        img = draw_box_3d(img, uv, box.label)

    cv2.namedWindow('img', 0)
    cv2.imshow('img', img)
    cv2.waitKey(1)
    camera_frame.unload_data()
