import cv2
import numpy as np
from tqdm import tqdm
from lib.utils_bbox import *

from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train, obj_test

bd = BoreasDataset(root='/data_1TB_2/boreas/', split=None, verbose=False, labelFolder="labels_detection")

if bd.split is None:
  num_split = 1
else:
  num_split = len(bd.split)

# ==================================================================================================================

for seq_i in range(num_split):

  seq = bd.sequences[seq_i]
  for i in tqdm(range(0, len(seq.camera_frames))):

    # loading image
    camera_frame = seq.get_camera(i)
    img = camera_frame.img
    img = jpg_compress(img, 90)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # loading label
    boxes = camera_frame.get_bounding_boxes(seq.labelFiles, seq.labelTimes, seq.labelPoses)
    if boxes is not None:
      for box in boxes.bbs:
        if not box.pos[2] < 0:
          uv = box.project(seq.calib.P0, checkdims=False)
          img = draw_box_3d(img, uv, box.label)

    cv2.namedWindow('img', 0)
    cv2.imshow('img', img)
    cv2.waitKey(1)
#    cv2.imwrite('outputs/camera/' + camera_frame.frame + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 10])
    camera_frame.unload_data()
