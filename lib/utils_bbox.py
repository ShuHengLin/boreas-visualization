import cv2
import numpy as np

# ==================================================================================================================

def color_select(cls):

    if cls == 'Car':
      color = (0, 255, 0)     # Green

    elif cls == 'Pedestrian':
      color = (0, 0, 255)     # Red

    elif cls == 'Cyclist':
      color = (0, 255, 255)   # Yellow

    elif cls == 'Truck':
      color = (255, 255, 0)   # Cyan

    elif cls == 'Van':
      color = (255, 0, 255)   # Purple

    else:
      color = (255, 255, 255) # White

    return color

# ==================================================================================================================

def is_invalid_bbox(x1, y1, x2, y2, width=2448, height=2048):
  across_center_line = (x1 > 1224 and x2 < 1224) or (x1 < 1224 and x2 > 1224)
  out_of_img = (x1 < 0) or (x1 > width)  or (x2 < 0) or (x2 > width) or \
               (y1 < 0) or (y1 > height) or (y2 < 0) or (y2 > height) 
  return (across_center_line and out_of_img)

face_idx = [[0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7]]

def draw_box_3d(image, corners, cls):

  if corners is None:
    return image
  
  c = color_select(cls)

  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      x1, y1 = int(corners[f[j],       0]), int(corners[f[j],       1])
      x2, y2 = int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])
      if is_invalid_bbox(x1, y1, x2, y2):
        return image
      cv2.line(image, (x1, y1), (x2, y2), c, 2, lineType=cv2.LINE_AA)

    if ind_f == 3:
      cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1]) ),
                      (int(corners[f[2], 0]), int(corners[f[2], 1]) ), c, 1, lineType=cv2.LINE_AA)
      cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1]) ),
                      (int(corners[f[3], 0]), int(corners[f[3], 1]) ), c, 1, lineType=cv2.LINE_AA)
  return image

# ==================================================================================================================

def draw_box_2d(image, corners, cls):

  c = color_select(cls)
  cv2.line(image, (corners[0][0], corners[1][0]), (corners[0][1], corners[1][1]), c, 1, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][1], corners[1][1]), (corners[0][2], corners[1][2]), c, 1, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][2], corners[1][2]), (corners[0][3], corners[1][3]), c, 1, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][3], corners[1][3]), (corners[0][0], corners[1][0]), c, 1, lineType=cv2.LINE_AA)

  return image

