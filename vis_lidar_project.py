import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train
from pyboreas.utils.utils import get_inverse_tf

bd = BoreasDataset(root='/data_1TB_2/boreas/', split=None, verbose=False, labelFolder="labels_detection")

if bd.split is None:
  num_split = 1
else:
  num_split = len(bd.split)

fig = plt.figure(figsize=(24.48, 20.48), dpi=50)
ax = fig.add_subplot()

# ==================================================================================================================

for seq_i in range(num_split):

  seq = bd.sequences[seq_i]
  seq.filter_frames_gt()
  seq.synchronize_frames('lidar')

  for i in tqdm(range(0, len(seq.lidar_frames))):

    # loading image
    camera_frame = seq.get_camera(i)
    img = camera_frame.img

    # Remove motion distortion from pointcloud:
    lidar_frame  = seq.get_lidar(i)
    lidar_frame.remove_motion(lidar_frame.body_rate)

    # Get the transform from lidar to camera:
    T_enu_camera = camera_frame.pose
    T_enu_lidar  = lidar_frame.pose
    T_camera_lidar = np.matmul(get_inverse_tf(T_enu_camera), T_enu_lidar)
    lidar_frame.transform(T_camera_lidar)

    # Remove points outside our region of interest
    lidar_frame.passthrough([-75, 75, -20, 10, 0, 40])  # xmin, xmax, ymin, ymax, zmin, zmax
    
    # Project lidar points onto the camera image, using the projection matrix, P0.
    uv, colors, _ = lidar_frame.project_onto_image(seq.calib.P0)

    # Draw the projection
    ax.clear()
    ax.imshow(img)
    ax.set_xlim(0, 2448)
    ax.set_ylim(2048, 0)
    ax.scatter(uv[:, 0], uv[:, 1], c=colors, marker=',', s=10, edgecolors='none', alpha=0.7, cmap='jet')
    ax.set_axis_off()
#    fig.savefig('outputs/lidar_project/' + camera_frame.frame + '.jpg', dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
    plt.pause(0.01)

    camera_frame.unload_data()
    lidar_frame.unload_data()

