# boreas-visualization

## Prepare Data
1) Follow the instructions below to download the dataset, refer to the [download instructions](https://github.com/utiasASRL/pyboreas/blob/master/download.md).
   * Install the AWS CLI：
     ```
     curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
     ```
     ```
     unzip awscliv2.zip
     ```
     ```
     sudo ./aws/install
     ```
     ```
     rm -rf aws && rm awscliv2.zip
     ```
   * Download the Boreas-Objects-V1 dataset：
     ```
     aws s3 sync s3://boreas/boreas-objects-v1 /data_1TB_2/boreas/boreas-objects-v1 --no-sign-reques
     ```
2) Clone and install the [pyboreas](https://github.com/utiasASRL/pyboreas) repository.


## Visualize lidar pointcloud
* Using rviz to visualize:
```
roscore
```
```
rosrun rviz rviz -d rviz_config.rviz
```
```
python -B vis_lidar.py
```


## Visualize camera image
```
python -B vis_camera.py
```


## Visualize radar image
```
python -B vis_radar.py
```


## Visualize all
* Using rviz to visualize:
```
roscore
```
```
rosrun rviz rviz -d rviz_config.rviz
```
```
python -B vis_all.py
```


## Video
[![](https://img.youtube.com/vi/fXD2hjp8eNg/0.jpg)](https://youtu.be/fXD2hjp8eNg)


## The Boreas label format
| uuid | type | dimensions | location | rotation_y | numPoints |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 070b7c14-6d3d-481b-a2de-b5d61d0fd4d7 | Car | 4.714 &thinsp; 2.123 &thinsp; 1.681 | -4.07860212693 &thinsp; -32.5713168804 &thinsp; -0.00723842866068 | 1.5368577939200003 | 208 |


## References
1) [Boreas: A Multi-Season Autonomous Driving Dataset](https://arxiv.org/abs/2203.10168)
