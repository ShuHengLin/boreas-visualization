# boreas-visualization

## Prepare Data
1) Follow the instructions below to download the dataset, refer to the [download instructions](https://github.com/utiasASRL/pyboreas/blob/master/download.md).
   * Install the AWS CLI：
     ```
     curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
     unzip awscliv2.zip
     sudo ./aws/install
     rm -rf aws
     rm awscliv2.zip
     ```
   * Download the Boreas-Objects-V1 dataset：
     ```
     aws s3 sync s3://boreas/boreas-objects-v1 /data_1TB_2/boreas/boreas-objects-v1 --no-sign-reques
     ```

## Video
[![](https://img.youtube.com/vi/fXD2hjp8eNg/0.jpg)](https://youtu.be/fXD2hjp8eNg)

## References
1) [Boreas: A Multi-Season Autonomous Driving Dataset](https://arxiv.org/abs/2203.10168)
