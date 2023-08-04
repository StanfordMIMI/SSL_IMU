
# Self-Supervised Learning for IMU-Driven Biomechanical Estimation
By Tian (Alan) Tan, Peter Shull, and Akshay Chaudhari

## Exclusive Summary
This repository includes the code and models for an abstract presented at ASB2023.
Full-length preprint is coming soon.

[//]: # (An [example implementation]&#40;#running-example-code&#41; is provided.)
[//]: # (When implementing our models, please place the IMUs according to [Hardware]&#40;#hardware&#41;)
[//]: # (and store the data according to [Data Format]&#40;#data-format&#41;.)

## How to Use
### Environment
Python 3.8; Pytorch 1.7.0; Cuda 11.0; Cudnn 8.0.4; matplotlib 3.3.2; numpy 1.19.4; h5py 3.0.0; Scikit-learn 0.23.2

Versions different from ours may still work.

## Pre-trained Tranformers

[//]: # (&#40;1&#41; [a fusion model based on eight IMUs and cameras]&#40;./trained_models_and_example_data/8IMU_camera.pth&#41;)

[//]: # (### Example code)

#### IMUs
One to eight IMUs should be placed on the body as shown in the figure below.
When using less than eight IMUs, specify the IMU indices in the `--imu_idx` argument.
Each IMU's z-axis is aligned with the body segment surface normal, y-axis points upwards,
and x-axis being perpendicular to the y and z axes following the right-hand rule.

<img src="figures/readme_fig/imu_position_and_orientation.png" width="200">

## Links for SSL datasets
[AMASS](https://amass.is.tue.mpg.de/download.php)

[MoVi](https://www.biomotionlab.ca/movi/)

## Links for downstream datasets
[Downstream dataset 1](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/)

[Downstream dataset 2](https://simtk.org/projects/imukinetics)

## References
