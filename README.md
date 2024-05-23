# camera_pose_estimation_lib


C++ CLI application implementing pipeline for the task of camera pose estimation in 3D scene using an image from camera.

Task: given a set of images used for 3D scene registration and camera pose for each single image we need to estimate the pose of the camera where a query image was captured. 

In this project we performed research on various techniques on image and point cloud matching was performed. 
Final solution: matching query image against all the original images used for 3D scene reconstruction. As keypoint detector and feature descriptor SIFT algorithm is used.

## Using package

Stack:
- OpenCV
- Eigen
- FLANN
- Conan

Install:

```
conan install .
conan build .
```

## Run tests

Move to folder bin and run tests:

```
cd bin && ./library_test
```
