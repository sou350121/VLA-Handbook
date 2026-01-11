# 相机标定指南 (Camera Calibration Guide)

> **核心概念**: 机器人只知道自己的关节角度和末端坐标 (Base Frame)，而相机只知道像素点 (Camera Frame)。**外参标定 (Extrinsic Calibration)** 就是要求解这两个坐标系之间的变换矩阵 $T$。

这是一项**通用**的部署技能，无论是 Pi0、OpenVLA 还是传统的抓取任务，只要涉及手眼协调，都必须进行标定。

## 1. 为什么必须标定? (Why Calibrate?)
如果不标定，模型看到"苹果在图像右边"，但机器人不知道"图像右边"对应物理世界的哪里 (是向右 10cm 还是 20cm?)。
- **目标**: 获得 $T_{base}^{camera}$ (相机相对于基座的位姿)。

## 2. 两种标定模式 (Calibration Modes)

### A. Eye-to-Hand (眼在手外)
- **适用相机**: High Cam, Low Cam (固定在架子上)。
- **原理**: 相机不动，机械臂动。
- **求解目标**: $T_{base}^{camera}$ (固定值)。
- **操作**: 将 Aruco 码贴在机械臂末端，控制机械臂移动到不同位置，相机拍摄。

### B. Eye-in-Hand (眼在手上)
- **适用相机**: Wrist Cam (绑在手腕上)。
- **原理**: 相机随机械臂运动。
- **求解目标**: $T_{end\_effector}^{camera}$ (固定值，即相机相对于手腕的偏移量)。
- **操作**: 将 Aruco 码固定在桌子上不动，机械臂带着相机从不同角度拍摄 Aruco 码。

## 3. 标定实战流程 (Workflow)

推荐使用 ROS 的 `easy_handeye` 包或 OpenCV 手写脚本。

### 3.1 准备工作
1.  **打印标定板**: 打印一张 **Aruco Marker** 或 **AprilTag**。
    - *关键点*: 必须贴在硬纸板或亚克力板上，保证绝对平整。纸张弯曲会导致严重误差。
    - *尺寸*: 测量打印出来的实际尺寸 (如 100mm)，精确到毫米。

### 3.2 数据采集 (Data Collection)
移动机械臂到 N 个 (推荐 15-20 个) 不同的姿态 (Pose)。
- **多样性**: 姿态要覆盖工作空间的不同位置和角度 (位置要变，旋转也要变)。
- **记录**:
    - 机械臂末端位姿 (Robot Pose: $x, y, z, qx, qy, qz, qw$)。
    - 相机中检测到的标定板位姿 (Marker Pose)。

### 3.3 求解方程 (Solving)
这是一个经典的 $AX=XB$ 问题。
- 使用 `cv2.calibrateHandEye()` 求解。

```python
import cv2

# R_gripper2base, t_gripper2base: 机械臂末端相对于基座的旋转和平移 (从机器人控制器读取)
# R_target2cam, t_target2cam: 标定板相对于相机的旋转和平移 (从 Aruco 检测算法读取)

R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)
```

### 3.4 验证 (Verification)
1.  **重投影误差**: 计算标定结果的残差。
2.  **可视化验证 (最直观)**:
    - 将标定结果写入 URDF 或配置文件。
    - 在 Rviz 中显示 PointCloud。
    - 将一个已知物体 (如标定板) 放在桌上，看点云中的标定板是否与机器人的模型 (Mesh) 或真实位置完美重合。

---
[← Back to Deployment](./README.md)
