# 点云理解与 SLAM (Point Cloud Intelligence & SLAM)

> **面试场景**: “比较 Visual SLAM 与 LiDAR SLAM 的区别；点云网络有哪些？”

---

## 🌐 任务概览

```
┌──────────────────────────────────────────────────────────────────────────┐
│  输入: 3D 点云 (LiDAR / 深度相机 / SfM)                                  │
│  → 几何建图 → 语义理解 → 自定位 (SLAM) → 语义地图                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 点云表示与特征

| 表示 | 优点 | 缺点 | 代表网络 |
|:-----|:-----|:-----|:---------|
| 原始点集 | 准确，无丢失 | 不规则 | PointNet/PointNet++ |
| 体素 (Voxel) | 可用 3D CNN | 稀疏计算浪费 | VoxelNet, MinkowskiNet |
| 点柱 (Pillar) | 2D BEV 计算快 | 高度信息压缩 | PointPillars |
| Range Image | 结构化 | 投影失真 | RangeNet++ |
| BEV | 规划友好 | 高度丢失 | BEVFusion |

**特征网络**: PointNet++、KPConv、SparseConv、Point Transformer、Point-BERT。

---

## 2. 点云语义任务

### 2.1 检测 (3D Object Detection)

| 算法 | 输入 | 特点 |
|:-----|:-----|:-----|
| PointRCNN | 原始点 | Two-stage，细粒度 |
| PV-RCNN | Voxel + Point | 先体素再回点，精度高 |
| CenterPoint | BEV | Anchor-free，快 |

### 2.2 分割

- **RangeNet++**: Range image 语义分割
- **MinkowskiNet**: 稀疏卷积，语义+实例
- **PolarNet**: 极坐标下的 BEV 分割

### 2.3 场景流

- FlowNet3D、HPLFlowNet、BEVFlow

---

## 3. 点云配准与重建

| 方法 | 思路 | 适用 |
|:-----|:-----|:-----|
| ICP/G-ICP | 最近邻配准 | 小偏差、刚体 |
| NDT | 高斯体素匹配 | 多模态，收敛范围大 |
| TEASER++ | 鲁棒估计 | 离群点多 |
| FCGF/Predator | 深度特征 + RANSAC | 大场景 |

---

## 4. SLAM 谱系

| 类型 | 代表 | 特点 |
|:-----|:-----|:-----|
| 视觉 SLAM | ORB-SLAM2/3, DSO | 低成本，依赖光照纹理 |
| VIO | VINS-Mono, OKVIS | IMU + Camera 紧耦合 |
| LiDAR SLAM | LOAM, LeGO-LOAM, Cartographer | 几何精度高 |
| LiDAR-IMU | LIO-SAM | 因子图 + 预积分 |
| 多传感器 | VINS-Fusion, Cartographer | 松/紧耦合，鲁棒 |

### 4.1 视觉 SLAM 流程

```
图像 → 特征提取 → 匹配 → PnP + RANSAC → 滑窗 BA → 回环检测 → Pose Graph
```

### 4.2 LiDAR SLAM

- 提取边缘/平面特征
- 扫描匹配 + IMU 补偿
- Pose Graph 优化 + 回环

---

## 5. 多模态融合

| 融合方式 | 描述 | 示例 |
|:---------|:-----|:-----|
| 松耦合 | 各自估计再 EKF 融合 | robot_localization |
| 紧耦合 | 同一优化问题 | LIO-SAM, VINS-Fusion |
| Graph-SLAM | 因子图 | GTSAM 系列 |

---

## 6. 工程 Checklist

- [ ] 传感器外参 (Hand-Eye) 与时间同步
- [ ] 动态物体剔除 (语义分割/运动一致性)
- [ ] 地图管理：关键帧下采样、回环策略
- [ ] ROS2 数据通道：`pointcloud2`, `/imu/data`
- [ ] GPU 加速库：Cupoch、TensorRT

---

## 7. 代码片段

```python
# Open3D ICP
import open3d as o3d
source = o3d.io.read_point_cloud('scan1.pcd')
target = o3d.io.read_point_cloud('scan2.pcd')
reg = o3d.pipelines.registration.registration_icp(
    source, target, 0.05, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
print(reg.transformation)
```

```xml
<!-- LIO-SAM launch -->
<launch>
  <node pkg="lio_sam" type="lio_sam_imuPreintegration" name="lio_sam">
    <param name="pointCloudTopic" value="/velodyne_points" />
    <param name="imuTopic" value="/imu/data" />
    <param name="gpsTopic" value="/gps/fix" />
  </node>
</launch>
```

---

## 8. 面试 Q&A

1. **Visual vs LiDAR SLAM?** 视觉信息多但受光照影响，LiDAR 精度高但成本高；融合最稳健。
2. **ICP 什么时候失败？** 初值差、动态物体多、噪声大；可用全局配准/语义剔除。
3. **回环检测怎么做？** Bag-of-Words (DBoW2)、Scan Context、基于学习的 OverlapNet。
4. **动态物体如何处理？** 语义分割 + 运动一致性，或使用静态地图掩码。

---

## 📚 推荐

- *3D Point Cloud Processing* (教程)
- Open3D / PCL / Cupoch
- LIO-SAM, VINS-Fusion, ORB-SLAM3
- Scan Context / OverlapNet 回环库

---

[← Back to Theory Index](./README.md)
