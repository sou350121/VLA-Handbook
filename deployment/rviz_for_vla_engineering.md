# RViz：VLA 工程師的真機調試顯微鏡 (RViz for VLA Engineering)

> **適用對象**：把 VLA 從仿真往真機搬、需要看「機器人到底看到什麼 / 動作到底執行成什麼」的工程師
> **核心定位**：RViz 不是「ROS 自帶的可視化工具」這麼簡單——它是 VLA 工程師**唯一能把感知 / TF / 動作 chunk / 規劃軌跡同時擺在一個座標系裡比對**的工具。matplotlib 看 numpy 陣列，RViz 看「物理世界」。
> **核心痛點**：90% 的真機 VLA bug 不是模型問題，而是**座標系錯了 / 時間戳對不齊 / 點雲在錯的 frame 裡**——這些問題在 Python notebook 裡看不出來，在 RViz 裡一秒抓出來。

---

## 1. 為什麼 VLA 工程師需要 RViz

| 場景 | 不用 RViz | 用 RViz |
|------|---------|--------|
| 看深度相機輸出 | matplotlib 畫 2D 圖 | 直接渲染為 3D 點雲，可旋轉 |
| 對齊機械臂與相機 | print TF 矩陣肉眼算 | 看 TF 樹箭頭是否對 |
| 看模型輸出的 action chunk | 印 N×7 list | MarkerArray 畫成 3D 軌跡 |
| 比對 ground-truth vs predicted | 兩條軌跡列印對齊 | 兩條軌跡疊放在同一個 base_link 下 |
| 多相機 fusion debug | 各畫一張圖 | 多 PointCloud2 同時 render，frame 自動對齊 |

**核心邏輯**：RViz 把所有有 `header.frame_id` 的 ROS message **自動投影到同一個座標系**。你只要 publish，它就能畫出來。

---

## 2. ROS1 (rviz) vs ROS2 (rviz2) 差異

| 項目 | ROS1 (`rviz`) | ROS2 (`rviz2`) |
|------|--------------|---------------|
| 啟動 | `rosrun rviz rviz` | `ros2 run rviz2 rviz2` |
| 配置文件 | `.rviz`（YAML） | `.rviz`（YAML，**格式略不同**） |
| TF 來源 | `/tf` + `/tf_static` | 同左但 QoS 不同 |
| 預設 QoS | reliable | **best_effort 不會自動匹配** ← 大坑 |
| Launch 整合 | `rviz` node | `Node(package='rviz2', executable='rviz2', arguments=['-d', config])` |
| Foxglove 替代 | 較新生態 | 原生 WebSocket bridge 更友善 |

**踩坑警告**：ROS2 下若 publisher 用 `best_effort` QoS（攝影機驅動常見），rviz2 預設 `reliable` 會**完全不顯示**——必須在 display 設定裡手動切 `Reliability Policy: Best Effort`。新手 70% 的「為什麼點雲不出來」都是這條。

---

## 3. 核心面板與 VLA 必裝 displays

```
┌─────────────────────────────────────────────────────────┐
│  Tool bar  │  Views (3D) ──────────────  │  Displays    │
│            │                            │  ┌──────────┐ │
│            │                            │  │ Grid     │ │
│            │       (3D scene)            │  │ TF       │ │
│            │                            │  │ Robot... │ │
│            │                            │  │ ...      │ │
│            ├────────────────────────────┤  └──────────┘ │
│            │  Time / Selection / Tool   │  Add | Reset │
└─────────────────────────────────────────────────────────┘
```

### VLA 工作必開的 9 個 displays

| Display | ROS msg type | VLA 用途 |
|---------|------------|---------|
| **TF** | `tf2_msgs/TFMessage` | 看所有座標系是否連通 |
| **RobotModel** | reads `/robot_description` URDF | 看機械臂自己是否在「對的位置」 |
| **PointCloud2** | `sensor_msgs/PointCloud2` | RGBD 或 LiDAR 輸入 |
| **Image** | `sensor_msgs/Image` | 原始相機圖；用 `compressed` 省頻寬 |
| **Camera** | `sensor_msgs/Image` + `CameraInfo` | 同 Image 但帶 frustum，看相機朝向 |
| **MarkerArray** | `visualization_msgs/MarkerArray` | **VLA 動作 chunk 可視化** |
| **Path** | `nav_msgs/Path` | 規劃 / 預測軌跡 |
| **PoseArray** | `geometry_msgs/PoseArray` | 多個候選 grasp / waypoint |
| **InteractiveMarkers** | `visualization_msgs/InteractiveMarker` | 手動拖機械臂測 IK |

### Fixed Frame 的選擇（最常踩的設計題）

```
Global Options
  Fixed Frame: ?
  ├── world         ← 仿真常用，多機器人場景
  ├── map           ← SLAM 場景，被 amcl/cartographer publish
  ├── odom          ← 純里程計參考
  ├── base_link     ← 看相對機械臂的東西（推薦 VLA 真機調試）
  └── camera_link   ← 看純相機本位東西（不推薦長期）
```

**規則**：Fixed frame 應該是「**相對於它，世界是靜止的**」的那個 frame。VLA 桌面操作場景多用 `base_link`（機械臂底座）；移動底盤場景用 `map`。**選錯會看到所有東西在抖**——抖的是 Fixed frame 自己。

---

## 4. TF 樹：VLA 工作中最容易壞的東西

```
        world
         │
         ▼
       map  ← SLAM 在這層 publish
         │
         ▼
       odom  ← 移動底盤 publish
         │
         ▼
     base_link  ← 機械臂 URDF 起點
         │
         ├──► shoulder_link ──► ... ──► tool0
         │                                │
         │                                ▼
         │                           gripper_tip ← 末端執行器
         │
         ├──► camera_mount ──► camera_link ← 相機外參
         │                          │
         │                          ▼
         │                     camera_optical_frame  ← 影像所在！
         │
         └──► table_frame ← 任務參考（手動 publish）
```

### 常見 TF bug（VLA 真機 80% 出在這）

| 症狀 | 根因 | 修法 |
|------|------|------|
| 點雲飄在桌子下方 | 相機外參 frame 錯（用了 `camera_link` 不是 `camera_optical_frame`） | URDF / static_transform 修 |
| TF 紅線（disconnected） | 沒人 publish 某段中介 frame | `ros2 run tf2_ros static_transform_publisher` 補 |
| Markers 在原點 | 你忘了設 `marker.header.frame_id` | 程式裡明確設 |
| 一切都在抖 | `Fixed Frame` 選了一個會動的 frame（如 `tool0`） | 改 `base_link` 或 `world` |
| 「TF_OLD_DATA」warning | 系統時鐘 drift / 跨機器沒 NTP | `chrony` 或 `ros2 param set use_sim_time true` |

**Debug 神器**：
```bash
# ROS1
rosrun tf view_frames    # 生成 PDF 看樹結構
rosrun tf tf_echo base_link camera_link    # 印實時變換

# ROS2
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo base_link camera_link
```

---

## 5. VLA 8 個調試場景的 RViz 配方

### 5.1 看 RGB-D 點雲對齊到機械臂

```
Add → PointCloud2
  Topic: /camera/depth/color/points
  Style: Points (or Flat Squares)
  Size: 0.005
  Color Transformer: RGB8
  Reliability: Best Effort   ← ROS2 必設
Fixed Frame: base_link
```
**判斷對齊**：點雲應該「貼」在機械臂能碰到的桌面位置。漂在空中 = 外參錯。

### 5.2 把 VLA 模型輸出的 action chunk 畫出來

```python
# 假設 model.predict() 返回 7-DoF × T 的 chunk
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

def chunk_to_marker(chunk, frame_id="base_link"):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = node.get_clock().now().to_msg()
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = 0.005
    m.color.r, m.color.g, m.color.a = 0.0, 1.0, 1.0   # 綠
    m.points = [Point(x=p[0], y=p[1], z=p[2]) for p in chunk]
    return m
```
RViz 端：`Add → Marker → Topic: /vla/action_chunk` → 看綠線就是模型的計畫。

### 5.3 同時看 ground-truth 和 predicted（疊放）

兩條 LINE_STRIP，紅 GT、綠 pred，同一 frame_id，同一 `MarkerArray.markers[]`，給不同 `id`。

### 5.4 雙視角（手腕 + 肩部相機）並排

```
Panels → Views: 
  - View 1: Orbit, Fixed Frame=base_link
Add Camera display × 2:
  - Camera 1: Topic /wrist_cam/image, Frame /wrist_optical
  - Camera 2: Topic /shoulder_cam/image, Frame /shoulder_optical
Window → Tile → 三窗對照
```

### 5.5 Gripper 開合視覺化

URDF 已含 gripper joint → `RobotModel` display 自動跟隨 `/joint_states`。
若沒 URDF 只有 width 數字 → publish 兩個 cube `Marker`，根據 width 動態調 scale.x。

### 5.6 看 TF tree 健康度（不離開 RViz）

`Add → TF` display：
- 取消勾選 `All Enabled`
- 在 `Frames` 列表裡只勾你關心的（避免 50+ frame 把畫面糊掉）
- `Show Names`、`Show Axes`、`Show Arrows` 全開
- `Marker Scale: 0.3`（單位米）

### 5.7 點雲 + RGB 影像同步檢查

兩個 display 都選 `Reliability: Best Effort`，看 timestamp 是否一致：
- Toolbar 開 `Time` panel
- 比對 PointCloud `Time` vs Image `Time`
- 偏移 > 100ms → 可能 hardware sync 問題

### 5.8 軌跡回放（rosbag）

```bash
ros2 bag play rollout_001.bag --clock --loop
# RViz: 設 use_sim_time=true，Fixed Frame=base_link
```
**坑**：如果 bag 含 `/tf_static`，必須加 `--clock --read-ahead-queue-size 2000` 避免 static TF 過早被消化掉。

---

## 6. `.rviz` 配置文件——版本控制必納入

`.rviz` 是 YAML，記錄所有 displays / view / tool 配置。**強烈建議**：
- 每個任務一份 `configs/rviz/{task}.rviz`
- 跟 launch file 綁定：`rviz2 -d $(ros2 pkg prefix mypkg)/share/mypkg/rviz/grasping.rviz`
- Git 版本控制（diff 友善，純 YAML）

**常見可重用樣板**（VLA 桌面操作）：
```yaml
# 摘要：Fixed Frame, Grid, RobotModel, 兩個 Camera, PointCloud2, TF, MarkerArray
# 完整檔案見：configs/rviz/vla_debug.rviz
```

---

## 7. 真機 / 遠端 / Headless 渲染

| 場景 | 解法 |
|------|------|
| 機器人在隔壁房，你在筆電 | `ssh -X user@robot rviz2 -d cfg.rviz`（X11 forward；慢） |
| 機器人是 headless 工控機 | **Foxglove Studio + foxglove_bridge**（WebSocket，原生遠端） |
| 大流量點雲卡頓 | `image_transport republish` 壓縮 + 降採樣 |
| Docker 裡跑 RViz | `docker run --net=host --env DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ...` |
| Mac 連 Linux 機器人 | XQuartz + ssh -X（不順）→ **強烈建議改 Foxglove** |

**坑**：X11 forward 在 4K 點雲下可能 < 1 fps；Foxglove 在內網能 30+ fps。VLA 真機調試強烈建議切 Foxglove，RViz 留給本機開發機。

---

## 8. 常見坑與一線修法

| 症狀 | 修法 |
|------|------|
| 「No tf data」 | `ros2 topic list \| grep tf` 確認有 publish；`use_sim_time` 是否一致 |
| ROS2 下點雲完全不顯示 | Display → Reliability Policy → **Best Effort** |
| RobotModel 顯示但不動 | `/joint_states` 沒 publish，或 `use_sim_time` 不對 |
| Markers 一閃就消失 | `marker.lifetime` 設了短時間；改 `Duration(0)` 永久 |
| 點雲飄 / 抖 | Fixed Frame 選錯（選了會動的 frame） |
| RViz 啟動就 segfault | OpenGL 驅動問題（NVIDIA `__GLX_VENDOR_LIBRARY_NAME=nvidia`）或 Mesa 太舊 |
| 大點雲卡頓 | Style 改 `Points`（不用 `Flat Squares`）；voxel downsample |
| 跨機器顯示亂跳 | 多機器 NTP 不同步；`chrony` 或 `ntpd` |
| `MultiThreadedExecutor` 下 callback 卡 | RViz 不是這原因；查你自己的 publisher |

---

## 9. 推薦插件 / 替代品

| 插件 / 工具 | 用途 | 安裝 |
|------------|------|------|
| `rviz_visual_tools` | 寫 demo 時方便畫框 / 文字 / 軌跡 | `apt install ros-${ROS_DISTRO}-rviz-visual-tools` |
| `moveit_visual_tools` | MoveIt 規劃可視化 | MoveIt 自帶 |
| `rqt_tf_tree` | TF 樹 GUI（比 view_frames 互動好） | `ros-${ROS_DISTRO}-rqt-tf-tree` |
| `foxglove_bridge` | WebSocket bridge → Foxglove Studio | `ros-${ROS_DISTRO}-foxglove-bridge` |
| **Foxglove Studio** | **遠端 / 多面板 / 時間軸更強的 RViz 替代** | 從 https://foxglove.dev 下載 |
| `plotjuggler` | 數值時序（joint angle / loss / reward）— 補 RViz 短板 | `apt install ros-${ROS_DISTRO}-plotjuggler-ros` |

---

## 10. 何時該放棄 RViz

| 場景 | 改用 |
|------|------|
| 看純數值（loss、reward、joint torque）時序 | **PlotJuggler** |
| 跨機器遠端調試 | **Foxglove Studio** |
| Web 端團隊共享 demo | **Foxglove Cloud / NerdyVis** |
| 大規模點雲 (>10M points) | **Open3D** 或 **CloudCompare**（離線） |
| 6-DoF pose ground-truth 標註 | **CVAT 3D / SuperAnnotate** |
| ML metric monitoring | **wandb / TensorBoard**（不要硬塞 RViz） |

---

## 11. VLA 工程師的 RViz 心法（總結）

1. **TF 是命**：90% 真機 bug 出在 frame，不是模型。先看 TF 再看模型輸出。
2. **Fixed Frame 選靜止的**：桌面 → `base_link`，移動底盤 → `map`。
3. **MarkerArray 是 VLA 可視化的瑞士刀**：action chunk、grasp candidates、reference trajectory 全用它。
4. **配置文件進 git**：`.rviz` 是 YAML，diff 友善，跟 launch 綁定。
5. **遠端用 Foxglove，本機用 RViz**：別跟 X11 forward 死磕。
6. **ROS2 預設 QoS 是 reliable，相機驅動常給 best_effort**：display 端必須調匹配。
7. **不要把 RViz 當 ML monitor**：loss / reward / metric 用 PlotJuggler 或 wandb。
8. **`tf2_echo` + `view_frames` 是你的兩個老朋友**：90% TF 問題不用打開 RViz，先用這兩個工具。

---

## 12. 進一步閱讀

- 官方文檔：[wiki.ros.org/rviz](http://wiki.ros.org/rviz)（ROS1）/ [docs.ros.org/en/humble/Tutorials/Intermediate/RViz](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.html)（ROS2）
- TF2 教學：[docs.ros.org/en/humble/Tutorials/Intermediate/Tf2](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html)
- Foxglove：[foxglove.dev/docs](https://foxglove.dev/docs)
- ROS2 QoS 細節：[design.ros2.org/articles/qos.html](https://design.ros2.org/articles/qos.html)

---

[← Back to Deployment](./README.md)
