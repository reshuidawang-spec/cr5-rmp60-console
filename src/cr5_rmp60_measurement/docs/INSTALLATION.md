# 安装、迁移和更新

## 环境与依赖

支持 Ubuntu 22.04（Jammy）、系统 Python 3.10 和 ROS 2 Humble。先按 [ROS 官方说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) 配置 Humble 的 apt 软件源。

在仓库根目录执行：

```bash
bash install.sh --system
bash install.sh --build
bash install.sh --check
```

系统依赖列在 `src/cr5_rmp60_measurement/dependencies/apt-runtime.txt`，包括 ROS 2、MoveIt、PyQt5、NumPy、SciPy、Matplotlib、PyYAML、lxml、FCL、Eigen、geometric_shapes、编译器和中文字体。ROS 消息及 rclpy 不能使用同名 pip 包替代。

四个所需 ROS 包（`dobot_msgs_v4`、`cr_robot_ros2`、`dobot_rviz`、`cr5_moveit`）的源码和机器人网格已包含。系统依赖由 apt 获取，不是可断网安装的完整操作系统镜像。第一次运行本地碰撞检查会编译 FCL 桥接库，因此运行机也需要编译器和头文件。

这是从源码目录启动的软件，不单独安装成 ament 包。不要移动仓库内部的相对目录。工作区根目录可以改名；软件不要求使用原开发者用户名。特殊布局可设置 `CR5_WORKSPACE`，ROS 前缀可设置 `CR5_ROS_PREFIX`。

## 启动

```bash
bash launch_console.sh
```

默认不自动启动驱动。界面包含模型设置、定位、仿真规划、自动批次和 CSV 导入入口。若需要真实连接，先检查配置和现场尺寸，再显式使用 `CR5_START_DRIVER=1`，并在界面中填写正确机器人地址。不要把预填夹具坐标当作当前现场的位置。

日志为 `/tmp/cr5_five_branch_workbench.log`。数据在操作者选择的实验目录保存，不随仓库分发。

可选桌面菜单入口：

```bash
python3 src/cr5_rmp60_measurement/scripts/install_desktop_entry.py
```

桌面入口调用根目录 `launch_console.sh`，默认不自动启动机器人驱动。

## 更新

在仓库根目录执行：

```bash
git pull --ff-only
bash install.sh --build
bash install.sh --check
```

更新模型或算法后，旧路径与冻结凭据可能失效，应重新规划和检查。自己的配置修改应先备份，不能用其他安装的冻结参数覆盖本次安装。

## 当前边界

公开版保留当前软件的执行门控。实机自动采集默认禁用，CSV 导入不会解除禁用，也不会自动认可球心。先完成针号、坐标定义、当前几何、完整路径与本次安装的误差验证，再进入真实采集流程。
