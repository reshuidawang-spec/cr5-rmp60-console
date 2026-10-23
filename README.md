# CR5 RMP60 五向测针控制台

公开的软件发布仓库，包含当前五向标准球实验台、2026-10-23 新增的旧 GUI 触发 CSV 导入入口、自动规划与整机碰撞检查，以及配套 Dobot 驱动、消息、MoveIt 配置和 CR5 网格。

这是独立的软件快照，不含原始实验数据、拟合结果、研究文稿、账号凭据或旧研究仓库的 Git 历史。配置文件中的几何和夹具数值是软件模型预填值，不能作为新安装的标定结果。

## 下载

公开仓库无需 GitHub 登录或邀请：

```bash
git clone https://github.com/reshuidawang-spec/cr5-rmp60-console.git ~/cr5_console_ws
cd ~/cr5_console_ws
```

也可以在 GitHub 页面选择 **Code → Download ZIP**。解压后的整个仓库目录就是工作区，保持里面的 `src/` 布局。

## 安装和启动

支持 **Ubuntu 22.04 / ROS 2 Humble / Python 3.10**。先按 [ROS 2 Humble 官方安装说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) 配好软件源，再执行：

```bash
bash install.sh --system
bash install.sh --build
bash install.sh --check
bash launch_console.sh
```

- `--system`：安装依赖清单，需要 sudo 和网络。
- `--build`：从源码编译四个必需 ROS 包，不连接机器人。
- `--check`：检查 Python/ROS 依赖、模型与本地 FCL 编译，不连接机器人。
- `launch_console.sh`：默认不自动启动机器人驱动；界面上的连接和点动操作仍由操作者控制。

需要连接真实控制器时，先确认现场配置和地址，再显式启动驱动：

```bash
IP_address=192.168.5.1 CR5_START_DRIVER=1 bash launch_console.sh
```

基础标定控制台与标准球实验台的默认入口均不会自动连接机器人。实机自动采集仍由 `sphere_motion_policy.json` 的 `commissioning.verified=false` 禁用。需要本次安装的跟踪、停止和几何误差验证证据；公开软件的预填配置不替代这些记录。

## 最新基础标定控制台

包含当前部署版本的按住点动、松开停止、DI 触发采集和触发记录多选删除／备份功能：

```bash
bash launch_calibration.sh
```

这个入口默认不自动连接机器人。先启动并核实配套驱动，再在界面中连接。

## 文档

- [安装、迁移和更新](src/cr5_rmp60_measurement/docs/INSTALLATION.md)
- [控制台及 CSV 导入说明](src/cr5_rmp60_measurement/docs/CONSOLE_GUIDE.md)
- [系统依赖清单](src/cr5_rmp60_measurement/dependencies/apt-runtime.txt)
- [第三方来源](THIRD_PARTY.md)

## 离线回归

```bash
source src/cr5_rmp60_measurement/scripts/source_runtime.sh
cd src/cr5_rmp60_measurement
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests
```

测试使用合成输入和临时文件，不需要实验原始数据或机器人连接。生成的运行数据、日志和编译产物已加入忽略清单。
