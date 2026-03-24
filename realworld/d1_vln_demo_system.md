# D1 Real-World VLN Demo System (3-Machine)

本文档对齐三个核心进程，构成真实世界 D1 机器狗 VLN Demo：

- 机器人本体：运行 `realworld/d1_vln_client.py`
- Jetson Orin：运行 `streamvln/http_realworld_server.py`
- 笔记本：运行 `ui/app.py`

## 1. 系统拓扑

数据流分为两条：

1. 机器人自治链路（真正控制 D1）
- D1 相机帧 -> `d1_vln_client.py`
- `d1_vln_client.py` HTTP 调用 Jetson `http_realworld_server.py:/eval_vln`
- Jetson 返回动作序列 `[1,2,3,0]`
- `d1_vln_client.py` 用 PID + D1 SDK `move(vx, vy, yaw_rate)` 控制本体

2. 浏览器调试链路（可视化与联调）
- 浏览器摄像头 -> 笔记本 `ui/app.py:/api/eval_vln`
- `ui/app.py` 转发到 Jetson `http_realworld_server.py:/eval_vln`
- 浏览器展示动作与时延（不直接控制机器狗）

## 2. 接口对齐点

已完成对齐：

- Jetson 服务 `POST /eval_vln` 同时支持
  - `files.image` 或 `files.file`
  - `data.json`（JSON 字符串）
  - 兼容 `instruction`、`instruction_text`、`reset`
- Jetson 返回统一结构
  - `{"action": [int, ...], "instruction": "...", "frame_id": n, "terminate": bool}`
- Jetson 新增 `GET /healthz`
  - 用于 UI 和运维检查
- UI 新增 `GET /api/vln_health`
  - 透传检查上游 Jetson `healthz`
- D1 客户端增强 HTTP 容错
  - 支持 `action` / `actions`
  - 支持超时与重试环境变量

## 3. 启动顺序

建议顺序：Jetson -> 笔记本 UI -> 机器人。

### 3.1 Jetson Orin（运行 VLN 推理服务）

在 Jetson 上进入仓库后运行：

```bash
cd /tf/StreamVLN
python3 streamvln/http_realworld_server.py --device cuda:0
```

默认监听：`0.0.0.0:5801`

可先检查：

```bash
curl http://<JETSON_IP>:5801/healthz
```

### 3.2 笔记本（运行 UI 调试服务）

```bash
cd /tf/StreamVLN/ui
export UI_DEFAULT_VLN_SERVER_URL="http://<JETSON_IP>:5801/eval_vln"
python3 app.py
```

打开页面：
- `http://<LAPTOP_IP>:5000/vln_web`

可检查：
- `http://<LAPTOP_IP>:5000/api/vln_health?server_url=http://<JETSON_IP>:5801/eval_vln`

### 3.3 D1 机器人（运行控制客户端）

```bash
cd /tf/StreamVLN/realworld

# D1 SDK 路径（二选一）
export D1_SDK_ROOT="/path/to/agibot_D1_Edu-Ultra"
# 或 export D1_SDK_LIB_PATH="/path/to/agibot_D1_Edu-Ultra/lib/zsl-1/aarch64"

# D1 SDK 通信参数
export D1_MODEL="zsl-1"                  # 或 zsl-1w
export D1_LOCAL_IP="192.168.234.15"      # 控制端本机IP
export D1_LOCAL_PORT="43988"
export D1_ROBOT_IP="192.168.234.1"

# 相机话题
export D1_RGB_TOPIC="/camera/camera/color/image_raw"

# 指向 Jetson 推理服务
export STREAMVLN_SERVER_URL="http://<JETSON_IP>:5801/eval_vln"
export STREAMVLN_INSTRUCTION="沿着办公室主要道路行走"

# 网络抖动可调
export STREAMVLN_TIMEOUT="150"
export STREAMVLN_RETRY="2"

python3 d1_vln_client.py
```

## 4. 常见联调问题

1. UI 正常但机器人不动
- 先看 D1 客户端日志里是否成功 `standUp`。
- 再确认 D1 SDK `initRobot` 三元组 IP/端口与机器人配置一致。

2. `connect time out` / 无法连接 D1
- 按 AgiBot 文档检查 `sdk_config.yaml` 的 `target_ip/target_port`。
- 确认 `SDK_CLIENT_IP` 与控制端 IP 同网段。

3. Jetson `/eval_vln` 报错缺少字段
- 现在服务支持多种表单格式；若仍报错，重点检查是否有 `multipart image`。

4. 动作返回正常但路径异常
- 优先检查 instruction 是否被正确传入。
- 确认当前相机视角与训练时前视相机一致。

## 5. 建议的实机演示流程

1. UI 页面先验证 Jetson 端 VLN 正常返回动作。
2. 机器人原地站立，先短指令测试（如“向前 1 米后停止”）。
3. 再切换到完整办公室导航指令。
4. 演示结束时停止 D1 客户端；可配置 `D1_LIE_DOWN_ON_EXIT=1` 自动趴下。
