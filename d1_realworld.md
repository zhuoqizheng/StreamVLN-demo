# D1 Real-World VLN Demo 部署说明

本说明用于真实机器狗演示场景，且与手机无关。

## 1. 部署目标

在同一个 WiFi 局域网下联通三台设备：

- 机器狗机载电脑：运行 [realworld/d1_vln_client.py](realworld/d1_vln_client.py)
- Jetson Orin 边缘计算电脑：运行 [streamvln/http_realworld_server.py](streamvln/http_realworld_server.py)
- 笔记本电脑（仅可视化）：运行 [ui/app.py](ui/app.py)

核心链路：

1. D1 机载相机帧 -> `d1_vln_client.py`
2. `d1_vln_client.py` -> Jetson `POST /eval_vln`
3. Jetson 返回动作序列 -> D1 执行 `move(vx, 0, yaw_rate)`
4. 笔记本 `vln_web.html` 仅做可视化和联调，不直接控制 D1

## 2. 已对齐的接口

`d1_vln_client.py` 请求格式：

- multipart 文件字段：`image`
- 表单字段：`json={"reset": bool, "instruction": "..."}`

`http_realworld_server.py` 兼容格式：

- `image` 或 `file`
- `json` 或 plain form 的 `reset/instruction/instruction_text`

`http_realworld_server.py` 返回格式：

- `{"action": [int, ...], "instruction": "...", "frame_id": n, "terminate": bool}`

健康检查：

- Jetson: `GET /healthz`
- UI 转发检查: `GET /api/vln_health?server_url=http://<JETSON_IP>:5801/eval_vln`

## 3. 启动步骤

### 3.1 Jetson Orin

```bash
cd /tf/StreamVLN
python3 streamvln/http_realworld_server.py --device cuda:0
```

检查：

```bash
curl http://<JETSON_IP>:5801/healthz
```

### 3.2 笔记本（可视化）

```bash
cd /tf/StreamVLN/ui
export UI_DEFAULT_VLN_SERVER_URL="http://<JETSON_IP>:5801/eval_vln"
python3 app.py
```

浏览器打开：

- `http://<LAPTOP_IP>:5000/vln_web`

检查 UI 到 Jetson：

- `http://<LAPTOP_IP>:5000/api/vln_health?server_url=http://<JETSON_IP>:5801/eval_vln`

### 3.3 D1 机载电脑

```bash
cd /tf/StreamVLN/realworld

# D1 SDK 路径（二选一）
export D1_SDK_ROOT="/path/to/agibot_D1_Edu-Ultra"
# 或
# export D1_SDK_LIB_PATH="/path/to/agibot_D1_Edu-Ultra/lib/zsl-1/aarch64"

# D1 SDK 通信参数
export D1_MODEL="zsl-1"               # 或 zsl-1w
export D1_LOCAL_IP="<DOG_PC_IP>"
export D1_LOCAL_PORT="43988"
export D1_ROBOT_IP="192.168.234.1"

# 相机话题（按机载 ROS2 实际话题修改）
export D1_RGB_TOPIC="/camera/camera/color/image_raw"

# 指向 Jetson 推理服务
export STREAMVLN_SERVER_URL="http://<JETSON_IP>:5801/eval_vln"
export STREAMVLN_INSTRUCTION="沿着办公室主走廊前进，到路口后左转"

# 网络鲁棒性
export STREAMVLN_TIMEOUT="150"
export STREAMVLN_RETRY="2"

python3 d1_vln_client.py
```

## 4. 网络与防火墙建议

- 三台设备先互相 `ping`，确保同网段可达。
- Jetson 放通 `5801` 端口，笔记本放通 `5000` 端口。
- 若跨子网，先确认路由策略与防火墙允许 TCP 转发。

## 5. 常见问题

1. 机器人无动作但 Jetson 有返回
- 检查 D1 SDK `initRobot` IP/端口与机器人配置是否一致。
- 检查 `standUp` 返回值是否正常。

2. UI 可视化失败
- 访问 `GET /api/vln_health` 查看上游错误信息。
- 确认 `UI_DEFAULT_VLN_SERVER_URL` 指向 Jetson 正确地址。

3. 动作明显不符合场景
- 检查 `STREAMVLN_INSTRUCTION` 是否传入并生效。
- 检查 D1 相机安装方向、分辨率与训练假设是否一致。

## 6. 说明

本系统的实机演示路径不依赖手机。

- `ui/index.html` 属于单帧 VLM 像素点分析页，可选使用。
- 实际 D1 VLN 演示建议使用 `ui/vln_web.html` 作为联调可视化页。
