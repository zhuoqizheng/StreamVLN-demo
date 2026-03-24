# UI 使用说明（D1 实机 VLN Demo）

本目录用于可视化与联调，默认面向 D1 实机 VLN 系统，不依赖手机。

## 1. 目录功能

- [ui/app.py](ui/app.py)
: Flask 后端，提供网页与 API 转发。
- [ui/templates/vln_web.html](ui/templates/vln_web.html)
: 浏览器摄像头 + VLN 动作可视化页面。
- [ui/templates/index.html](ui/templates/index.html)
: 单帧图像分析页面（可选，不是实机必需）。
- [ui/llm_config.py](ui/llm_config.py)
: 单帧 VLM 功能的模型配置（可选）。

## 2. D1 实机推荐使用页面

优先使用：

- `http://<LAPTOP_IP>:5000/vln_web`

该页面已支持：

- 摄像头选择
- 指令输入
- VLN 服务地址输入
- 单次调用 / 自动调用
- 动作序列可视化

## 3. 环境变量

运行前建议设置：

```bash
export UI_DEFAULT_VLN_SERVER_URL="http://<JETSON_IP>:5801/eval_vln"
```

可选（仅 index 页面用）：

```bash
export UI_SHOT_URL="http://<IMAGE_SOURCE>/shot.jpg"
```

## 4. 启动

```bash
cd /tf/StreamVLN/ui
python3 app.py
```

默认监听 `0.0.0.0:5000`。

## 5. 健康检查

UI 提供上游 VLN 服务探活接口：

- `GET /api/vln_health?server_url=http://<JETSON_IP>:5801/eval_vln`

当返回 `ok: true`，说明 Laptop 到 Jetson 的链路可用。

## 6. 与三机系统的关系

完整三机部署请参考仓库根目录说明：

- [d1_realworld.md](d1_realworld.md)

分工如下：

- D1 机载电脑：`realworld/d1_vln_client.py`
- Jetson：`streamvln/http_realworld_server.py`
- Laptop（本目录）：`ui/app.py`

## 7. 备注

- `index.html` 适合离线或单帧 VLM 验证，不是 D1 实机 VLN 主流程必需。
- D1 实机控制链路由机载客户端直接调用 Jetson `/eval_vln` 完成。
