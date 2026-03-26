# VLN 通信与控制分阶段测试脚本说明

本文件为分阶段测试的脚本入口和说明，覆盖阶段 1~4 的自动化与手动测试需求。

## 阶段 1：基础通信测试
- vln_client_upload_image.py：上传图像到 vln_host，打印返回指令
- vln_client_cmdvel_test.py：直接发固定 cmd_vel，底层测试

## 阶段 2：PID 算法单独验证
- pid_offline_test.py：离线输入指令，输出 cmd_vel 曲线
- pid_openloop_test.py：手动输入指令，PID 输出 cmd_vel，记录数据

## 阶段 3：遥控器接管功能
- remote_mux_node.py：监听遥控器输入，优先级切换
- remote_joy_listener.py：监听 /joy topic，发布遥控 cmd_vel

## 阶段 4: 添加DWA局部规划器
- navigation包
- depthimage-to-laserscan包：机器狗前置的深度相机模拟激光后转成可以用于导航的costmap
- move_base包
- local_costmap配置
- 

## 阶段 5：端到端闭环测试
- vln_end2end_test.md：静态、低速、避障、长时间测试说明与数据记录建议

---

# 具体脚本清单
- realworld/vln_client_upload_image.py
- realworld/vln_client_cmdvel_test.py
- realworld/pid_offline_test.py
- realworld/pid_openloop_test.py
- realworld/remote_mux_node.py
- realworld/remote_joy_listener.py
- realworld/vln_end2end_test.md

---

# 说明
- 每个脚本均可独立运行，按阶段推进。
- 数据记录建议：rosbag/csv，便于后续分析。
- 详细参数与接口见各脚本注释。
