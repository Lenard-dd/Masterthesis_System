# 桌面 sparse 抓取策略

## 当前行为

`config/grasping.yaml` 的 `graspgenx_subprocess.strategy` 默认设为 `sparse`。
项目扩展服务复用本机 GraspGenX 的 `run_graspmoe`，强制 OBB 分支使用
`obb_density="sparse"`，同时返回扩散分支候选。不会自动切换到 dense。

默认生成设置：

- 扩散采样数：200。
- sparse 平面角度采样数：36；相对于物体点云包围盒顶面的高度偏移：`-2,0` cm。
- 返回候选上限：100；其中最多为 sparse 保留 24 个名额，其余给扩散候选。
- MoveIt 的有界筛选最多检查 30 个，其中优先为 sparse 保留最多 12 个名额。
- 两个分支仍使用模型分数；后续已有的分数、位姿及运动检查继续生效。
- sparse 候选使用完整姿态约束进行规划，避免姿态放宽后重新产生明显倾斜。

`-2,0` 是候选采样高度，不是已校准的闭爪深度。低矮物体的可用高度需要通过
预览和实验判断；本次没有新增环境点云碰撞过滤。

## 坐标关系

支持上下文的生成入口通过现有 `TFPoseTransformer` 将目标点云从其原始 frame
转换到 `base_link`，保存单独的 PLY。生成成功后同步更新用于候选定位的
frame、中心点和平面主轴。原始相机 PLY 保留。

runner 对新点云去中心化，模型返回的候选由已有定位流程恢复到 `base_link`。
随后沿用原有 grasp-frame → tool0 偏置，避免重复添加中心点或工具长度。

此版本按 `base_link` 的 +Z 定义竖直方向，适用于当前机器人底座安装于桌面的
工位。它没有单独旋转到配置的实测桌面法线；桌面明显倾斜时需要扩展此转换。
TF 使用现有工具的最新变换，因此采集点云到生成候选期间应保持机械臂静止。
这不是图像时间戳同步或手眼标定的实现。

没有点云 frame、TF 不可用或转换失败时，会明确报告并使用原扩散路径。
仍使用旧版服务时，也会报告 `diffusion fallback`。模型推理错误不会伪装成
成功或静默切换到 mock。

## 启动

GUI 的源配置 `hitl_gui/config/gui_config.yaml` 已指向新服务脚本；重启 GUI
及它管理的 GraspGenX 服务后生效。若从安装目录加载配置，应先按原方式重新
构建 `llm_skill_robot`、`hitl_gui` 并 source 工作区。已经运行的旧服务不会热更新。

手动启动时，在原 GraspGenX 环境内运行：

```bash
conda activate graspgenx
cd ~/GraspGenX/GraspGenX
python /home/lenard/dev_ws/src/LLM_Ros/scripts/graspgenx_sparse_server.py \
  --config ~/GraspGenX/GraspGenX/ext/graspgenx_checkpoints/release \
  --assets_dir ~/GraspGenX/GraspGenX/assets \
  --default_gripper robotiq_2f_140 \
  --port 5556
```

使用同一端口时只运行一个服务。仍沿用现有 SAM3 / GraspGenX GPU 交接流程。
没有 TF 上下文的旧宽度测试等入口会继续请求扩散模式。

## 首轮验证

1. 先用原有 GUI 预览或 `ros2 run llm_skill_robot ros-graspgenx-plan-only-demo`。
2. 确认生成消息包含 `strategy=sparse`。若出现 fallback，检查服务版本和 TF。
3. 候选的 `metadata.generation_branch` 区分 `obb_sparse` 和 `diffusion`。
   sparse 候选的接近轴应接近 `base_link` 的 `[0,0,-1]`。
4. 检查物体点云和候选在 RViz 中的位置一致，夹爪宽度、抓取深度和姿态合理。
5. 测试时记录物体尺寸、最终候选来源/分数、是否可达及预览中的关节运动量。

## 回退

将 `graspgenx_subprocess.strategy` 改为 `diffusion` 并重启读取该配置的应用。
扩展服务仍支持标准 `infer`，因此回退不要求更换服务。也可以使用官方原服务。

## 验证范围

单元测试覆盖参数解析、旧服务回退、候选名额、点云旋转与中心点恢复、标准
服务动作兼容；没有将 mock 模型测试当成真实模型质量或实机抓取成功的证据。
