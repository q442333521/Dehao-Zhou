# 机械臂智能抓取系统

## 说明
该系列代码基于b站分享教学视频开源：https://space.bilibili.com/22108883/lists

如果对您有帮助，麻烦点个star，谢谢！

---

## 📋 项目概述

这是一个基于视觉的机械臂智能抓取系统,包含多种抓取策略和AI模型集成。项目视频教程见: https://space.bilibili.com/22108883/lists

---

## 🎯 主要功能模块

### **1. GraspNet全流程代码** (深度学习抓取)

**位置**: `graspnet全流程代码/`

**功能**: 使用GraspNet深度学习模型进行智能抓取位姿预测

**核心文件**:
- `grasp.py` - 主程序入口
- `cv_process.py` - 视觉处理(YOLO-World目标检测 + SAM分割)
- `grasp_process.py` - GraspNet抓取推理

**使用步骤**:

1. **环境准备**:
   ```bash
   # 需要安装到graspnet官方根目录下
   pip install torch==1.8.0 numpy==1.23.5
   pip install ultralytics open-clip
   ```

2. **配置文件** (`config.yaml`):
   ```yaml
   ROBOT_TYPE: "RM65"  # 机械臂型号
   ```

3. **运行**:
   ```bash
   python grasp.py
   ```

4. **工作流程**:
   - 连接RealSense D435相机获取RGB-D图像
   - 通过YOLO-World检测目标物体(可指定类别)
   - 使用SAM模型进行精确分割
   - GraspNet预测最佳抓取位姿(位置+旋转+夹爪宽度)
   - 手眼标定坐标转换(相机坐标→机械臂基坐标)
   - 执行预抓取→抓取→放置流程

**关键参数** (grasp.py):
- `color_intr/depth_intr`: 相机内参
- `rotation_matrix/translation_vector`: 手眼标定外参
- `pre_grasp_offset`: 预抓取偏移距离(0.1m)

---

### **2. 传统抓取全流程代码** (传统视觉方法)

**位置**: `传统抓取全流程代码/`

**功能**: 使用传统计算机视觉方法进行垂直抓取

**核心文件**:
- `grasp.py` - 主程序
- `cv_process.py` - 视觉分割
- `vertical_grab/interface.py` - 垂直抓取算法

**使用方法**:
```bash
python grasp.py
```

**工作流程**:
- 获取RGB-D图像
- 目标分割(YOLO-World + SAM)
- 计算垂直抓取的三个关键位姿:
  1. `above_object_pose` - 物体上方位姿
  2. `correct_angle_pose` - 角度调整位姿
  3. `finally_pose` - 最终抓取位姿
- 执行抓取动作序列

**特点**:
- 固定垂直角度抓取 `[3.14, 0, 0]`
- 适合规则物体的快速抓取

---

### **3. 多模态交互式抓取** ⭐(最先进)

**位置**: `利用多模态模型Qwen2.5-VL理解人类意图，进行交互式抓取/`

**功能**: 通过语音交互,理解人类意图并执行抓取

**核心文件**:
- `grasp.py` - 主程序
- `vlm_process.py` - 多模态+语音处理
- `grasp_process.py` - GraspNet推理

**使用步骤**:

1. **部署多模态模型** (见`指令.txt`):
   ```bash
   conda create -n vllm python=3.10
   pip install vllm

   # 启动Qwen2.5-VL模型服务
   CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
     --model "/path/to/Qwen2.5-3B-Instruct-GPTQ-Int8/" \
     --served-model-name "qwen7b" \
     --host 0.0.0.0 --port 1034
   ```

2. **配置API**:
   - **VLM服务**: `http://192.168.1.6:1034` (vlm_process.py:59)
   - **ASR语音识别**: `http://192.168.1.6:3003` (vlm_process.py:231)
   - **TTS语音合成**: `http://192.168.1.6:3002` (vlm_process.py:256)

3. **运行**:
   ```bash
   python grasp.py
   ```

4. **交互流程**:
   ```
   系统: 🎙️ 请通过语音描述目标物体及抓取指令...
   用户: (语音) "请帮我拿起红色的苹果"
   系统:
     - 识别语音文本
     - 调用Qwen2.5-VL分析图像,返回物体边界框
     - 使用SAM精确分割
     - GraspNet预测抓取位姿
     - TTS播报: "好的,我看到了红色的苹果,正在为您抓取~"
     - 执行抓取动作
   ```

**技术亮点**:
- **多模态理解**: 图像+语音联合理解
- **自然交互**: 支持日常语言描述目标
- **智能选择**: AI自动从场景中识别并选择目标物体

---

### **4. 大模型任务分解**

**位置**: `利用大模型进行任务分解/利用vllm部署私有化大模型/`

**功能**: 使用vllm部署本地大模型进行复杂任务分解

**部署命令**:
```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model "/local/HXH/llm/Qwen2.5/Qwen2.5-3B-Instruct-GPTQ-Int8/" \
  --served-model-name "Qwen3B" \
  --host 0.0.0.0 --port 1034 \
  --gpu-memory-utilization 0.6  # 可选
```

---

## 🔧 通用组件

### **机械臂控制** (`robotic_arm_package/`)
- `robotic_arm.py`: 封装机械臂API
  - `Movej_P_Cmd()`: 位姿运动
  - `Movej_Cmd()`: 关节运动
  - `Set_Gripper_Pick()`: 夹爪闭合
  - `Set_Gripper_Release()`: 夹爪松开

### **坐标转换**
- `convert_d.py` / `convert_update.py`: 手眼标定坐标系转换
  - 相机坐标 → 末端执行器坐标 → 机械臂基坐标

### **辅助工具** (`libs/`)
- `auxiliary.py`: 工具函数(IP获取、弹窗提示等)
- `log_setting.py`: 日志配置

---

## 📊 系统架构流程

```
相机采集(RealSense D435)
    ↓
视觉处理(YOLO-World + SAM)
    ↓
[可选] 多模态意图理解(Qwen2.5-VL + ASR)
    ↓
抓取位姿预测(GraspNet/传统算法)
    ↓
坐标系转换(手眼标定)
    ↓
运动规划(预抓取→抓取→放置)
    ↓
机械臂执行
```

---

## ⚙️ 关键配置

**相机内参示例** (640x480):
```python
color_intr = {"ppx": 331.054, "ppy": 240.211,
              "fx": 604.248, "fy": 604.376}
depth_intr = {"ppx": 319.304, "ppy": 236.915,
              "fx": 387.897, "fy": 387.897}
```

**手眼标定外参** (需根据实际标定):
```python
rotation_matrix = [...]    # 3x3旋转矩阵
translation_vector = [...]  # [x, y, z]平移向量
```

---

## 🚀 快速开始建议

1. **初学者**: 从`传统抓取全流程代码`开始,理解基础流程
2. **进阶**: 使用`graspnet全流程代码`,体验深度学习抓取
3. **前沿体验**: 部署`多模态交互式抓取`,实现语音控制

---

## 📝 技术栈

- **深度学习**: GraspNet, YOLO-World, SAM, Qwen2.5-VL
- **视觉**: OpenCV, Open3D, RealSense
- **机器人**: 机械臂运动控制, 手眼标定
- **AI服务**: vLLM, ASR, TTS
- **框架**: PyTorch, Ultralytics

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 许可证

本项目遵循开源协议，详见视频教程说明。
