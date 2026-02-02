# Audio2Face Web UI

这是一个 Audio2Face SDK 的 Web 前端界面，允许用户通过网页上传音频文件，选择推理模型，并下载生成的面部动画数据。

## 功能特点

- 🎵 **音频上传**：支持 WAV, MP3, OGG, FLAC, M4A 格式
- 🤖 **模型选择**：支持 Regression 和 Diffusion 两种模型类型
- 📊 **实时状态**：显示推理进度和状态
- 📥 **结果下载**：下载 JSON 格式的面部动画数据
- 🚀 **真实推理**：使用 C++ 编译的 `a2f-web-inference` 程序进行 GPU 加速推理

## 系统要求

- Python 3.8 - 3.10
- NVIDIA GPU 支持 CUDA 12.8+
- 已完成 Audio2Face SDK 的构建和模型生成
- ffmpeg（可选，用于音频格式转换）

## 快速开始

### 1. 构建 C++ 推理程序

```bash
# 确保已经构建整个项目
./build.sh all release
```

这会编译 `a2f-web-inference` 程序到 `_build/release/audio2face-sdk/bin/` 目录。

### 2. 安装 Python 依赖

```bash
# 确保在项目根目录的虚拟环境中
source venv/bin/activate

# 安装 Web UI 依赖
pip install -r webui/requirements.txt
```

### 3. 确保模型已生成

在运行 Web UI 之前，请确保已经运行过 `gen_testdata.sh` 来生成 TensorRT 模型：

```bash
./gen_testdata.sh
```

### 4. 启动服务

```bash
# 运行启动脚本
./webui/run_webui.sh
```

或者手动启动：

```bash
# 启动后端 API 服务
cd webui/backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 在另一个终端启动前端服务（可选）
cd webui/frontend
python -m http.server 3000
```

### 5. 访问界面

- **前端界面**：http://localhost:3000 或直接打开 `webui/frontend/index.html`
- **后端 API**：http://localhost:8000
- **API 文档**：http://localhost:8000/docs

## C++ 推理程序使用

可以直接使用命令行运行推理：

```bash
# 查看帮助
./_build/release/audio2face-sdk/bin/a2f-web-inference --help

# 列出可用模型
./_build/release/audio2face-sdk/bin/a2f-web-inference --list

# 运行推理
./_build/release/audio2face-sdk/bin/a2f-web-inference \
  --model mark \
  --audio sample-data/audio_4sec_16k_s16le.wav \
  --output result.json
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | 模型 ID (mark, claire, james, multi-diffusion) | mark |
| `-a, --audio` | 输入音频文件路径 (推荐 16kHz WAV) | 必需 |
| `-o, --output` | 输出 JSON 文件路径 (- 表示标准输出) | - |
| `-d, --data-dir` | 数据目录路径 | _data |
| `-f, --fps` | 输出帧率 | 60 |
| `-i, --identity` | 扩散模型的身份索引 | 0 |
| `-l, --list` | 列出可用模型 | - |
| `-h, --help` | 显示帮助 | - |

## 使用流程

1. **上传音频**：点击上传区域或拖拽音频文件
2. **选择模型**：从可用的模型中选择一个
3. **开始推理**：点击"开始生成面部动画"按钮
4. **下载结果**：推理完成后下载 JSON 结果文件

## API 接口

### 获取可用模型
```
GET /api/models
```

### 上传音频
```
POST /api/upload
Content-Type: multipart/form-data
Body: file=<audio_file>
```

### 开始推理
```
POST /api/inference
Content-Type: multipart/form-data
Body: model_id=<model_id>&audio_file_id=<file_id>
```

### 获取结果状态
```
GET /api/results/{job_id}
```

### 下载结果
```
GET /api/download/{job_id}
```

## 输出格式

推理结果以 JSON 格式返回，兼容 Apple ARKit 的 52 个 FACS 混合形状（blendshapes）格式：

```json
{
  "exportFps": 60.0,
  "trackPath": "/path/to/audio.wav",
  "numPoses": 68,
  "numFrames": 240,
  "facsNames": [
    "eyeBlinkLeft",
    "eyeLookDownLeft",
    "eyeLookInLeft",
    "...",
    "tongueOut",
    "tongueTipUp",
    "..."
  ],
  "weightMat": [
    [0.015, 0.0, 0.0, ...],
    [0.023, 0.001, 0.0, ...],
    ...
  ],
  "joints": ["jaw", "eye_L", "eye_R"],
  "rotations": [],
  "translations": []
}
```

### 输出字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `exportFps` | float | 输出帧率 (默认 60) |
| `trackPath` | string | 音频文件路径 |
| `numPoses` | int | 混合形状数量 (52 皮肤 + 16 舌头 = 68) |
| `numFrames` | int | 总帧数 |
| `facsNames` | array | 52 个 FACS 混合形状名称 + 16 个舌头形状 |
| `weightMat` | 2D array | [numFrames][numPoses] 的混合形状权重矩阵 |
| `joints` | array | 关节名称列表 |
| `rotations` | array | 关节旋转数据（预留） |
| `translations` | array | 关节平移数据（预留） |

### FACS 混合形状列表

皮肤混合形状 (52个):
- 眼睛: eyeBlinkLeft/Right, eyeLookDown/In/Out/UpLeft/Right, eyeSquintLeft/Right, eyeWideLeft/Right
- 下巴: jawForward, jawLeft, jawRight, jawOpen
- 嘴巴: mouthClose, mouthFunnel, mouthPucker, mouthLeft/Right, mouthSmile/Frown/Dimple/Stretch/Press/Roll/Shrug/LowerDown/UpperUpLeft/Right
- 眉毛: browDownLeft/Right, browInnerUp, browOuterUpLeft/Right
- 脸颊: cheekPuff, cheekSquintLeft/Right
- 鼻子: noseSneerLeft/Right
- 舌头: tongueOut

舌头混合形状 (16个):
- tongueTipUp/Down/Left/Right, tongueRollUp/Down/Left/Right
- tongueUp/Down/Left/Right/In/Stretch/Wide/Narrow
```

## 可用模型

| 模型 ID | 名称 | 类型 | 描述 |
|---------|------|------|------|
| mark | Mark (Regression v2.3) | regression | 基于回归的 Mark 角色模型 |
| claire | Claire (Regression v2.3.1) | regression | 基于回归的 Claire 角色模型 |
| james | James (Regression v2.3.1) | regression | 基于回归的 James 角色模型 |
| multi-diffusion | Multi-Diffusion (v3.0) | diffusion | 多身份扩散模型 |

## 故障排除

### 模型显示"不可用"
请确保已运行 `./gen_testdata.sh` 生成 TensorRT 模型文件。

### 音频转换失败
安装 ffmpeg 或 pydub：
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# 或使用 pydub
pip install pydub
```

### 无法连接到服务器
确保后端服务正在运行，且端口 8000 未被占用。

## 文件结构

```
webui/
├── README.md           # 本文档
├── run_webui.sh        # 启动脚本
├── backend/
│   └── app.py          # FastAPI 后端服务
├── frontend/
│   └── index.html      # 前端界面
├── uploads/            # 上传的音频文件（自动创建）
└── results/            # 推理结果文件（自动创建）
```

## 许可证

MIT License - 详见项目根目录的 LICENSE.txt
