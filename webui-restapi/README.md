# Audio2Face REST API Web Interface

这是一个基于 Audio2Face REST API 的 Web 前端界面，允许用户通过浏览器上传音频文件并生成面部动画数据。

## 功能特点

- 🎵 **音频上传**: 支持 WAV, MP3, OGG, FLAC, M4A 格式
- 🔄 **音频复用**: 选择已上传的音频文件重新推理
- 🎭 **自动推理**: 使用 Audio2Face REST API 自动生成面部动画
- 📊 **实时日志**: 显示处理过程和状态
- 📥 **一键下载**: 直接下载生成的动画数据文件
- 🌐 **局域网访问**: 支持多设备同时访问
- 🎨 **现代界面**: 简洁美观的深色主题 UI

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd Audio2Face-3D-SDK
```

### 2. 启动 Audio2Face Headless 服务

在一个终端窗口中启动 Audio2Face headless 模式（需要先安装 Audio2Face 2023.2.0）：

```bash
~/.local/share/ov/pkg/audio2face-2023.2.0/audio2face_headless.sh
```

等待服务启动完成，REST API 将运行在 `http://127.0.0.1:8011`

你可以访问 http://127.0.0.1:8011/docs 查看 API 文档。

### 3. 安装 Python 依赖

```bash
pip install -r webui-restapi/requirements.txt
```

或手动安装：

```bash
pip install fastapi uvicorn httpx python-multipart
```

### 4. 启动 Web 服务器

在另一个终端窗口中：

```bash
cd webui-restapi
python3 server.py
```

服务器将启动在 `http://0.0.0.0:8000`

### 5. 访问 Web 界面

**本地访问：**
```
http://127.0.0.1:8000
```

**局域网访问：**

1. 查看服务器 IP 地址：
   ```bash
   hostname -I
   ```

2. 在局域网内其他设备浏览器访问：
   ```
   http://<服务器IP>:8000
   ```
   例如：`http://192.168.1.100:8000`

## 使用说明

### 第一步: 选择或上传音频

**方式一：上传新音频**
1. 点击上传区域或拖拽音频文件
2. 支持的格式: WAV, MP3, OGG, FLAC, M4A

**方式二：使用已有音频**
1. 点击"🔄 刷新"按钮获取已上传的音频列表
2. 从下拉框中选择音频文件

### 第二步: 配置设置

- **导出格式**:
  - **JSON (Blendshape Weights)**: 导出 blendshape 权重数据（推荐）
  - **USD (Geometry Cache)**: 导出几何缓存动画
  - **JSON (Emotion Keys)**: 导出情感关键帧数据
- **帧率 (FPS)**: 默认 25，范围 1-120

### 第三步: 生成并下载

1. 点击 "▶️ 开始处理" 按钮
2. 系统会自动执行以下步骤:
   - 加载 USD 文件
   - 上传/选择音频并设置轨道
   - 生成面部动画（自动推理）
   - 导出结果文件
3. 处理完成后，点击 "⬇️ 下载结果" 按钮获取生成的文件

## 系统架构

```
┌─────────────┐         ┌──────────────┐         ┌─────────────────┐
│   Browser   │ ◄─────► │  FastAPI     │ ◄─────► │  Audio2Face     │
│  (前端界面)  │  HTTP   │  Server      │  REST   │  Headless       │
│             │         │  (port 8000) │  API    │  (port 8011)    │
└─────────────┘         └──────────────┘         └─────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  tmp/        │
                        │  - 音频文件   │
                        │  - 导出结果   │
                        │  - 日志文件   │
                        └──────────────┘
```

## 文件结构

```
webui-restapi/
├── index.html                           # 前端界面
├── server.py                            # Python 后端服务器
├── requirements.txt                     # Python 依赖
├── README.md                            # 本文档
├── claire_solved_arkit_offline.usd      # USD 角色文件
└── tmp/                                 # 临时文件目录
    ├── *.wav, *.mp3                    # 上传的音频
    ├── *_bsweight.json                 # 导出的动画数据
    └── server.log                      # 服务器日志
```

## API 调用流程

```
1. GET  /api/config              → 获取服务器配置（USD路径、导出目录）
2. POST /A2F/USD/Load            → 加载 USD 文件到场景
3. POST /api/upload              → 上传音频文件到服务器（可选）
4. GET  /A2F/Player/GetInstances → 获取播放器实例
5. POST /A2F/Player/SetRootPath  → 设置音频根路径
6. POST /A2F/Player/GetTracks    → 获取可用音频列表（刷新功能）
7. POST /A2F/Player/SetTrack     → 设置音频轨道
8. POST /A2F/A2E/GenerateKeys    → 生成情感关键帧（触发推理）
9. GET  /A2F/Exporter/GetBlendShapeSolvers → 获取解算器
10. POST /A2F/Exporter/ExportBlendshapes   → 导出动画数据
11. GET  /api/download/{filename} → 下载结果文件
```

## 进阶配置

### 修改监听端口

编辑 `server.py`，修改最后一行：

```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # 改为其他端口
```

### 使用不同的 USD 文件

将你的 USD 文件放在 `webui-restapi/` 目录下，系统会自动使用该目录下的 USD 文件。

### 后台运行服务器

```bash
nohup python3 server.py > /dev/null 2>&1 &
```

查看日志：
```bash
tail -f webui-restapi/tmp/server.log
```

停止服务器：
```bash
pkill -f "python3 server.py"
```

## 常见问题

### Q: API 连接失败（检查API连接...）

**A:** 确保 Audio2Face headless 服务正在运行：
```bash
# 检查进程
ps aux | grep audio2face_headless

# 重新启动
~/.local/share/ov/pkg/audio2face-2023.2.0/audio2face_headless.sh
```

访问 http://127.0.0.1:8011/docs 确认 API 可用。

### Q: 局域网无法访问

**A:** 检查防火墙设置：
```bash
# Ubuntu/Debian
sudo ufw allow 8000/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

### Q: 上传失败

**A:** 检查以下问题：
- 文件格式是否支持（WAV, MP3, OGG, FLAC, M4A）
- 磁盘空间是否充足
- `tmp/` 目录权限是否正确

### Q: 推理失败

**A:** 可能的原因：
- USD 文件未正确加载
- 音频文件格式不兼容
- TensorRT 模型未正确加载（首次运行需要时间构建）
- 内存不足

查看详细日志：
```bash
tail -f webui-restapi/tmp/server.log
```

### Q: 已有音频列表为空

**A:** 点击"🔄 刷新"按钮后，列表会显示 `tmp/` 目录下的所有音频文件。确保至少上传过一次音频。

## 系统要求

- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **Python**: 3.8+
- **Audio2Face**: 2023.2.0
- **内存**: 16GB+ 推荐
- **显卡**: NVIDIA GPU with CUDA support

## 技术栈

- **前端**: HTML5, CSS3, Vanilla JavaScript
- **后端**: Python, FastAPI, Uvicorn
- **API 代理**: httpx (解决 CORS 问题)
- **文件处理**: Python multipart
- **API**: Audio2Face REST API

## 开发说明

### 路径自动配置

系统会自动使用服务器当前目录的相对路径：
- USD 文件：`webui-restapi/claire_solved_arkit_offline.usd`
- 导出目录：`webui-restapi/tmp/`
- 日志文件：`webui-restapi/tmp/server.log`

前端会通过 `/api/config` 接口获取完整路径。

### CORS 代理

所有前端对 Audio2Face API 的请求都通过后端代理（`/api/a2f/*`），避免跨域问题。

### 调试模式

查看浏览器控制台（F12）获取详细的调试信息。

## 许可证

此项目遵循 Audio2Face SDK 的许可证。

## 支持

如有问题，请参考：
- Audio2Face SDK 文档
- REST API 文档: http://127.0.0.1:8011/docs
- 项目主 README: `/Audio2Face-3D-SDK/README.md`
