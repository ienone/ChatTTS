# ChatTTS 毕设 API

基于文本描述的零样本语音合成 - FastAPI 后端

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件配置模型路径等
```

### 3. 启动服务

```bash
# 正常模式 (需要 GPU)
python main.py

# Mock 模式 (本地调试，无需 GPU)
MOCK_MODE=true python main.py
```

### 4. 访问 API 文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 端点

### TTS 推理

| 端点 | 方法 | 描述 |
|------|------|------|
| `/tts/generate` | POST | 文本转语音 |
| `/tts/generate_random` | POST | 使用随机音色 |

### 音色管理

| 端点 | 方法 | 描述 |
|------|------|------|
| `/speaker/register` | POST | 上传音频注册音色 |
| `/speaker/list` | GET | 列出所有音色 |
| `/speaker/{id}` | DELETE | 删除音色 |
| `/speaker/predict_voice` | POST | **[毕设核心]** 描述映射 |

### 有声书生成

| 端点 | 方法 | 描述 |
|------|------|------|
| `/novel/analyze` | POST | 分析长文本 |
| `/novel/generate` | POST | 生成有声书 |

### 实验工具

| 端点 | 方法 | 描述 |
|------|------|------|
| `/research/tsne` | GET | t-SNE 可视化数据 |
| `/research/embedding/{id}` | GET | 查看 embedding 信息 |
| `/research/health` | GET | 健康检查 |

## 项目结构

```
api_thesis/
├── main.py              # 主入口
├── config.py            # 配置管理
├── requirements.txt     # 依赖
├── .env.example         # 环境变量示例
├── core/
│   ├── __init__.py
│   └── engine.py        # ChatTTS 单例引擎
├── routers/
│   ├── __init__.py
│   ├── tts.py           # TTS 路由
│   ├── speaker.py       # 音色管理路由
│   ├── novel.py         # 有声书路由
│   └── research.py      # 实验工具路由
├── schemas/
│   └── __init__.py      # Pydantic 模型
├── storage/
│   └── __init__.py      # 音色存储管理
└── utils/
    └── __init__.py      # 工具函数
```

## 已解决的坑

### 1. 维度冲突
ChatTTS DVAE 对维度敏感，提取 Embedding 时必须 `squeeze()` 处理。

```python
embedding = self.chat.dvae.sample_audio(waveform)
embedding = embedding.squeeze()  # 关键！
```

### 2. LZMA 损坏
严禁使用 `sample_audio_speaker` 返回的 LZMA 压缩字符串，所有 Embedding 以 `.pt` 格式保存：

```python
torch.save(embedding, f"{speaker_id}.pt")
```

### 3. 类型匹配
推理时强制转换为 `float32`：

```python
if spk_emb.dtype != torch.float32:
    spk_emb = spk_emb.to(torch.float32)
```

## 使用示例

### Python 客户端

```python
import requests

# 注册音色
with open("sample.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/speaker/register",
        files={"audio": f},
        data={"speaker_id": "my_voice", "description": "我的声音"}
    )
    print(resp.json())

# 文本转语音
resp = requests.post(
    "http://localhost:8000/tts/generate",
    json={
        "text": "你好，这是一段测试语音。",
        "speaker_id": "my_voice",
        "speed": 5
    }
)
with open("output.wav", "wb") as f:
    f.write(resp.content)

# 描述映射 (毕设核心)
resp = requests.post(
    "http://localhost:8000/speaker/predict_voice",
    json={"description": "沙哑的男声"}
)
print(resp.json())
```

### cURL

```bash
# 健康检查
curl http://localhost:8000/health

# 列出音色
curl http://localhost:8000/speaker/list

# t-SNE 可视化
curl http://localhost:8000/research/tsne
```

## 后续开发

### Adapter 模型集成

在 `routers/speaker.py` 的 `predict_voice` 接口中替换 Mock 实现：

```python
# 替换这行
embedding = engine.random_speaker_embedding()

# 为
embedding = adapter_model.predict(request.description)
```

### Agent 逻辑集成

在 `routers/novel.py` 的 `mock_analyze_sentence` 函数中替换为真正的 NLP 分析。
