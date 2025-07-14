# AutoSubtitleGenerator

[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

* 自动化字幕生成工具，能够为视频文件批量生成字幕

* 本工具使用VAD确定时间戳，调参之后，切分比直接使用`Whisper`精准很多。整合了一些模型，包括 **Demucs** 用于人声分离，**Whisper (large-v3)** 用于语音识别，以及 **DeepSeek API** 用于翻译，提供一个从视频到双语字幕的端到端的方案

---

## ⚠️WARNING
**项目还没完成,最后的main函数还没写**

## 📺 效果演示

<https://github.com/ienone/AutoSubtitleGenerator//assets/0139c793-3a5b-4735-95f7-c95b72a4c7c2>

未经手动处理修正别的字幕效果

## ✨ 项目特性

-   **高质量人声分离**: 使用 [Demucs](https://github.com/facebookresearch/demucs) 模型提取人声，最大程度减少背景音乐和音效的干扰
-   **精准语音识别**: 采用 OpenAI 的 [Whisper (large-v3)](https://github.com/openai/whisper) 模型
-   **VAD获得精准的时间戳**: 内置 [Silero-VAD](https://github.com/snakers4/silero-vad)，在识别前精确切分音频，避免AI在长静音段落产生“幻觉”文本，同时提高字幕时间准确性
-   **定制化翻译**: 用 [DeepSeek API](https://www.deepseek.com/) 和设计 Prompt 模板，实现契合特定动漫内容的翻译

## 📂 文件结构

目录布局大致如下，以确保输入、输出和中间文件的清晰分离，避免混淆

注意workspce文件夹部分是自动生成的，需要放进来的只有.ipynb文件和视频文件夹

```
/Your_Project_Root
├── release.ipynb    # Jupyter Notebook主文件
├── README.md                     # 本说明文件

├── GrandBlue/                      # ◀── 【输入】将你的视频文件夹放在这里，具体文件夹名在notebook中可以修改
│   ├── S02E01.mp4
│   ├── S02E02.mkv
│   └── ...

└── workspace/                      # ◀── 【工作区】所有自动生成的文件都在这里
    ├── output/                     # ◀── 【最终产物】字幕文件
    │   ├── S02E01_ja.srt           # 日语字幕
    │   ├── S02E01_zh.srt           # 中文字幕
    │   └── ...
    │
    └── temp/                       # 【中间文件】用于调试，可以安全删除
        ├── S02E01_full_audio.wav   # 提取出的高质量音轨
        ├── htdemucs/               # Demucs分离出的人声
        ├── S02E01_16k_mono.wav     # 简化的人声音频
        ├── S02E01_vad_timestamps_ms.json # VAD时间戳
        └── S02E01_vad_chunks/      # 切割后的音频片段文件夹
            └── ...
```

## 🚀 使用指南

1.  **克隆仓库**
    ```bash
    git clone https://github.com/ienone/AutoSubtitleGenerator.git
    cd AutoSubtitleGenerator

2.  **创建 Conda 环境 (推荐)**
    本项目依赖 `PyTorch` 和 `ffmpeg`，可以使用 Conda 管理这些依赖
    
    ```bash
    # 创建一个新的conda环境
    conda create -n anisub python=3.10 -y
    conda activate anisub
    
    # 安装 PyTorch, CUDA Toolkit 和 ffmpeg
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 ffmpeg -c pytorch -c nvidia 
    
    # 安装其他核心依赖
    conda install demucs requests pydub ipywidgets jupyter ipykernel
    
3.  **安装 Pip 包**
    在激活 `anisub` 环境后，安装剩余的 Python 包
    
    ```bash
    pip install openai-whisper
    pip install git+https://github.com/snakers4/silero-vad.git
    pip install librosa numpy matplotlib
    ```

## 🛠️ 使用方法

1.  **配置项目**
    
    *   打开 `release.ipynb`
    *   在 **模块一：全局配置与环境初始化** 中，找到 `3.2. API与模型配置` 部分
    *   **填入DeepSeek API 密钥**:
        ```python
        DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # 替换为你的真实密钥
        ```
    *   **务必修改 `PROMPT_TEMPLATE` 中的术语表和翻译风格指南(随便哪家大模型联网之后让他搜罗信息/自己拟稿)**
    *   （可选）根据需要调整 `VAD_PARAMS`, `ASR_PARAMS` 等参数，不过目前参数经过参数空间搜索，应该相对适合番剧字幕识别场景
2.  **准备视频文件**
    
    *   将需要处理的视频文件夹放入根目录下，将`INPUT_DIR`变量修改为文件夹名，注意文件夹名不能有空格等
3.  **运行程序**
    *   在 Jupyter Notebook 中，从上到下依次执行所有单元格
    *   程序将开始处理，日志输出中可以看到处理进度

4.  **获取字幕**
    
    *   处理完成后，进入 `workspace/output/` 文件夹
    *   您将找到与每个视频文件同名的 `_ja.srt` (日语) 和 `_zh.srt` (中文) 字幕文件

## 📜 用到的开源项目

*   **[Demucs](https://github.com/facebookresearch/demucs)** by Meta Research: for state-of-the-art music source separation.
*   **[Whisper](https://github.com/openai/whisper)** by OpenAI: for robust speech recognition.
*   **[Silero VAD](https://github.com/snakers4/silero-vad)** by Silero Team: for a simple and effective voice activity detector.
*   **[DeepSeek AI](https://www.deepseek.com/)**: for providing the powerful language model for translation.
*   **[PyTorch](https://pytorch.org/)**, **[FFmpeg](https://ffmpeg.org/)**

## 📄 许可证

本项目采用 [MIT License](LICENSE) 授权

