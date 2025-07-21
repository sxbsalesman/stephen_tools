# YouTube Transcript and Summarizer Project ver 1.3.0

This project allows you to download YouTube audio, transcribe it using Whisper, and have interactive chat sessions with AI models (OpenAI, LM Studio, or Ollama).  

✅ The project is designed to use a local `ffmpeg` binary placed in your project folder — no global install or system PATH needed.

---

## 🚀 Features

- Download and convert YouTube videos to audio (mp3)
- Transcribe audio files automatically using Whisper
- Choose and load transcript or summary files as context for chat
- Flexible support for OpenAI cloud or local LLMs (Ollama, LM Studio)
- Uses local `ffmpeg` binary in your project folder

---

## ⚙️ Setup & Usage

```bash
# Clone the repository
git clone <your-repo-url>
cd your-repo

# (Optional but recommended) Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file in the root folder and add:
# OPENAI_API_KEY=your_openai_api_key_here

# Start the script
python main2.py
```

Follow the interactive menu:

1️⃣ Download a YouTube audio file  
2️⃣ Transcribe an audio file  
3️⃣ Start a chat session  
4️⃣ Exit

---

## 🎧 FFmpeg Setup

This project requires FFmpeg to extract audio from YouTube videos.

### ✅ Setup

1️⃣ Download FFmpeg for your OS from:  
https://ffmpeg.org/download.html

2️⃣ Unzip the package.

3️⃣ Move `ffmpeg.exe` (Windows) or the `ffmpeg` binary (macOS/Linux) into:

```
your-repo/
├── ffmpeg/
│   └── bin/
│       └── ffmpeg.exe (or ffmpeg binary)
```

⚠️ **Important:** The script expects `ffmpeg` at `./ffmpeg/bin/ffmpeg.exe` by default.

---

## 🤖 Local LLM Support (Optional)

You can run the chat feature locally using **Ollama** or **LM Studio**.

### Ollama

```bash
# Install Ollama from https://ollama.com
ollama pull llama3
# Start Ollama; it will serve models locally automatically
```

### LM Studio

- Download from [lmstudio.ai](https://lmstudio.ai/)
- Download a compatible model (e.g., LLaMA 3 or Mistral)
- Start the LM Studio server before running this project

---

## ⚡️ GPU Support (NVIDIA CUDA)

- If you have an NVIDIA GPU and want to use CUDA acceleration for Whisper and PyTorch:
    1. Install the correct CUDA-enabled version of PyTorch. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and use the command for your CUDA version (e.g., CUDA 12.1).
    2. Example for CUDA 12.1:
        ```
        pip install torch --index-url https://download.pytorch.org/whl/cu121
        ```
    3. If you only install `torch` with `pip install torch`, you may get the CPU-only version and CUDA will not work.
    4. After installing, test with:
        ```python
        import torch
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        ```
    5. If you see `True` and your GPU name, CUDA is working!

- If you do **not** have an NVIDIA GPU, the app will automatically use CPU.

---

## 📝 Whisper Compatibility

- For best compatibility on Windows, use the `openai-whisper` package (not the original `whisper`).
- In your requirements.txt, use:
    ```
    openai-whisper
    ```
- Advanced users can try `whisper-cpp` for faster performance.

---

## 📂 Recommended Project Structure

```
your-repo/
├── ffmpeg/
│   └── bin/
│       └── ffmpeg.exe
├── audio/
├── transcripts/
├── summaries/
├── main2.py
├── requirements.txt
├── .gitignore
├── .env
├── README.md
```

---

## ✅ Notes

- Do **not** commit your `.venv` folder — include it in `.gitignore`.
- By default, transcripts are saved to `transcripts/` and audio files to `audio/`.
- The script will raise an error if `ffmpeg.exe` is missing in the expected folder.

---

## ⚠️ Legal & Ethical Notice

This tool allows downloading audio from YouTube and SoundCloud for personal, educational, or research purposes only.

- Downloading content from YouTube or SoundCloud may violate their Terms of Service unless the content is explicitly provided for download by the platform or the copyright owner.
- By using this tool, you agree to comply with the terms of the respective platforms and all applicable copyright laws.
- The developer is not responsible for any misuse of this tool. Please respect content creators’ rights and only download content you have permission to use.

---

## License

This project is licensed under the MIT License.  
See the LICENSE file for details.

Copyright (c) 2025 Stephen [Your Last Name or GitHub Username]

This project uses open source libraries such as yt-dlp, openai-whisper, torch, and others.  
Please refer to their respective licenses for more information.
