# Stephen's Tools v3.0

A comprehensive toolkit for audio transcription, OCR, PDF generation, and AI-powered chat. This project allows you to download audio from YouTube and SoundCloud, transcribe it using Whisper, perform OCR on images, combine images into PDFs, and have interactive chat sessions with AI models (OpenAI, LM Studio, or Ollama).

✅ Auto-detects `ffmpeg` from system PATH or common installation locations (Homebrew, apt, Windows local).

---

## 🚀 Features

### Audio & Transcription
- Download and convert YouTube videos to audio (mp3)
- Download audio from SoundCloud
- Transcribe audio files automatically using Whisper (GPU/CUDA support)
- Transcribe local audio files from your computer
- Combine multiple transcript files

### OCR & PDF Tools
- Perform OCR on images using Tesseract
- Extract text from HEIF/HEIC images
- Combine multiple images into a single PDF

### AI Chat & Summarization
- Interactive chat sessions with AI models
- Load transcript or summary files as context for chat
- Summarize long transcripts automatically (splits into chunks if needed)
- Flexible support for OpenAI Cloud, Ollama, or LM Studio

### System Integration
- Auto-detects `ffmpeg` from system PATH or common locations
- GPU acceleration support (NVIDIA CUDA) for faster transcription
- Cross-platform compatibility (Windows, macOS, Linux)

---

## ⚙️ Setup & Usage

```bash
# Clone the repository
git clone https://github.com/sxbsalesman/stephen_tools.git
cd stephen_tools

# (Optional but recommended) Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Create .env file in the root folder and add:
# OPENAI_API_KEY=your_openai_api_key_here

# Start the application
python src/main.py
```

Follow the interactive menu:

1️⃣ Download YouTube audio  
2️⃣ Download SoundCloud audio  
3️⃣ Transcribe audio file  
4️⃣ Transcribe local audio file  
5️⃣ Combine transcript files  
6️⃣ Chat session (with optional transcript context)  
7️⃣ Summarize transcript  
8️⃣ Delete files  
9️⃣ OCR image to text  
🔟 Combine images to PDF  
0️⃣ Exit

---

## 🎧 FFmpeg Setup

This project requires FFmpeg to extract audio from YouTube and SoundCloud.

### ✅ Auto-Detection

The application automatically detects `ffmpeg` from:
- System PATH
- `/opt/homebrew/bin/ffmpeg` (macOS Homebrew)
- `/usr/local/bin/ffmpeg` (macOS/Linux)
- `/usr/bin/ffmpeg` (Linux)
- `./ffmpeg/bin/ffmpeg.exe` (Windows local)

### Installation

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Linux:**
```bash
sudo apt install ffmpeg
```

**Windows:**
1. Download from https://ffmpeg.org/download.html
2. Extract and place `ffmpeg.exe` in `./ffmpeg/bin/` folder, or add to system PATH

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
1️⃣ Download FFmpeg for your OS from:  

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

## 📦 Required Packages

The project uses the following Python packages (see `requirements.txt`):

- **pillow-heif** - HEIF/HEIC image support
- **yt-dlp** - YouTube and SoundCloud audio downloading
- **openai** - OpenAI API client
- **openai-whisper** - Audio transcription
- **colorama** - Terminal color output
- **python-dotenv** - Environment variable management
- **requests** - HTTP requests
- **pytesseract** - OCR engine wrapper
- **Pillow** - Image processing
- **reportlab** - PDF generation

### Additional System Requirements

- **Tesseract OCR**: Required for OCR functionality
  - macOS: `brew install tesseract`
  - Ubuntu/Linux: `sudo apt install tesseract-ocr`
  - Windows: Download from https://github.com/tesseract-ocr/tesseract

## 📂 Project Structure

```
stephen_tools/
├── src/
│   └── main.py          # Main application
├── audio/               # YouTube downloads (auto-created)
├── soundcloud/          # SoundCloud downloads (auto-created)
├── transcripts/         # Transcription files (auto-created)
├── summaries/           # Summary files (auto-created)
├── combined_transcription/  # Combined transcripts (auto-created)
├── requirements.txt
├── .gitignore
├── .env                 # Optional: API keys
└── README.md
```

---

## 💡 Usage Tips

- **Virtual Environment**: Do **not** commit your `.venv` folder — it's included in `.gitignore`.
- **File Organization**: 
  - YouTube audio → `audio/`
  - SoundCloud audio → `soundcloud/`
  - Transcripts → `transcripts/`
  - Summaries → `summaries/`
  - Combined transcripts → `combined_transcription/`
- **FFmpeg**: The application will warn you if `ffmpeg` is not found and provide installation instructions.
- **GPU Acceleration**: For faster Whisper transcription, ensure CUDA-enabled PyTorch is installed (see GPU Support section).
- **API Keys**: For OpenAI Cloud, you can either:
  - Set `OPENAI_API_KEY` in a `.env` file
  - Enter it when prompted in the application
- **Local LLMs**: For offline AI chat, use Ollama or LM Studio (no API key required).

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

Copyright (c) 2025 sxbsalesman

This project uses open source libraries such as yt-dlp, openai-whisper, torch, and others.  
Please refer to their respective licenses for more information.
