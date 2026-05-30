import pytesseract # type: ignore
from PIL import Image, ImageEnhance, ImageFilter, ImageOps # type: ignore
import glob
import os
import time
import json
import requests # type: ignore
import re
import shutil
import platform
import base64
import gc
import sys
import math
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp # type: ignore
import whisper # type: ignore
import torch # type: ignore
import openai # type: ignore
import subprocess
from colorama import Fore, Style, init # type: ignore

# Debug toggles via environment
DEBUG_OCR = os.environ.get("DEBUG_OCR", "").lower() in ("1", "true", "yes")

def _get_tesseract_max_side() -> int:
    """Return max long-side pixels for Tesseract preprocessing from env or default."""
    default_max_side = 2800
    raw_value = os.environ.get("OCR_TESSERACT_MAX_SIDE", "").strip()
    if not raw_value:
        return default_max_side
    try:
        parsed = int(raw_value)
        if parsed < 512:
            return default_max_side
        return parsed
    except ValueError:
        return default_max_side

OCR_TESSERACT_MAX_SIDE = _get_tesseract_max_side()


def _get_int_env(var_name: str, default: int, min_value: int) -> int:
    """Return an integer env var with lower-bound validation."""
    raw_value = os.environ.get(var_name, "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
        return parsed if parsed >= min_value else default
    except ValueError:
        return default


OCR_TILE_TRIGGER_LONG_SIDE = _get_int_env("OCR_TILE_TRIGGER_LONG_SIDE", 4200, 1024)
OCR_TILE_TRIGGER_PIXELS = _get_int_env("OCR_TILE_TRIGGER_PIXELS", 12_000_000, 1_000_000)
OCR_TILE_TARGET_SIDE = _get_int_env("OCR_TILE_TARGET_SIDE", 1800, 768)
OCR_TILE_MAX_TILES = _get_int_env("OCR_TILE_MAX_TILES", 24, 4)
OCR_TILE_OVERLAP_PX = _get_int_env("OCR_TILE_OVERLAP_PX", 160, 0)

# ---------------------------------------------------------------------------
#  Reusable HTTP session (connection pooling = faster sequential requests)
# ---------------------------------------------------------------------------
_http_session = requests.Session()
_http_session.headers.update({"Connection": "keep-alive"})

# ---------------------------------------------------------------------------
#  Cached Whisper model (avoids reloading ~500MB on every transcription)
# ---------------------------------------------------------------------------
_whisper_model = None

def _get_whisper_model():
    """Load Whisper model once and cache it for subsequent calls."""
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(Fore.CYAN + f"Loading Whisper model on {device} (one-time)..." + Style.RESET_ALL)
        _whisper_model = whisper.load_model("base", device=device)
    return _whisper_model

# ---------------------------------------------------------------------------
#  Image file discovery helper (replaces 5+ duplicated glob blocks)
# ---------------------------------------------------------------------------
_IMAGE_EXTENSIONS = (
    '*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG',
    '*.bmp', '*.BMP', '*.heic', '*.HEIC', '*.heif', '*.HEIF',
)

def find_image_files(directory: str) -> list:
    """Return a sorted list of image file paths in *directory*, excluding already-converted files."""
    files = []
    for ext in _IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, ext)))
    files.sort()
    # Filter out _converted.png files that were generated from HEIC/HEIF originals
    # (they don't need separate processing as we already process the originals)
    filtered = [f for f in files if not f.endswith('_converted.png')]
    return filtered

# ---------------------------------------------------------------------------
#  Tesseract fallback helper (replaces 3 duplicated blocks)
# ---------------------------------------------------------------------------
def _tesseract_ocr(img) -> str:
    """Run Tesseract with minimal PSM settings and return the best result. Fast timeout to avoid hangs."""
    configs = [
        "--psm 6 --oem 3 -l eng -c preserve_interword_spaces=1",
        "--psm 4 --oem 3 -l eng -c preserve_interword_spaces=1",
    ]
    candidates = []
    for cfg in configs:
        try:
            # Use a 15 second timeout per PSM config - if Tesseract doesn't respond, move on
            text = pytesseract.image_to_string(img, config=cfg, timeout=15)
            if text.strip():  # Only keep non-empty results
                candidates.append((len(text.strip()), text))
        except Exception as e:
            if "timeout" in str(e).lower():
                print(Fore.YELLOW + f"  Tesseract timeout on PSM config (skipping)" + Style.RESET_ALL)
            continue
    return max(candidates, key=lambda x: x[0])[1] if candidates else ""

# ---------------------------------------------------------------------------
#  Preprocess an image for Tesseract (only called when Tesseract is actually needed)
# ---------------------------------------------------------------------------
def _preprocess_for_tesseract(img, source_name: str = ""):
    """Convert to grayscale, constrain very large images, upscale small images, enhance, and auto-orient."""
    processed = img.convert('L')

    original_w, original_h = processed.size

    # Downscale very large images to avoid Tesseract timeouts/memory spikes
    max_side_for_tesseract = OCR_TESSERACT_MAX_SIDE
    if max(original_w, original_h) > max_side_for_tesseract:
        downscale = max_side_for_tesseract / float(max(original_w, original_h))
        processed = processed.resize(
            (int(original_w * downscale), int(original_h * downscale)),
            Image.Resampling.LANCZOS
        )

    # Upscale small images for better OCR accuracy
    w, h = processed.size
    if w < 1800 or h < 1800:
        scale = max(1800 / w, 1800 / h)
        processed = processed.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    # Enhance contrast & sharpness
    processed = ImageEnhance.Contrast(processed).enhance(2.5)
    processed = ImageEnhance.Sharpness(processed).enhance(2.0)
    processed = ImageOps.autocontrast(processed, cutoff=2)

    # Try to correct orientation via Tesseract OSD
    try:
        osd = pytesseract.image_to_osd(processed)
        match = re.search(r"Rotate:\s*(\d+)", osd)
        if match:
            angle = int(match.group(1))
            if angle and angle % 360 != 0:
                processed = processed.rotate(360 - angle, expand=True)
    except Exception:
        pass

    if DEBUG_OCR:
        final_w, final_h = processed.size
        label = source_name if source_name else "image"
        print(
            Fore.CYAN
            + f"  [DEBUG] Tesseract preprocess size for {label}: {original_w}x{original_h} -> {final_w}x{final_h} (max_side={max_side_for_tesseract})"
            + Style.RESET_ALL
        )

    return processed


def _edge_density_score(img) -> float:
    """Estimate text/line density using edge coverage on a downsampled grayscale image."""
    try:
        probe = img.convert('L')
        max_probe_side = 1400
        w, h = probe.size
        if max(w, h) > max_probe_side:
            scale = max_probe_side / float(max(w, h))
            probe = probe.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)

        edges = probe.filter(ImageFilter.FIND_EDGES)
        # Fixed threshold keeps this fast and deterministic.
        binary = edges.point(lambda p: 255 if p >= 24 else 0)
        hist = binary.histogram()
        edge_pixels = hist[255] if len(hist) > 255 else 0
        total_pixels = probe.size[0] * probe.size[1]
        return (edge_pixels / total_pixels) if total_pixels else 0.0
    except Exception:
        return 0.0


def _should_tile_for_ocr(img, source_name: str = "") -> tuple[bool, str]:
    """Decide whether OCR should run per tile for large, dense document images."""
    width, height = img.size
    area = width * height
    long_side = max(width, height)
    edge_density = _edge_density_score(img)

    oversized = long_side >= OCR_TILE_TRIGGER_LONG_SIDE or area >= OCR_TILE_TRIGGER_PIXELS
    dense_document = edge_density >= 0.035

    should_tile = oversized or (dense_document and area >= 6_000_000)
    reason = (
        f"{width}x{height}, area={area:,}, edge_density={edge_density:.3f}, "
        f"oversized={oversized}, dense={dense_document}"
    )

    if DEBUG_OCR:
        label = source_name if source_name else "image"
        print(Fore.CYAN + f"  [DEBUG] Tile decision for {label}: {reason}" + Style.RESET_ALL)

    return should_tile, reason


def _make_ocr_tiles(img) -> list:
    """Split an image into overlapping tiles in reading order (top-left to bottom-right)."""
    width, height = img.size
    cols = max(1, math.ceil(width / float(OCR_TILE_TARGET_SIDE)))
    rows = max(1, math.ceil(height / float(OCR_TILE_TARGET_SIDE)))

    # Keep tile count bounded for extremely large scans.
    while rows * cols > OCR_TILE_MAX_TILES:
        if cols >= rows and cols > 1:
            cols -= 1
        elif rows > 1:
            rows -= 1
        else:
            break

    tile_w = max(1, math.ceil(width / float(cols)))
    tile_h = max(1, math.ceil(height / float(rows)))
    overlap_x = min(OCR_TILE_OVERLAP_PX, max(0, tile_w // 3))
    overlap_y = min(OCR_TILE_OVERLAP_PX, max(0, tile_h // 3))

    tiles = []
    for row in range(rows):
        for col in range(cols):
            left = max(0, col * tile_w - (overlap_x if col > 0 else 0))
            top = max(0, row * tile_h - (overlap_y if row > 0 else 0))
            right = min(width, (col + 1) * tile_w + (overlap_x if col + 1 < cols else 0))
            bottom = min(height, (row + 1) * tile_h + (overlap_y if row + 1 < rows else 0))
            tiles.append((row + 1, col + 1, img.crop((left, top, right, bottom))))
    return tiles


def _tesseract_tiled_ocr(img, source_name: str = "") -> str:
    """Run Tesseract on generated tiles and merge OCR text in reading order."""
    merged_parts = []
    tiles = _make_ocr_tiles(img)
    total_tiles = len(tiles)
    label = source_name if source_name else "image"

    for idx, (row, col, tile) in enumerate(tiles, 1):
        tile_name = f"{label} tile {idx}/{total_tiles} (r{row}c{col})"
        preprocessed = _preprocess_for_tesseract(tile, tile_name)
        tile_text = _tesseract_ocr(preprocessed).strip()

        if tile_text:
            if DEBUG_OCR:
                merged_parts.append(f"\n[Tile r{row}c{col}]\n{tile_text}")
            else:
                merged_parts.append(tile_text)

    return "\n\n".join(merged_parts).strip()


def _run_tesseract_with_auto_tiling(img, source_name: str = "") -> str:
    """Use tiled OCR for oversized/dense images, otherwise run single-image OCR."""
    should_tile, reason = _should_tile_for_ocr(img, source_name)
    if should_tile:
        print(Fore.CYAN + f"  Using tiled OCR ({reason})" + Style.RESET_ALL)
        tiled_text = _tesseract_tiled_ocr(img, source_name)
        if tiled_text.strip():
            return tiled_text
        print(Fore.YELLOW + "  Tiled OCR produced no text; retrying as a single image." + Style.RESET_ALL)

    return _tesseract_ocr(_preprocess_for_tesseract(img, source_name))

# ---------------------------------------------------------------------------
#  Image-to-base64 helper (moved out of per-image loop)
# ---------------------------------------------------------------------------
def img_to_b64(source_img, max_side=1280, quality=85) -> str:
    """Compress and base64-encode a PIL Image for multimodal API calls."""
    buf = BytesIO()
    w, h = source_img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        img_small = source_img.convert('RGB').resize(new_size, Image.Resampling.LANCZOS)
    else:
        img_small = source_img.convert('RGB')
    img_small.save(buf, format='JPEG', quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ---------------------------------------------------------------------------
#  Vision-model detection helper
# ---------------------------------------------------------------------------
_VISION_KEYWORDS = ('vl', 'vision', 'qwen', 'llava', 'olmocr', 'ocr')

def is_vision_model(name: str) -> bool:
    """Return True if *name* looks like a vision-capable model."""
    lower = name.lower()
    return any(kw in lower for kw in _VISION_KEYWORDS)

# Auto-detect ffmpeg location based on OS
def get_ffmpeg_path():
    """
    Auto-detects ffmpeg location. Works on Ubuntu, macOS, and Windows.
    """
    # First, check if ffmpeg is in system PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    
    # Fallback to OS-specific locations
    if platform.system() == "Darwin":  # macOS
        if os.path.exists("/opt/homebrew/bin/ffmpeg"):
            return "/opt/homebrew/bin/ffmpeg"
        elif os.path.exists("/usr/local/bin/ffmpeg"):
            return "/usr/local/bin/ffmpeg"
    elif platform.system() == "Linux":  # Ubuntu/Linux
        if os.path.exists("/usr/bin/ffmpeg"):
            return "/usr/bin/ffmpeg"
    elif platform.system() == "Windows":
        # Check local project folder first
        local_path = os.path.join(os.path.dirname(__file__), "..", "ffmpeg", "bin", "ffmpeg.exe")
        if os.path.exists(local_path):
            return local_path
    
    return None

ffmpeg_path = get_ffmpeg_path()
if ffmpeg_path:
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
    print(f"Using ffmpeg at: {ffmpeg_path}")
else:
    print("⚠️  FFmpeg not found! Please install it:")
    print("   Ubuntu/Linux: sudo apt install ffmpeg")
    print("   macOS: brew install ffmpeg")
    print("   Windows: Download from https://ffmpeg.org/download.html")

init(autoreset=True)

def select_backend_and_model():
    """
    Auto-selects backend/model without prompting:
    - Prefer LM Studio if running with a loaded model
    - Else prefer Ollama if running with an available model
    - Else use OpenAI Cloud (if OPENAI_API_KEY is present)
    Returns the base_url, api_key, backend name, and detected default model name.
    """
    # Detect running local backends to set a smarter default and inform user
    def _is_up(url: str) -> bool:
        try:
            r = requests.get(url, timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    ollama_up = _is_up("http://localhost:11434/v1/models")
    lmstudio_up = _is_up("http://localhost:1234/v1/models")
    # Also check common olmocr / alternative LM Studio ports
    # (LM Studio defaults to 1234 but users can change it)
    openai_key = os.environ.get("OPENAI_API_KEY")

    print("\nBackend status:")
    print(f" - Ollama (11434): {'running' if ollama_up else 'not detected'}")
    print(f" - LM Studio (1234): {'running' if lmstudio_up else 'not detected'}")
    print(f" - OpenAI Cloud: {'API key detected' if openai_key else 'no API key'}")

    if lmstudio_up:
        base_url = "http://localhost:1234/v1"
        api_key = "sk-local"
        backend = "LM Studio"
        # Check which models are loaded in LM Studio
        model_name = "local-model"  # Default fallback
        try:
            response = _http_session.get("http://localhost:1234/v1/models", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                loaded_models = models_data.get('data', [])
                if loaded_models:
                    model_name = loaded_models[0].get('id', 'local-model')
            else:
                print(Fore.YELLOW + "LM Studio is running but model list could not be read; using fallback model name." + Style.RESET_ALL)
        except Exception:
            print(Fore.YELLOW + "LM Studio is running but model detection failed; using fallback model name." + Style.RESET_ALL)

    elif ollama_up:
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"
        backend = "Ollama"
        # Check which models are available in Ollama
        try:
            response = _http_session.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available = [m.get('name') for m in models_data.get('models', [])]
                if available:
                    model_name = available[0]  # Use first available model as default
                else:
                    model_name = "llava:7b"  # Fallback if no models found
            else:
                model_name = "llava:7b"  # Fallback
        except Exception:
            model_name = "llava:7b"  # Fallback if Ollama not running yet

    else:
        base_url = None
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(Fore.RED + "No local backend detected and OPENAI_API_KEY is not set." + Style.RESET_ALL)
        backend = "OpenAI Cloud"
        model_name = "gpt-4o"

    print(f"\n{Fore.GREEN}Auto-selected backend: {backend}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Auto-selected model: {model_name}{Style.RESET_ALL}")

   ## device = "cuda" if torch.cuda.is_available() else "cpu"
   ## model_name = whisper.load_model("base", device=device)
   ## print('Are you are able to the Nvidia Cuda support?', torch.cuda.is_available())
   ## print(torch.cuda.get_device_name(0))

    return base_url, api_key, backend, model_name

def _strip_playlist_params(url: str) -> str:
    """Remove playlist/radio query params so yt-dlp treats the URL as a single video."""
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    # Keep only the video id; drop list, start_radio, index, etc.
    cleaned = {k: v for k, v in params.items() if k in ('v',)}
    return urlunparse(parsed._replace(query=urlencode(cleaned, doseq=True)))


def download_audio(youtube_url, output_path="audio"):
    """
    Downloads audio from a YouTube video using yt_dlp and saves it as an mp3 file
    in the specified output directory. Uses the local ffmpeg binary.
    Returns the path to the downloaded audio file.
    """
    # Strip playlist/radio params to avoid downloading entire playlists
    youtube_url = _strip_playlist_params(youtube_url)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'ffmpeg_location': ffmpeg_path,
        'noplaylist': True,           # Never download playlists
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)
        audio_file = os.path.splitext(filename)[0] + '.mp3'
        return audio_file

def download_soundcloud_audio(soundcloud_url, output_path="soundcloud"):
    """
    Downloads audio from a SoundCloud URL using yt_dlp and saves it as an mp3 file
    in the specified output directory.
    Returns the path to the downloaded audio file.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'ffmpeg_location': ffmpeg_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(soundcloud_url, download=True)
        filename = ydl.prepare_filename(info)
        audio_file = os.path.splitext(filename)[0] + '.mp3'
        return audio_file

def transcribe_audio(audio_file, transcript_dir="transcripts"):
    """
    Transcribes the given audio file using Whisper and saves the transcript as a text file
    in the transcripts directory. Returns the path to the transcript file.
    Uses cached Whisper model to avoid reloading on every call.
    """
    transcript_path = os.path.join(os.path.dirname(__file__), transcript_dir)
    if not os.path.exists(transcript_path):
        os.makedirs(transcript_path)
    model = _get_whisper_model()
    result = model.transcribe(audio_file)
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    transcript_file = os.path.join(transcript_path, f"{base_name}.txt")
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    return transcript_file

def list_audio_files(audio_dir="audio"):
    """
    Lists all audio files (.mp3, .wav, .m4a) in the specified audio directory.
    Returns a list of filenames.
    """
    if not os.path.exists(audio_dir):
        print(Fore.RED + "No audio directory found." + Style.RESET_ALL)
        return []
    return [f for f in os.listdir(audio_dir) if f.lower().endswith(('.mp3', '.wav', '.m4a'))]

def list_transcript_files(transcript_dir="transcripts"):
    """
    Lists all transcript text files (.txt) in the transcripts directory.
    Returns a list of filenames.
    """
    transcript_path = os.path.join(os.path.dirname(__file__), transcript_dir)
    if not os.path.exists(transcript_path):
        print(Fore.RED + "No transcripts directory found." + Style.RESET_ALL)
        return []
    return [f for f in os.listdir(transcript_path) if f.lower().endswith('.txt')]

def chat_session(model_name, backend):
    """
    Starts an interactive chat session with the selected LLM model.
    Allows the user to optionally provide a system prompt and/or transcript file as context.
    """
    print(Fore.MAGENTA + "Starting chat session. Type 'exit' to quit." + Style.RESET_ALL)

    system_prompt = input(Fore.YELLOW + "Enter a system prompt (or press Enter to skip): " + Style.RESET_ALL).strip()

    # Prompt to include a transcript file as context
    add_file_context = input(Fore.YELLOW + "Do you want to include a text file for context? (y/n): " + Style.RESET_ALL).strip().lower()
    file_content = ""
    if add_file_context == "y":
        transcript_files = list_transcript_files()
        if transcript_files:
            print("\nAvailable transcript files:")
            for idx, fname in enumerate(transcript_files, 1):
                print(f"{idx}. {fname}")
            file_choice = input(Fore.YELLOW + "Select a file number to include (or 'q' to skip): " + Style.RESET_ALL).strip()
            if file_choice.lower() != "q" and file_choice != "":
                try:
                    file_idx = int(file_choice) - 1
                    if 0 <= file_idx < len(transcript_files):
                        transcript_file = os.path.join(os.path.dirname(__file__), "transcripts", transcript_files[file_idx])
                        with open(transcript_file, "r", encoding="utf-8") as f:
                            file_content = f.read()
                        print(Fore.GREEN + f"Loaded file: {transcript_file}" + Style.RESET_ALL)
                    else:
                        print(Fore.RED + "Invalid selection." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
        else:
            print(Fore.RED + "No transcript files found." + Style.RESET_ALL)

    messages = []
    # Instead of using "system", prepend as a user message for local LLMs
    if backend in ["LM Studio", "Ollama"]: # type: ignore
        context_message = ""
        if system_prompt:
            context_message += system_prompt + "\n"
        if file_content:
            context_message += f"Here is the file content for context:\n\n{file_content}\n"
        if context_message:
            messages.append({"role": "user", "content": context_message})
    else:
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if file_content:
            messages.append({"role": "system", "content": f"Here is the file content for context:\n\n{file_content}"})

    while True:
        user_input = input(Fore.YELLOW + "You: " + Style.RESET_ALL)
        if user_input.lower() == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        bot_message = response.choices[0].message.content
        print(Fore.GREEN + "Bot: " + bot_message + Style.RESET_ALL)
        messages.append({"role": "assistant", "content": bot_message})

def summarize_text(text, model_name):
    """
    Prompts the user for a summary format, sends the text and instruction to the LLM,
    and returns the summary. Allows user to quit and return to main menu.
    """
    print("\nChoose a summary format (or enter 'q' to return to main menu):")
    print("1. Bullet points")
    print("2. Numbered list")
    print("3. Single paragraph")
    print("4. JSON object")
    format_choice = input("Enter your choice (1/2/3/4 or q): ").strip()
    if format_choice.lower() == "q" or format_choice == "":
        print("Returning to main menu.")
        return None

    if format_choice == "1":
        instruction = "Summarize the following text as concise bullet points:"
    elif format_choice == "2":
        instruction = "Summarize the following text as a numbered list:"
    elif format_choice == "3":
        instruction = "Summarize the following text in a single paragraph:"
    elif format_choice == "4":
        instruction = "Summarize the following text and return the result as a JSON object with keys 'main_points' and 'action_items':"
    else:
        print("Invalid choice. Returning to main menu.")
        return None

    chunks = split_text(text, max_words=3000)
    summaries = []

    for i, chunk in enumerate(chunks, 1):
        print(f"\nSummarizing chunk {i}/{len(chunks)}...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": f"{instruction}\n\n{chunk}"}
            ]
        )
        summaries.append(response.choices[0].message.content)

    final_summary = "\n\n".join(summaries)
    return final_summary

def delete_file_menu():
    """
    Allows the user to delete files from either the audio or transcripts folder.
    User can select the folder, see a list of files, and confirm deletion.
    """
    while True:
        print(Fore.CYAN + "\nDelete a file from which folder?" + Style.RESET_ALL)
        print("1. audio")
        print("2. transcripts")
        print("q. Return to main menu")
        folder_choice = input(Fore.YELLOW + "Enter your choice (1/2/q): " + Style.RESET_ALL).strip().lower()
        if folder_choice == "q" or folder_choice == "":
            print("Returning to main menu.")
            return
        if folder_choice == "1":
            folder = "audio"
            files = list_audio_files()
        elif folder_choice == "2":
            folder = "transcripts"
            files = list_transcript_files()
        else:
            print(Fore.RED + "Invalid choice." + Style.RESET_ALL)
            continue

        if not files:
            print(Fore.RED + f"No files found in {folder} folder." + Style.RESET_ALL)
            continue

        print(f"\nAvailable files in {folder}:")
        for idx, fname in enumerate(files, 1):
            print(f"{idx}. {fname}")
        print("q. Return to previous menu")
        file_choice = input(Fore.YELLOW + "Select a file number to delete (or 'q' to return): " + Style.RESET_ALL).strip().lower()
        if file_choice == "q" or file_choice == "":
            print("Returning to previous menu.")
            continue
        try:
            file_idx = int(file_choice) - 1
            if 0 <= file_idx < len(files):
                file_path = os.path.join(os.path.dirname(__file__), folder, files[file_idx])
                confirm = input(Fore.RED + f"Are you sure you want to delete '{files[file_idx]}'? (y/n): " + Style.RESET_ALL).strip().lower()
                if confirm == "y":
                    os.remove(file_path)
                    print(Fore.GREEN + f"Deleted: {file_path}" + Style.RESET_ALL)
                else:
                    print("Delete cancelled.")
            else:
                print(Fore.RED + "Invalid selection." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)

def combine_transcript_files():
    """
    Allows the user to select multiple transcript files and combine them into a single file.
    The combined file is saved in the combined_transcription folder with a user-specified name.
    """
    transcript_files = list_transcript_files()
    if not transcript_files:
        print(Fore.RED + "No transcript files found in the transcripts directory." + Style.RESET_ALL)
        return

    print("\nAvailable transcript files:")
    for idx, fname in enumerate(transcript_files, 1):
        print(f"{idx}. {fname}")
    print("Enter the numbers of the files to combine, separated by commas (e.g., 1,2,4), or 'q' to return to main menu.")
    file_choices = input(Fore.YELLOW + "Your selection: " + Style.RESET_ALL).strip()
    if file_choices.lower() == "q" or file_choices == "":
        print("Returning to main menu.")
        return

    try:
        indices = [int(num.strip()) - 1 for num in file_choices.split(",") if num.strip().isdigit()]
        selected_files = [transcript_files[i] for i in indices if 0 <= i < len(transcript_files)]
        if not selected_files:
            print(Fore.RED + "No valid files selected." + Style.RESET_ALL)
            return
    except Exception as e:
        print(Fore.RED + f"Invalid input: {e}" + Style.RESET_ALL)
        return

    combined_text = ""
    for fname in selected_files:
        file_path = os.path.join(os.path.dirname(__file__), "transcripts", fname)
        with open(file_path, "r", encoding="utf-8") as f:
            combined_text += f"\n--- {fname} ---\n"
            combined_text += f.read() + "\n"

    combined_dir = os.path.join(os.path.dirname(__file__), "combined_transcription")
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    output_name = input(Fore.YELLOW + "Enter a name for the combined file (without extension): " + Style.RESET_ALL).strip()
    if not output_name:
        print("No name entered. Returning to main menu.")
        return

    output_path = os.path.join(combined_dir, f"{output_name}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined_text)
    print(Fore.GREEN + f"Combined file saved to: {output_path}" + Style.RESET_ALL)

def split_text(text, max_words=3000):
    """
    Splits the input text into chunks, each with up to max_words words.
    Returns a list of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def check_server_running(base_url, backend):
    """
    Checks if the LM Studio or Ollama server is running by sending a GET request to /models endpoint.
    For LM Studio, also verifies that at least one model is loaded.
    Returns True if running (and model loaded for LM Studio), False otherwise.
    """
    try:
        # base_url already includes /v1 for both LM Studio and Ollama
        endpoint = f"{base_url}/models"
        response = _http_session.get(endpoint, timeout=5)
        if response.status_code == 200:
            # For LM Studio, check if any models are loaded
            if backend == "LM Studio":
                try:
                    models_data = response.json()
                    loaded_models = models_data.get('data', [])
                    if not loaded_models:
                        print(Fore.RED + "\n" + "="*60 + Style.RESET_ALL)
                        print(Fore.RED + "LM Studio: No models loaded!" + Style.RESET_ALL)
                        print(Fore.RED + "="*60 + Style.RESET_ALL)
                        print(Fore.YELLOW + "\nTo fix this, please load a model in LM Studio:" + Style.RESET_ALL)
                        print(Fore.CYAN + "  Option 1: Use LM Studio GUI" + Style.RESET_ALL)
                        print("    • Open LM Studio → Go to 'Local Server' tab")
                        print("    • Click 'Select a model to load' dropdown")
                        print("    • Choose and load a model (e.g., llama, mistral, phi)")
                        print(Fore.CYAN + "\n  Option 2: Use terminal command" + Style.RESET_ALL)
                        print("    • Run: lms load <model-name>")
                        print("    • Example: lms load llama-3.2-1b-instruct")
                        print(Fore.CYAN + "\n  To see available models:" + Style.RESET_ALL)
                        print("    • Run: lms ls")
                        print(Fore.RED + "\n" + "="*60 + Style.RESET_ALL)
                        return False
                    else:
                        # Show which model is loaded
                        model_ids = [m.get('id', 'unknown') for m in loaded_models]
                        print(Fore.GREEN + f"LM Studio model(s) loaded: {', '.join(model_ids)}" + Style.RESET_ALL)
                except Exception as parse_err:
                    print(Fore.YELLOW + f"Warning: Could not parse LM Studio models response: {parse_err}" + Style.RESET_ALL)
            return True
        else:
            print(Fore.RED + f"{backend} server responded with status code {response.status_code}." + Style.RESET_ALL)
            return False
    except Exception as e:
        print(Fore.RED + f"Could not connect to {backend} server at {base_url}: {e}" + Style.RESET_ALL)
        return False

def print_gpu_status(context: str = ""):
    """
    Prints a concise GPU status readout (utilization, VRAM, temperature) if available.
    Tries NVIDIA via nvidia-smi first, then PyTorch CUDA info as a fallback.
    """
    header = f"GPU Status{': ' + context if context else ''}"
    print(Fore.CYAN + f"\n{header}" + Style.RESET_ALL)
    try:
        import shutil as _shutil
        if _shutil.which("nvidia-smi"):
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader"
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if proc.returncode == 0 and proc.stdout.strip():
                for line in proc.stdout.strip().splitlines():
                    # Example: NVIDIA GeForce RTX 3050 Laptop GPU, 12 %, 1024 MiB, 6144 MiB, 58 C
                    print(Fore.GREEN + f"  {line}" + Style.RESET_ALL)
                return
    except Exception:
        pass

    # Fallback to PyTorch CUDA info
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            used = torch.cuda.memory_allocated(0) / (1024**2)
            print(Fore.GREEN + f"  {name}: used {used:.0f} MiB / {total:.0f} MiB" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "  No CUDA GPU detected or drivers not available." + Style.RESET_ALL)
    except Exception:
        print(Fore.YELLOW + "  GPU metrics unavailable (no nvidia-smi and PyTorch not reporting)." + Style.RESET_ALL)


def cleanup_vram(context: str = "", aggressive: bool = False, verbose: bool = True):
    """
    Release reclaimable Python/CUDA memory without unloading models.
    Safe for LM Studio/Ollama workflows where the user manages loaded models.
    """
    if verbose:
        label = f" ({context})" if context else ""
        print(Fore.CYAN + f"\nVRAM cleanup{label}..." + Style.RESET_ALL)

    # Reclaim Python objects first.
    gc.collect()

    if torch.cuda.is_available():
        try:
            if aggressive:
                # Ensure pending kernels complete before cache trim.
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if aggressive and hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

            if verbose:
                allocated = torch.cuda.memory_allocated(0) / (1024**2)
                reserved = torch.cuda.memory_reserved(0) / (1024**2)
                print(Fore.GREEN + f"  CUDA cache trimmed: allocated={allocated:.0f} MiB, reserved={reserved:.0f} MiB" + Style.RESET_ALL)
        except Exception as e:
            if verbose:
                print(Fore.YELLOW + f"  CUDA cleanup warning: {e}" + Style.RESET_ALL)
    elif verbose:
        print(Fore.YELLOW + "  No CUDA device detected; ran Python GC only." + Style.RESET_ALL)

def ensure_single_llm_running(backend: str, model_name: str):
    """
    Ensures only the selected LLM is running.
    - For Ollama: stops any other running models via `ollama stop <model>`.
    - For LM Studio/OpenAI: prints guidance (no safe API to stop server).
    """
    try:
        if backend == "Ollama":
            ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
            running_models = []
            for line in ps.stdout.splitlines():
                line = line.strip()
                if not line or line.lower().startswith("name"):
                    continue
                # First column is the model name
                name = line.split()[0]
                if name:
                    running_models.append(name)
            to_stop = [m for m in running_models if m != model_name]
            for m in to_stop:
                subprocess.run(["ollama", "stop", m], capture_output=True, text=True)
            if to_stop:
                print(Fore.YELLOW + f"Stopped other Ollama models: {', '.join(to_stop)}" + Style.RESET_ALL)
        elif backend == "LM Studio":
            print(Fore.YELLOW + "LM Studio server cannot be programmatically stopped here; ensure only one model is serving in LM Studio." + Style.RESET_ALL)
        else:
            # OpenAI Cloud: nothing to stop locally
            pass
    except Exception as e:
        print(Fore.YELLOW + f"Could not enforce single LLM running: {e}" + Style.RESET_ALL)

def transcribe_local_audio():
    """
    Allows the user to select an audio file from the 'local' folder and transcribes it,
    saving the transcript in the 'transcripts' folder.
    """
    local_dir = os.path.join(os.path.dirname(__file__), "local")
    if not os.path.exists(local_dir):
        print(Fore.RED + "No 'local' directory found." + Style.RESET_ALL)
        return
    audio_files = [f for f in os.listdir(local_dir) if f.lower().endswith(('.mp3', '.wav', '.m4a'))]
    if not audio_files:
        print("No audio files found in the 'local' directory.")
        return
    print("\nAvailable local audio files:")
    for idx, fname in enumerate(audio_files, 1):
        print(f"{idx}. {fname}")
    file_choice = input("Select a file number to transcribe (or 'q' to return): ").strip()
    if file_choice.lower() == "q" or file_choice == "":
        print("Returning to main menu.")
        return
    try:
        file_idx = int(file_choice) - 1
        if 0 <= file_idx < len(audio_files):
            audio_file = os.path.abspath(os.path.join(local_dir, audio_files[file_idx]))
            print(f"Selected file: {audio_file}")
            if not os.path.exists(audio_file):
                print(f"File does not exist (double-check!): {audio_file}")
                return
            print("Transcribing audio...")
            transcript_file = transcribe_audio(audio_file)
            print(f"Transcript saved to: {transcript_file}")
            with open(transcript_file, "r", encoding="utf-8") as f:
                print("Transcript Preview:")
                print(f.read(500))
        else:
            print("Invalid selection.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    """
    Main entry point for the application.
    Handles backend/model selection, displays the main menu, and routes user choices.
    """
    print("\n")
    print(" ███████╗████████╗███████╗██████╗ ██╗  ██╗███████╗███╗   ██╗")
    print(" ██╔════╝╚══██╔══╝██╔════╝██╔══██╗██║  ██║██╔════╝████╗  ██║")
    print(" ███████╗   ██║   █████╗  ██████╔╝███████║█████╗  ██╔██╗ ██║")
    print(" ╚════██║   ██║   ██╔══╝  ██╔═══╝ ██╔══██║██╔══╝  ██║╚██╗██║")
    print(" ███████║   ██║   ███████╗██║     ██║  ██║███████╗██║ ╚████║")
    print(" ╚══════╝   ╚═╝   ╚══════╝╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝")
    print()
    print("          🛠️  Stephen's AI-Powered Toolkit 🛠️")
    print()

    global client
    base_url, api_key, backend, default_model = select_backend_and_model()

    try:
        if base_url:
            client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=60
            )
            # Only check models.list() for LM Studio (not Ollama, which uses different API)
            if backend != "Ollama":
                client.models.list()
        else:
            client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(Fore.RED + f"\nCould not connect to local backend ({backend}). Falling back to OpenAI Cloud." + Style.RESET_ALL)
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        backend = "OpenAI Cloud"
        default_model = "gpt-4o"

    model_name = default_model
    print(Fore.GREEN + f"Using model: {model_name}" + Style.RESET_ALL)

    # Check if local backend server is running first
    if backend in ["LM Studio", "Ollama"]:
        if not check_server_running(base_url, backend):
            print(Fore.RED + f"\n{backend} server is not running. Please start it and try again." + Style.RESET_ALL)
            return
        else:
            print(Fore.GREEN + f"{backend} server is running and ready!" + Style.RESET_ALL)
            # Show GPU status at startup
            print_gpu_status("Startup")

    # Optional: Initial greeting from the assistant (skipped for faster startup)
    # Uncomment below to enable greeting message
    # try:
    #     response = client.chat.completions.create(
    #         model=model_name,
    #         messages=[{"role": "user", "content": "Say hello and briefly introduce yourself as a YouTube transcription assistant."}],
    #         timeout=10
    #     )
    #     print(f"\n{response.choices[0].message.content}\n")
    # except Exception as e:
    #     pass  # Skip greeting if it fails

    while True:
        print(Fore.CYAN + "\nChoose an option:" + Style.RESET_ALL)
        print("1. Download a YouTube file for processing")
        print("2. Transcribe a local audio file")
        print("3. Select an audio file to transcribe")
        print("4. Combine transcript files")
        print("5. Start a chat session")
        print("6. Summarize a transcript file")
        print("7. Download a SoundCloud audio file")
        print("8. Delete a file")
        print("9. OCR images in a folder")
        print("10. Combine images to PDF")
        print("11. Exit")
        choice = input(Fore.YELLOW + "Enter your choice (1-11): " + Style.RESET_ALL).strip()

        if choice == "1":
            youtube_url = input(Fore.YELLOW + "Enter the YouTube Video URL: " + Style.RESET_ALL).strip()
            try:
                print("Downloading audio...")
                audio_file = download_audio(youtube_url)
                print(Fore.GREEN + f"Audio downloaded to: {audio_file}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)

        elif choice == "2":
            transcribe_local_audio()
        elif choice == "3":
            audio_files = list_audio_files()
            if not audio_files:
                print("No audio files found in the audio directory.")
                continue
            print("\nAvailable audio files:")
            for idx, fname in enumerate(audio_files, 1):
                print(f"{idx}. {fname}")
            file_choice = input("Select a file number to transcribe (or 'q' to return): ").strip()
            if file_choice.lower() == "q" or file_choice == "":
                print("Returning to main menu.")
                continue
            try:
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(audio_files):
                    audio_file = os.path.abspath(os.path.join("audio", audio_files[file_idx]))
                    print(f"Selected file: {audio_file}")
                    if not os.path.exists(audio_file):
                        print(f"File does not exist (double-check!): {audio_file}")
                        continue
                    print("Transcribing audio...")
                    transcript_file = transcribe_audio(audio_file)
                    print(f"Transcript saved to: {transcript_file}")
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        print("Transcript Preview:")
                        print(f.read(500))
                else:
                    print("Invalid selection.")
            except Exception as e:
                print(f"An error occurred: {e}") 

        elif choice == "4":
            combine_transcript_files()
        elif choice == "5":
            chat_session(model_name, backend)
        elif choice == "6":
            transcript_files = list_transcript_files()
            if not transcript_files:
                print("No transcript files found in the transcripts directory.")
                continue
            print("\nAvailable transcript files:")
            for idx, fname in enumerate(transcript_files, 1):
                print(f"{idx}. {fname}")
            file_choice = input("Select a file number to summarize (or 'q' to return): ").strip()
            if file_choice.lower() == "q" or file_choice == "":
                print("Returning to main menu.")
                continue
            try:
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(transcript_files):
                    transcript_file = os.path.join(os.path.dirname(__file__), "transcripts", transcript_files[file_idx])
                    print(f"Selected file: {transcript_file}")
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        text = f.read()
                    print("Summarizing transcript (this may take a moment)...")
                    summary = summarize_text(text, model_name)
                    if summary is None:
                        continue
                    print("\nSummary:\n", summary)
                    save_choice = input("\nWould you like to save this summary to the summaries directory? (y/n): ").strip().lower()
                    if save_choice == "y":
                        summaries_dir = os.path.join(os.path.dirname(__file__), "summaries")
                        if not os.path.exists(summaries_dir):
                            os.makedirs(summaries_dir)
                        base_name = os.path.splitext(os.path.basename(transcript_file))[0]
                        summary_file = os.path.join(summaries_dir, f"{base_name}_summary.txt")
                        with open(summary_file, "w", encoding="utf-8") as f:
                            f.write(summary)
                        print(f"Summary saved to: {summary_file}")
                else:
                    print("Invalid selection.")
            except Exception as e:
                print(f"An error occurred: {e}")

        elif choice == "7":
            soundcloud_url = input(Fore.YELLOW + "Enter the SoundCloud URL: " + Style.RESET_ALL).strip()
            try:
                print("Downloading SoundCloud audio...")
                audio_file = download_soundcloud_audio(soundcloud_url)
                print(Fore.GREEN + f"Audio downloaded to: {audio_file}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
        elif choice == "8":
            delete_file_menu()
        elif choice == "9":
            ocr_main()
        elif choice == "10":
            combine_images_to_pdf()
        elif choice == "11":
            print(Fore.CYAN + "Goodbye!" + Style.RESET_ALL)
            break
def combine_images_to_pdf():
    """
    Combines images from the OCR directory into a single PDF file.
    Supports both direct images in src/ocr/ or organized in subfolders.
    """
    try:
        from PIL import Image as PILImage
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except ImportError:
        print(Fore.RED + "reportlab is required for PDF creation. Install with 'pip install reportlab'." + Style.RESET_ALL)
        return
    
    # Look for ocr directory
    ocr_root = os.path.join(os.path.dirname(__file__), 'ocr')
    if not os.path.exists(ocr_root):
        print(Fore.RED + f"OCR directory does not exist at: {ocr_root}" + Style.RESET_ALL)
        return
    
    # Check for subfolders and direct images
    folders = [f for f in os.listdir(ocr_root) if os.path.isdir(os.path.join(ocr_root, f))]
    direct_images = find_image_files(ocr_root)
    
    if not folders and not direct_images:
        print(Fore.RED + "No folders or images found in OCR directory." + Style.RESET_ALL)
        return
    
    # Let user choose folder
    chosen_folder = None
    if direct_images and not folders:
        print(Fore.CYAN + f"\nFound {len(direct_images)} images in OCR root directory." + Style.RESET_ALL)
        process_choice = input(Fore.YELLOW + "Create PDF from these images? (y/n): " + Style.RESET_ALL).strip().lower()
        if process_choice != "y":
            return
        chosen_folder = ocr_root
    elif direct_images and folders:
        print("\nOptions:")
        print(f"0. Use {len(direct_images)} images from OCR root directory")
        for idx, folder in enumerate(folders, 1):
            print(f"{idx}. {folder}")
        folder_choice = input(Fore.YELLOW + "Select an option (or 'q' to return): " + Style.RESET_ALL).strip()
        if folder_choice.lower() == "q":
            return
        try:
            choice_idx = int(folder_choice)
            if choice_idx == 0:
                chosen_folder = ocr_root
            elif 1 <= choice_idx <= len(folders):
                chosen_folder = os.path.join(ocr_root, folders[choice_idx - 1])
            else:
                print(Fore.RED + "Invalid selection." + Style.RESET_ALL)
                return
        except ValueError:
            print(Fore.RED + "Invalid input." + Style.RESET_ALL)
            return
    else:
        print("\nAvailable OCR folders:")
        for idx, folder in enumerate(folders, 1):
            print(f"{idx}. {folder}")
        folder_choice = input(Fore.YELLOW + "Select a folder number (or 'q' to return): " + Style.RESET_ALL).strip()
        if folder_choice.lower() == "q":
            return
        try:
            folder_idx = int(folder_choice) - 1
            if 0 <= folder_idx < len(folders):
                chosen_folder = os.path.join(ocr_root, folders[folder_idx])
            else:
                print(Fore.RED + "Invalid selection." + Style.RESET_ALL)
                return
        except ValueError:
            print(Fore.RED + "Invalid input." + Style.RESET_ALL)
            return
    
    # Get all images from chosen folder
    image_files = find_image_files(chosen_folder)
    
    if not image_files:
        print(Fore.RED + "No image files found in selected folder." + Style.RESET_ALL)
        return
    
    image_files.sort()  # Sort by filename
    
    # Ask for PDF filename
    pdf_name = input(Fore.YELLOW + "Enter PDF filename (without .pdf extension): " + Style.RESET_ALL).strip()
    if not pdf_name:
        print("No filename entered. Returning to main menu.")
        return
    
    pdf_path = os.path.join(chosen_folder, f"{pdf_name}.pdf")
    
    # Create PDF
    try:
        print(f"\n{Fore.CYAN}Creating PDF with {len(image_files)} images...{Style.RESET_ALL}")
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        page_width, page_height = letter
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"{Fore.CYAN}Adding image {idx}/{len(image_files)}: {os.path.basename(img_path)}{Style.RESET_ALL}")
            
            try:
                # Handle HEIC conversion
                ext = os.path.splitext(img_path)[1].lower()
                if ext in ['.heic']:
                    try:
                        import pillow_heif
                        heif_file = pillow_heif.read_heif(img_path)
                        img = PILImage.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
                    except ImportError:
                        print(Fore.YELLOW + f"  ⚠ Skipping HEIC file (pillow-heif not installed): {img_path}" + Style.RESET_ALL)
                        continue
                else:
                    img = PILImage.open(img_path)
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Calculate scaling to fit page
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                
                if img_width > page_width or img_height > page_height:
                    if aspect > 1:
                        # Portrait
                        display_height = page_height - 40
                        display_width = display_height / aspect
                    else:
                        # Landscape
                        display_width = page_width - 40
                        display_height = display_width * aspect
                else:
                    display_width = img_width
                    display_height = img_height
                
                # Center image on page
                x = (page_width - display_width) / 2
                y = (page_height - display_height) / 2
                
                # Draw image
                c.drawImage(ImageReader(img), x, y, width=display_width, height=display_height)
                c.showPage()  # Start new page for next image
                
            except Exception as e:
                print(Fore.RED + f"  ✗ Error adding {img_path}: {e}" + Style.RESET_ALL)
                continue
        
        c.save()
        print(Fore.GREEN + f"\n✓ PDF created successfully: {pdf_path}" + Style.RESET_ALL)
        
    except Exception as e:
        print(Fore.RED + f"Error creating PDF: {e}" + Style.RESET_ALL)

def _cleanup_lmstudio_for_ocr(vision_model: str):
    """
    For LM Studio: unload chat/other models to free GPU VRAM for OCR.
    Keeps only the vision model loaded.
    """
    try:
        print(Fore.CYAN + "\nFreeing GPU memory for OCR..." + Style.RESET_ALL)
        response = _http_session.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            loaded = [m.get('id') or m.get('model') or m.get('name') for m in models_data.get('data', [])]
            loaded = [m for m in loaded if isinstance(m, str)]
            to_unload = [m for m in loaded if m != vision_model]
            
            if to_unload:
                print(Fore.YELLOW + f"Found other models loaded: {', '.join(to_unload)}" + Style.RESET_ALL)
                for model in to_unload:
                    try:
                        # Unload by posting a minimal request that will release the model
                        print(Fore.YELLOW + f"  Unloading {model} to free VRAM..." + Style.RESET_ALL)
                        _http_session.post(
                            "http://localhost:1234/v1/chat/completions",
                            json={
                                "model": model,
                                "messages": [{"role": "user", "content": "unload"}],
                                "max_tokens": 1,
                                "temperature": 0
                            },
                            timeout=5
                        )
                        time.sleep(1)
                    except Exception as e:
                        print(Fore.YELLOW + f"  Could not unload {model}: {e}" + Style.RESET_ALL)
                
                print(Fore.GREEN + f"✓ LM Studio memory optimization complete" + Style.RESET_ALL)
                time.sleep(2)
            else:
                print(Fore.GREEN + f"✓ Only {vision_model} is loaded; GPU ready for OCR" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.YELLOW + f"Could not optimize LM Studio memory: {e}" + Style.RESET_ALL)


def ocr_main():
    """
    Processes images in the OCR directory using whatever AI vision model is
    already loaded in LM Studio or Ollama.  If no vision model is detected
    the user is told to load one and the function returns.
    Tesseract is kept only as a per-image fallback when the AI model fails
    on an individual image.
    """
    # Show current GPU status and reclaim cache before OCR workload.
    print_gpu_status("Before OCR")
    cleanup_vram("Pre-OCR", aggressive=True, verbose=True)
    print_gpu_status("After Pre-OCR Cleanup")

    vision_model = None
    ai_base_url = None
    
    print(Fore.CYAN + "\nChecking for a loaded AI vision model..." + Style.RESET_ALL)
    
    # Helper: ask a backend which models are loaded and pick the first vision one
    def _detect_loaded_vision_model(url, backend_name):
        try:
            response = _http_session.get(f"{url}/v1/models", timeout=3)
            if response.status_code == 200:
                models_data = response.json()
                loaded = [
                    (m.get('id') or m.get('model') or m.get('name'))
                    for m in models_data.get('data', [])
                ]
                loaded = [m for m in loaded if isinstance(m, str)]
                print(Fore.CYAN + f"{backend_name} models loaded: {', '.join(loaded) if loaded else 'None'}" + Style.RESET_ALL)
                vision = [m for m in loaded if is_vision_model(m)]
                if vision:
                    return vision[0]          # use whatever is already loaded
            return None
        except Exception as e:
            print(Fore.YELLOW + f"{backend_name} not accessible: {e}" + Style.RESET_ALL)
            return None
    
    # Check LM Studio first, then Ollama
    model = _detect_loaded_vision_model("http://localhost:1234", "LM Studio")
    if model:
        vision_model = model
        ai_base_url = "http://localhost:1234"
        print(Fore.GREEN + f"✓ Using LM Studio vision model: {vision_model}" + Style.RESET_ALL)
    else:
        model = _detect_loaded_vision_model("http://localhost:11434", "Ollama")
        if model:
            vision_model = model
            ai_base_url = "http://localhost:11434"
            print(Fore.GREEN + f"✓ Using Ollama vision model: {vision_model}" + Style.RESET_ALL)
    
    # ---- No vision model? Tell the user and stop ----
    if not vision_model:
        print(Fore.RED + "\n⚠  No AI vision model is loaded!" + Style.RESET_ALL)
        print(Fore.YELLOW + "Please load a vision-capable model before running OCR:" + Style.RESET_ALL)
        print(Fore.YELLOW + "  • LM Studio — load qwen2.5-vl, llava, or similar" + Style.RESET_ALL)
        print(Fore.YELLOW + "  • Ollama   — ollama run qwen2.5-vl:7b" + Style.RESET_ALL)
        print(Fore.YELLOW + "Then re-select option 9 from the main menu." + Style.RESET_ALL)
        return
    
    # Look for ocr directory in the same folder as main.py (src/ocr)
    ocr_root = os.path.join(os.path.dirname(__file__), 'ocr')
    if not os.path.exists(ocr_root):
        print(Fore.RED + f"OCR directory does not exist at: {ocr_root}" + Style.RESET_ALL)
        print(Fore.YELLOW + "Please create the directory: src/ocr/" + Style.RESET_ALL)
        return
    
    # Check for subfolders
    folders = [f for f in os.listdir(ocr_root) if os.path.isdir(os.path.join(ocr_root, f))]
    
    # Check for images directly in ocr root
    direct_images = find_image_files(ocr_root)
    
    if not folders and not direct_images:
        print(Fore.RED + "No folders or images found in OCR directory." + Style.RESET_ALL)
        return
    
    # If there are images directly in ocr root, process them
    if direct_images and not folders:
        print(Fore.CYAN + f"\nFound {len(direct_images)} images in OCR directory." + Style.RESET_ALL)
        process_choice = input(Fore.YELLOW + "Process these images? (y/n): " + Style.RESET_ALL).strip().lower()
        if process_choice != "y":
            print("Returning to main menu.")
            return
        chosen_folder = ocr_root
    # If both exist, let user choose
    elif direct_images and folders:
        print("\nOptions:")
        print(f"0. Process {len(direct_images)} images in OCR root directory")
        for idx, folder in enumerate(folders, 1):
            print(f"{idx}. {folder}")
        folder_choice = input(Fore.YELLOW + "Select an option (or 'q' to return): " + Style.RESET_ALL).strip()
        if folder_choice.lower() == "q" or folder_choice == "":
            print("Returning to main menu.")
            return
        try:
            choice_idx = int(folder_choice)
            if choice_idx == 0:
                chosen_folder = ocr_root
            elif 1 <= choice_idx <= len(folders):
                chosen_folder = os.path.join(ocr_root, folders[choice_idx - 1])
            else:
                print(Fore.RED + "Invalid selection." + Style.RESET_ALL)
                return
        except ValueError:
            print(Fore.RED + "Invalid input." + Style.RESET_ALL)
            return
    # Only folders exist
    else:
        print("\nAvailable OCR folders:")
        for idx, folder in enumerate(folders, 1):
            print(f"{idx}. {folder}")
        folder_choice = input(Fore.YELLOW + "Select a folder number to process (or 'q' to return): " + Style.RESET_ALL).strip()
        if folder_choice.lower() == "q" or folder_choice == "":
            print("Returning to main menu.")
            return
        try:
            folder_idx = int(folder_choice) - 1
            if 0 <= folder_idx < len(folders):
                chosen_folder = os.path.join(ocr_root, folders[folder_idx])
            else:
                print(Fore.RED + "Invalid selection." + Style.RESET_ALL)
                return
        except ValueError:
            print(Fore.RED + "Invalid input." + Style.RESET_ALL)
            return
    
    # Now process the chosen folder
    try:
        image_files = find_image_files(chosen_folder)
        if not image_files:
            print(Fore.RED + "No image files found in selected folder." + Style.RESET_ALL)
            return
        image_files.sort()  # Sort files by filename
        folder_name = os.path.basename(chosen_folder) if chosen_folder != ocr_root else "OCR root"
        print(f"\nProcessing {len(image_files)} images in '{folder_name}' (sorted by filename)...")
        
        # Warn about large batches
        if len(image_files) > 20 and vision_model:
            print(Fore.YELLOW + f"⚠ Large batch ({len(image_files)} images) — pausing every 5 images to stay stable." + Style.RESET_ALL)
        
        ocr_txt_files = []
        consecutive_failures = 0  # Track consecutive failures to detect model degradation
        
        for idx, img_path in enumerate(image_files, 1):
            print(Fore.MAGENTA + f"\n[{idx}/{len(image_files)}] Processing: {os.path.basename(img_path)}" + Style.RESET_ALL)
            try:
                # Periodic cleanup while OCR is running helps reduce VRAM pressure.
                if idx == 1 or idx % 5 == 0:
                    cleanup_vram(f"Before image {idx}/{len(image_files)}", aggressive=(idx % 5 == 0), verbose=True)

                # Reset best_text at the start of each image to prevent stale data
                best_text = ""
                ext = os.path.splitext(img_path)[1].lower()
                original_path = img_path  # Save original path for deletion later
                
                if ext in ['.heic', '.heif']:
                    try:
                        import pillow_heif
                        heif_file = pillow_heif.read_heif(img_path)
                        img = Image.frombytes(
                            heif_file.mode,
                            heif_file.size,
                            heif_file.data,
                            "raw"
                        )
                        png_path = os.path.splitext(img_path)[0] + '_converted.png'
                        img.save(png_path, format='PNG')
                        print(Fore.YELLOW + f"Converted {img_path} to {png_path}" + Style.RESET_ALL)
                        img_path = png_path
                        # Re-open the PNG for preprocessing
                        img = Image.open(img_path)
                    except ImportError:
                        print(Fore.RED + "pillow-heif is required for HEIC/HEIF support. Install with 'pip install pillow-heif'." + Style.RESET_ALL)
                        continue
                    except Exception as e:
                        print(Fore.RED + f"Error converting {img_path}: {e}" + Style.RESET_ALL)
                        continue
                else:
                    # Load non-HEIC images
                    img = Image.open(img_path)
                
                # Convert to RGB up-front (needed for both AI and Tesseract paths)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')

                # Perform OCR based on available method
                print(Fore.CYAN + f"Running OCR on {os.path.basename(img_path)}..." + Style.RESET_ALL)
                
                if vision_model:
                        # Use AI vision model for OCR
                        try:
                            ocr_prompt = (
                                "You are an OCR engine. Extract ALL text from this image exactly as it appears. "
                                "Preserve the original layout, line breaks, and formatting. "
                                "Output ONLY the extracted text — no commentary, no descriptions, no markdown."
                            )

                            # Encode image for the vision API — use smaller size to avoid overwhelming LM Studio
                            img_base64 = img_to_b64(img, max_side=1024, quality=70)
                            
                            # Determine which API to use based on where we found the model
                            vision_response = None
                            skip_tesseract = False  # Reserved for truly non-image failures
                            
                            # Use the detected backend (ai_base_url tells us which one)
                            is_ollama = ai_base_url and "11434" in ai_base_url
                            is_lmstudio = ai_base_url and "1234" in ai_base_url
                            
                            # Try LM Studio first if that's what we detected
                            if is_lmstudio and not vision_response:
                                lms_resp = None  # Initialize to track if request was made
                                try:
                                    lms_resp = _http_session.post(
                                        "http://localhost:1234/v1/chat/completions",
                                        json={
                                            "model": vision_model,
                                            "messages": [
                                                {
                                                    "role": "user",
                                                    "content": [
                                                        {"type": "text", "text": ocr_prompt},
                                                        {
                                                            "type": "image_url",
                                                            "image_url": {
                                                                "url": f"data:image/jpeg;base64,{img_base64}"
                                                            }
                                                        }
                                                    ]
                                                }
                                            ],
                                            "max_tokens": 4096,
                                            "temperature": 0.0
                                        },
                                        timeout=(10, 90)  # (connect timeout, read timeout)
                                    )
                                    if lms_resp.status_code == 200:
                                        result = lms_resp.json()
                                        candidate = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                                        cleaned = candidate.replace('?', '').replace(' ', '').replace('\n', '').replace('\r', '')
                                        print(Fore.CYAN + f"  [DEBUG] LM Studio response length: {len(candidate)} chars, cleaned: {len(cleaned)} chars" + Style.RESET_ALL)
                                        if candidate and len(candidate.strip()) > 20:
                                            print(Fore.CYAN + f"  [DEBUG] First 200 chars: {candidate[:200]!r}" + Style.RESET_ALL)
                                        if candidate and len(cleaned) > len(candidate) * 0.1:
                                            best_text = candidate
                                            print(Fore.GREEN + f"✓ AI OCR completed (LM Studio)" + Style.RESET_ALL)
                                            vision_response = True
                                        else:
                                            print(Fore.YELLOW + f"⚠ LM Studio returned low-quality result (cleaned {len(cleaned)}/{len(candidate)} chars)" + Style.RESET_ALL)
                                            if candidate:
                                                print(Fore.YELLOW + f"  Response preview: {candidate[:300]!r}" + Style.RESET_ALL)
                                    else:
                                        err = ""
                                        try: err = lms_resp.text[:200]
                                        except Exception: pass
                                        print(Fore.YELLOW + f"LM Studio returned {lms_resp.status_code}: {err}" + Style.RESET_ALL)
                                        # Do not skip OCR fallback on 400; this is often recoverable
                                        # (payload/image-size issues can still OCR via local Tesseract).
                                        if lms_resp.status_code == 400:
                                            vision_response = False
                                            skip_tesseract = False
                                except Exception as e:
                                    print(Fore.YELLOW + f"LM Studio error: {e}" + Style.RESET_ALL)
                                
                                # Retry once with a smaller image when first attempt fails.
                                if not vision_response:
                                    try:
                                        time.sleep(3)
                                        img_b64_small = img_to_b64(img, max_side=512, quality=50)
                                        lms_resp2 = _http_session.post(
                                            "http://localhost:1234/v1/chat/completions",
                                            json={
                                                "model": vision_model,
                                                "messages": [{"role": "user", "content": [
                                                    {"type": "text", "text": ocr_prompt},
                                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64_small}"}}
                                                ]}],
                                                "max_tokens": 4096,
                                                "temperature": 0.0
                                            },
                                            timeout=(5, 45)  # (connect timeout, read timeout)
                                        )
                                        if lms_resp2.status_code == 200:
                                            r2 = lms_resp2.json()
                                            t2 = r2.get('choices', [{}])[0].get('message', {}).get('content', '')
                                            c2 = t2.replace('?', '').replace(' ', '').replace('\n', '').replace('\r', '')
                                            if t2 and len(c2) > len(t2) * 0.1:
                                                best_text = t2
                                                print(Fore.GREEN + f"✓ AI OCR completed (LM Studio, retry)" + Style.RESET_ALL)
                                                vision_response = True
                                    except Exception as e2:
                                        print(Fore.YELLOW + f"LM Studio retry error: {e2}" + Style.RESET_ALL)
                            
                            # Try Ollama if that's what we detected (or as fallback)
                            if is_ollama and not vision_response:
                                def _ollama_generate_call(b64_img, timeout_read=120):
                                    payload = {
                                        "model": vision_model,
                                        "prompt": ocr_prompt,
                                        "images": [b64_img],
                                        "stream": False,
                                        "keep_alive": "10m",
                                        "num_predict": 1024,
                                        "temperature": 0.0,
                                    }
                                    return _http_session.post(
                                        "http://localhost:11434/api/generate",
                                        json=payload,
                                        timeout=(5, timeout_read)
                                    )

                                try:
                                    start = time.perf_counter()
                                    gen_resp = _ollama_generate_call(img_base64, timeout_read=120)
                                    duration = time.perf_counter() - start
                                    if DEBUG_OCR:
                                        dbg_path = os.path.splitext(img_path)[0] + '_ai_ocr_debug.json'
                                        dbg = {
                                            "attempt": "primary",
                                            "endpoint": "/api/generate",
                                            "status_code": gen_resp.status_code,
                                            "duration_sec": round(duration, 3),
                                            "model": vision_model,
                                            "image_b64_len": len(img_base64),
                                        }
                                        try:
                                            dbg["response_json"] = gen_resp.json()
                                        except Exception:
                                            dbg["response_text"] = gen_resp.text
                                        try:
                                            with open(dbg_path, 'w', encoding='utf-8') as df:
                                                json.dump(dbg, df, ensure_ascii=False, indent=2)
                                            print(Fore.CYAN + f"Debug saved: {dbg_path}" + Style.RESET_ALL)
                                        except Exception:
                                            pass
                                    if gen_resp.status_code == 200:
                                        result = gen_resp.json()
                                        candidate_text = result.get('response', '') or ''
                                        # Validate: check if response is mostly garbage (all ?'s or non-printable)
                                        cleaned = candidate_text.replace('?', '').replace(' ', '').replace('\n', '').replace('\r', '')
                                        # Check if model refused to OCR (blurry, can't read, etc.)
                                        refusal_phrases = ['blurry', 'unable to', 'cannot read', 'can\'t read', 'not able to', 'unclear', 'illegible', 'too low', 'higher resolution']
                                        is_refusal = any(phrase in candidate_text.lower() for phrase in refusal_phrases)
                                        if is_refusal:
                                            print(Fore.YELLOW + f"⚠ AI declined to OCR (chart/blurry?), falling back to Tesseract" + Style.RESET_ALL)
                                        elif candidate_text and len(cleaned) > len(candidate_text) * 0.1:
                                            best_text = candidate_text
                                            print(Fore.GREEN + f"✓ AI OCR completed (Ollama generate)" + Style.RESET_ALL)
                                            vision_response = True
                                        else:
                                            print(Fore.YELLOW + f"⚠ AI returned garbage response, will retry..." + Style.RESET_ALL)
                                    else:
                                        print(Fore.YELLOW + f"Ollama generate returned {gen_resp.status_code}; retrying with stronger compression" + Style.RESET_ALL)
                                except Exception as e:
                                    print(Fore.YELLOW + f"Ollama generate error: {e}; retrying smaller image" + Style.RESET_ALL)

                                # Retry once with smaller image and lower quality to avoid timeouts
                                if not vision_response:
                                    try:
                                        img_b64_small = img_to_b64(img, max_side=1024, quality=75)
                                        start2 = time.perf_counter()
                                        gen_resp2 = _http_session.post(
                                            "http://localhost:11434/api/generate",
                                            json={
                                                "model": vision_model,
                                                "prompt": ocr_prompt,
                                                "images": [img_b64_small],
                                                "stream": False,
                                                "keep_alive": "10m",
                                                "num_predict": 768,
                                                "temperature": 0.0,
                                            },
                                            timeout=(5, 120)
                                        )
                                        duration2 = time.perf_counter() - start2
                                        if DEBUG_OCR:
                                            dbg_path2 = os.path.splitext(img_path)[0] + '_ai_ocr_debug_retry.json'
                                            dbg2 = {
                                                "attempt": "retry",
                                                "endpoint": "/api/generate",
                                                "status_code": gen_resp2.status_code,
                                                "duration_sec": round(duration2, 3),
                                                "model": vision_model,
                                                "image_b64_len": len(img_b64_small),
                                            }
                                            try:
                                                dbg2["response_json"] = gen_resp2.json()
                                            except Exception:
                                                dbg2["response_text"] = gen_resp2.text
                                            try:
                                                with open(dbg_path2, 'w', encoding='utf-8') as df2:
                                                    json.dump(dbg2, df2, ensure_ascii=False, indent=2)
                                                print(Fore.CYAN + f"Debug saved: {dbg_path2}" + Style.RESET_ALL)
                                            except Exception:
                                                pass
                                        if gen_resp2.status_code == 200:
                                            result2 = gen_resp2.json()
                                            candidate_text2 = result2.get('response', '') or ''
                                            # Validate: check if response is mostly garbage
                                            cleaned2 = candidate_text2.replace('?', '').replace(' ', '').replace('\n', '').replace('\r', '')
                                            if candidate_text2 and len(cleaned2) > len(candidate_text2) * 0.1:
                                                best_text = candidate_text2
                                                print(Fore.GREEN + f"✓ AI OCR completed (Ollama generate, compressed)" + Style.RESET_ALL)
                                                vision_response = True
                                            else:
                                                print(Fore.YELLOW + f"⚠ AI retry also returned garbage, falling back to Tesseract" + Style.RESET_ALL)
                                        else:
                                            print(Fore.YELLOW + f"Ollama generate returned {gen_resp2.status_code} on retry" + Style.RESET_ALL)
                                    except Exception as e2:
                                        print(Fore.YELLOW + f"Ollama generate retry error: {e2}" + Style.RESET_ALL)
                            
                            if not vision_response:
                                if skip_tesseract:
                                    print(Fore.YELLOW + f"⚠ AI OCR failed with non-image error, skipping Tesseract" + Style.RESET_ALL)
                                    best_text = ""
                                else:
                                    print(Fore.YELLOW + f"⚠ AI OCR failed, falling back to Tesseract" + Style.RESET_ALL)
                                    best_text = _run_tesseract_with_auto_tiling(img, os.path.basename(img_path))
                        except Exception as e:
                            print(Fore.YELLOW + f"⚠ AI OCR error: {e}, using Tesseract" + Style.RESET_ALL)
                            best_text = _run_tesseract_with_auto_tiling(img, os.path.basename(img_path))
                else:
                    # No vision model — use traditional Tesseract OCR
                    best_text = _run_tesseract_with_auto_tiling(img, os.path.basename(img_path))

                # Last-chance rescue: prevent blank output files after AI path failures.
                if not best_text.strip():
                    print(Fore.YELLOW + "⚠ Empty OCR result; retrying with forced Tesseract rescue..." + Style.RESET_ALL)
                    rescue_text = _run_tesseract_with_auto_tiling(img, os.path.basename(img_path))
                    if rescue_text.strip():
                        best_text = rescue_text
                        print(Fore.GREEN + "✓ Tesseract rescue extracted text." + Style.RESET_ALL)

                # If still empty, write a marker instead of a blank file for easier triage.
                if not best_text.strip():
                    best_text = "[OCR_NO_TEXT]\n"
                    print(Fore.YELLOW + "⚠ No text could be extracted; writing OCR_NO_TEXT marker." + Style.RESET_ALL)
                
                # Validate OCR output - warn if it looks like garbage
                cleaned_text = best_text.replace('?', '').replace(' ', '').replace('\n', '')
                is_garbage = len(cleaned_text) == 0 and len(best_text) > 10
                is_empty = len(best_text.strip()) == 0
                
                if is_garbage:
                    print(Fore.RED + f"⚠ WARNING: OCR returned only garbage characters for {os.path.basename(img_path)}" + Style.RESET_ALL)
                    print(Fore.YELLOW + f"  This may indicate the AI model is overwhelmed or the image is unreadable." + Style.RESET_ALL)
                    consecutive_failures += 1
                elif is_empty:
                    print(Fore.YELLOW + f"⚠ No text extracted from {os.path.basename(img_path)}" + Style.RESET_ALL)
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0  # Reset on success
                
                # If we see 3+ consecutive failures, the model is likely degraded - force recovery
                if consecutive_failures >= 3 and vision_model:
                    print(Fore.RED + f"\n⚠ Detected {consecutive_failures} consecutive failures - model appears degraded" + Style.RESET_ALL)
                    print(Fore.YELLOW + f"  Forcing 10s pause and model context flush to recover..." + Style.RESET_ALL)
                    time.sleep(10)
                    
                    # Flush model context
                    is_ollama = ai_base_url and "11434" in ai_base_url
                    is_lmstudio = ai_base_url and "1234" in ai_base_url
                    if is_ollama:
                        try:
                            _http_session.post(
                                "http://localhost:11434/api/generate",
                                json={"model": vision_model, "keep_alive": "0"},
                                timeout=10
                            )
                            time.sleep(2)
                            _http_session.post(
                                "http://localhost:11434/api/generate",
                                json={"model": vision_model, "keep_alive": "10m", "prompt": ""},
                                timeout=30
                            )
                            print(Fore.GREEN + f"  Model context flushed, continuing..." + Style.RESET_ALL)
                        except Exception:
                            pass
                    elif is_lmstudio:
                        # LM Studio recovery: minimal reset request
                        print(Fore.YELLOW + f"  LM Studio may need higher Context Length (currently limited)." + Style.RESET_ALL)
                        print(Fore.YELLOW + f"  TIP: In LM Studio settings, increase Context Length to 8192+." + Style.RESET_ALL)
                        try:
                            _http_session.post(
                                "http://localhost:1234/v1/chat/completions",
                                json={
                                    "model": vision_model,
                                    "messages": [{"role": "user", "content": "reset"}],
                                    "max_tokens": 1,
                                    "temperature": 0
                                },
                                timeout=10
                            )
                            print(Fore.GREEN + f"  LM Studio context reset attempted, continuing..." + Style.RESET_ALL)
                        except Exception:
                            pass
                    
                    cleanup_vram("Post-recovery", aggressive=True, verbose=True)
                    
                    consecutive_failures = 0  # Reset after recovery
                
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(best_text)
                ocr_txt_files.append(txt_path)
                print(Fore.GREEN + f"Saved OCR text to: {txt_path}" + Style.RESET_ALL)
                
                # Aggressively free memory — delete the image and all encoded copies
                del img
                for _var in ('img_base64', 'img_b64_small', 'img_original',
                             'lms_resp', 'lms_resp2', 'gen_resp', 'gen_resp2'):
                    try:
                        del locals()[_var]  # noqa
                    except (KeyError, NameError):
                        pass
                cleanup_vram(verbose=False)
                
                # Delay between images to let vision model stabilize
                if vision_model and idx < len(image_files):
                    is_lmstudio = ai_base_url and "1234" in ai_base_url
                    is_ollama = ai_base_url and "11434" in ai_base_url
                    
                    # Every 5 images, take a longer pause and flush model context
                    if idx % 5 == 0:
                        print(Fore.CYAN + f"  Batch checkpoint ({idx}/{len(image_files)}): Pausing & flushing context..." + Style.RESET_ALL)
                        cleanup_vram("Batch checkpoint", aggressive=True, verbose=True)
                        time.sleep(3)
                        if is_ollama:
                            try:
                                _http_session.post(
                                    "http://localhost:11434/api/generate",
                                    json={"model": vision_model, "keep_alive": "0"},
                                    timeout=10
                                )
                                time.sleep(2)
                                _http_session.post(
                                    "http://localhost:11434/api/generate",
                                    json={"model": vision_model, "keep_alive": "10m", "prompt": ""},
                                    timeout=30
                                )
                                print(Fore.CYAN + f"  Flushed Ollama model context" + Style.RESET_ALL)
                            except Exception:
                                pass
                        elif is_lmstudio:
                            # LM Studio: unload/reload to clear KV cache
                            try:
                                _http_session.post(
                                    "http://localhost:1234/v1/chat/completions",
                                    json={
                                        "model": vision_model,
                                        "messages": [{"role": "user", "content": "ok"}],
                                        "max_tokens": 1,
                                        "temperature": 0
                                    },
                                    timeout=15
                                )
                                print(Fore.CYAN + f"  LM Studio context checkpoint" + Style.RESET_ALL)
                            except Exception:
                                pass
                    elif is_lmstudio:
                        time.sleep(2)
                    elif is_ollama:
                        time.sleep(1.5)
                    else:
                        time.sleep(1)
                
                # Delete original HEIC file if it was converted (keep the PNG)
                if original_path != img_path and os.path.exists(original_path):
                    if ext in ['.heic', '.heif']:
                        try:
                            os.remove(original_path)
                            print(Fore.YELLOW + f"Deleted original HEIC file: {original_path}" + Style.RESET_ALL)
                        except Exception as e:
                            print(Fore.RED + f"Error deleting {original_path}: {e}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"Error processing {img_path}: {e}" + Style.RESET_ALL)
        # Combine all OCR .txt files into one file
        combined_text = ""
        if ocr_txt_files:
            # Use folder name for the combined file
            folder_name = os.path.basename(chosen_folder) if chosen_folder != ocr_root else "ocr_root"
            combined_filename = f"{folder_name}_combined.txt"
            combined_path = os.path.join(chosen_folder, combined_filename)
            with open(combined_path, 'w', encoding='utf-8') as fout:
                for txt_file in ocr_txt_files:
                    with open(txt_file, 'r', encoding='utf-8') as fin:
                        content = fin.read()
                        fout.write(content + "\n")
                        combined_text += content + "\n"
            print(Fore.GREEN + f"\nCombined OCR text saved to: {combined_path}" + Style.RESET_ALL)
            
            # AI verification and cleanup
            print(Fore.CYAN + "\n" + "="*60 + Style.RESET_ALL)
            print(Fore.CYAN + "AI VERIFICATION & CLEANUP" + Style.RESET_ALL)
            print(Fore.CYAN + "="*60 + Style.RESET_ALL)
            
            # Flush stdout to ensure prompt appears
            sys.stdout.flush()
            
            # Get AI summary for verification
            verify = input(Fore.YELLOW + f"\nWould you like AI to verify the combined text captured all {len(image_files)} images correctly? (y/n): " + Style.RESET_ALL).strip().lower()
            
            if verify == 'y':
                print(Fore.CYAN + "\nGenerating AI verification summary..." + Style.RESET_ALL)
                
                # Use the SAME backend that did OCR for verification to avoid resource conflicts
                verification_model = None
                verification_base_url = None
                
                # If OCR used LM Studio, use LM Studio for verification
                is_ocr_lmstudio = ai_base_url and "1234" in ai_base_url
                is_ocr_ollama = ai_base_url and "11434" in ai_base_url
                
                if is_ocr_lmstudio:
                    verification_base_url = "http://localhost:1234/v1"
                    verification_model = vision_model  # Use same model as OCR
                    print(Fore.CYAN + f"Using LM Studio ({vision_model}) for verification..." + Style.RESET_ALL)
                elif is_ocr_ollama:
                    # For Ollama, resources are likely exhausted after heavy OCR
                    print(Fore.YELLOW + "⚠ Ollama was used for OCR and may be resource-exhausted." + Style.RESET_ALL)
                    print(Fore.YELLOW + "Verification would require unloading and reloading models." + Style.RESET_ALL)
                    skip_verification = input(Fore.YELLOW + "Skip verification to preserve resources? (y/n, default: y): " + Style.RESET_ALL).strip().lower()
                    if skip_verification != 'n':
                        verify = 'n'
                        print(Fore.YELLOW + "Verification skipped." + Style.RESET_ALL)
                    else:
                        try:
                            response = _http_session.get("http://localhost:11434/api/tags", timeout=5)
                            if response.status_code == 200:
                                models_data = response.json()
                                available = [m.get('name') for m in models_data.get('models', [])]
                                text_models = [m for m in available if not is_vision_model(m)]
                                if text_models:
                                    verification_model = text_models[0]
                                elif available:
                                    verification_model = available[0]
                            verification_base_url = "http://localhost:11434/v1"
                        except Exception as e:
                            print(Fore.YELLOW + f"Could not access Ollama for verification: {e}" + Style.RESET_ALL)
                            verify = 'n'
                
                if not verification_model:
                    print(Fore.YELLOW + "No model available for verification. Skipping." + Style.RESET_ALL)
                    verify = 'n'
                else:
                    print(Fore.CYAN + f"Using {verification_model} for verification..." + Style.RESET_ALL)
                
                if verify == 'y':
                    try:
                        # Create verification prompt
                        verification_prompt = f"""I have processed {len(image_files)} images through OCR. Please review this combined text and provide:

1. A brief summary of what content was captured
2. Whether the text appears complete and coherent
3. Any obvious gaps or missing sections
4. Overall quality assessment (Good/Fair/Poor)

Combined text:
{combined_text[:8000]}{"... [truncated]" if len(combined_text) > 8000 else ""}"""

                        # Initialize OpenAI client with correct backend credentials
                        verify_api_key = "sk-local" if "1234" in verification_base_url else "ollama"
                        client_verify = openai.OpenAI(
                            base_url=verification_base_url,
                            api_key=verify_api_key,
                            timeout=120.0,  # 2-minute hard timeout
                        )
                        
                        # Get AI verification
                        print(Fore.CYAN + "Waiting for model response (timeout: 120s)..." + Style.RESET_ALL)
                        completion = client_verify.chat.completions.create(
                            model=verification_model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that verifies OCR results for completeness and quality."},
                                {"role": "user", "content": verification_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=1000,
                            timeout=120.0,  # per-request timeout
                        )
                        
                        # Safely extract the response
                        if completion and hasattr(completion, 'choices') and completion.choices is not None and len(completion.choices) > 0:
                            verification_result = completion.choices[0].message.content
                            print(Fore.GREEN + "\n" + "="*60 + Style.RESET_ALL)
                            print(Fore.GREEN + "AI VERIFICATION RESULT:" + Style.RESET_ALL)
                            print(Fore.GREEN + "="*60 + Style.RESET_ALL)
                            print(verification_result)
                            print(Fore.GREEN + "="*60 + Style.RESET_ALL)
                        else:
                            print(Fore.RED + "\nNo response received from AI model." + Style.RESET_ALL)
                            print(Fore.YELLOW + "This might be due to resource exhaustion after heavy OCR." + Style.RESET_ALL)
                    
                    except (openai.APITimeoutError, openai.APIConnectionError) as e:
                        print(Fore.RED + f"\nAI verification timed out or failed to connect: {e}" + Style.RESET_ALL)
                        print(Fore.YELLOW + "The model may still be recovering. Verification skipped." + Style.RESET_ALL)
                        
                    except Exception as e:
                        import traceback
                        print(Fore.RED + f"\nError during AI verification: {e}" + Style.RESET_ALL)
                        print(Fore.YELLOW + f"Verification failed (likely due to resource limits after heavy OCR). Continuing..." + Style.RESET_ALL)
                        if DEBUG_OCR:
                            print(Fore.YELLOW + "Debug info:" + Style.RESET_ALL)
                            print(Fore.YELLOW + traceback.format_exc() + Style.RESET_ALL)
            
            # Cleanup individual files
            print(Fore.YELLOW + f"\n{'='*60}" + Style.RESET_ALL)
            sys.stdout.flush()
            cleanup = input(Fore.YELLOW + f"Delete all individual image and text files, keeping only the combined file? (y/n): " + Style.RESET_ALL).strip().lower()
            
            if cleanup == 'y':
                deleted_images = 0
                deleted_texts = 0
                
                # Delete individual text files
                for txt_file in ocr_txt_files:
                    try:
                        os.remove(txt_file)
                        deleted_texts += 1
                    except Exception as e:
                        print(Fore.RED + f"Error deleting {txt_file}: {e}" + Style.RESET_ALL)
                
                # Re-scan for all image files in the folder (including any converted files)
                all_images = find_image_files(chosen_folder)
                
                # Delete all image files
                for img_path in all_images:
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            deleted_images += 1
                    except Exception as e:
                        print(Fore.RED + f"Error deleting {img_path}: {e}" + Style.RESET_ALL)
                
                print(Fore.GREEN + f"\n✓ Cleanup complete!" + Style.RESET_ALL)
                print(Fore.GREEN + f"  • Deleted {deleted_images} image file(s)" + Style.RESET_ALL)
                print(Fore.GREEN + f"  • Deleted {deleted_texts} text file(s)" + Style.RESET_ALL)
                print(Fore.GREEN + f"  • Kept combined file: {combined_path}" + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + "Files kept. You can manually delete them later." + Style.RESET_ALL)
                
    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
