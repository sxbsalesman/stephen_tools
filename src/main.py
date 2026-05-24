import pytesseract # type: ignore
from PIL import Image # type: ignore
import glob
import os
import re
import requests # type: ignore
import shutil
import platform
import time

import yt_dlp # type: ignore
import whisper # type: ignore
import torch # type: ignore
import openai # type: ignore
import subprocess
from colorama import Fore, Style, init # type: ignore

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
    Prompts the user to select which LLM backend to use (OpenAI, LM Studio, or Ollama).
    Returns the base_url, api_key, backend name, and default model name for the chosen backend.
    """
    print("\nChoose which LLM backend to use:")
    print("1. OpenAI Cloud (default) - gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.")
    print("2. LM Studio (local)")
    print("3. Ollama (local)")

    choice = input("Enter 1, 2, or 3 (or press Enter for OpenAI Cloud): ").strip()

    if choice == "2":
        base_url = "http://localhost:1234/v1"
        api_key = "sk-local"
        backend = "LM Studio"
        model_name = "local-model"
    elif choice == "3":
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"
        backend = "Ollama"
        # Check which models are available in Ollama
        try:
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available = [m.get('model') or m.get('name') for m in models_data.get('data', [])]
                if available:
                    model_name = available[0]  # Use first available model as default
                else:
                    model_name = "llama3:8b"  # Fallback if no models found
            else:
                model_name = "llama3:8b"  # Fallback
        except Exception:
            model_name = "llama3:8b"  # Fallback if Ollama not running yet
    else:
        base_url = None
        print("\nYou selected OpenAI Cloud.")
        api_key = input("Enter your OpenAI API key (or press Enter to use OPENAI_API_KEY environment variable): ").strip()
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print(Fore.RED + "No API key found! Please set OPENAI_API_KEY environment variable or enter it above." + Style.RESET_ALL)
        backend = "OpenAI Cloud"
        model_name = "gpt-4o"

    print(f"\n{Fore.GREEN}Selected backend: {backend}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Default model: {model_name}{Style.RESET_ALL}")

   ## device = "cuda" if torch.cuda.is_available() else "cpu"
   ## model_name = whisper.load_model("base", device=device)
   ## print('Are you are able to the Nvidia Cuda support?', torch.cuda.is_available())
   ## print(torch.cuda.get_device_name(0))

    return base_url, api_key, backend, model_name

def download_audio(youtube_url, output_path="audio"):
    """
    Downloads audio from a YouTube video using yt_dlp and saves it as an mp3 file
    in the specified output directory. Uses the local ffmpeg binary.
    Returns the path to the downloaded audio file.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    base_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'noplaylist': True,
        'retries': 10,
        'fragment_retries': 10,
        'extractor_retries': 3,
        'geo_bypass': True,
        'socket_timeout': 30,
        'quiet': False,
        'ffmpeg_location': ffmpeg_path,
    }

    # Some YouTube clients now force SABR/PO-token flows and can cause 403 responses.
    # Prefer non-Android clients first to reduce PO-token warnings, then fall back.
    attempts = [
        {
            'extractor_args': {
                'youtube': {
                    'player_client': ['tv', 'web']
                }
            }
        },
        {
            'extractor_args': {
                'youtube': {
                    'player_client': ['mweb', 'ios']
                }
            },
            'cookiesfrombrowser': ('chrome',)
        },
        {
            'extractor_args': {
                'youtube': {
                    'player_client': ['safari', 'web']
                }
            }
        },
        {
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'tv']
                }
            }
        },
    ]

    last_error = None
    for idx, extra_opts in enumerate(attempts, 1):
        ydl_opts = dict(base_opts)
        ydl_opts.update(extra_opts)
        # The "page needs to be reloaded" error can be transient; retry each strategy briefly.
        strategy_attempts = 2
        for attempt_no in range(1, strategy_attempts + 1):
            try:
                print(Fore.CYAN + f"Trying YouTube download strategy {idx}/{len(attempts)} (attempt {attempt_no}/{strategy_attempts})..." + Style.RESET_ALL)
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=True)
                    filename = ydl.prepare_filename(info)
                    audio_file = os.path.splitext(filename)[0] + '.mp3'
                    return audio_file
            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()
                should_retry_same_strategy = "page needs to be reloaded" in error_text and attempt_no < strategy_attempts
                if should_retry_same_strategy:
                    print(Fore.YELLOW + f"Transient YouTube error on strategy {idx}; retrying strategy..." + Style.RESET_ALL)
                    continue
                print(Fore.YELLOW + f"Download strategy {idx} failed: {exc}" + Style.RESET_ALL)
                break

    raise RuntimeError(
        "All YouTube download strategies failed. If this persists, upgrade yt-dlp and run Python 3.10+ for best compatibility."
    ) from last_error

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
    """
    transcript_path = os.path.join(os.path.dirname(__file__), transcript_dir)
    if not os.path.exists(transcript_path):
        os.makedirs(transcript_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
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

def _is_codeish_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("$"):
        return True
    if "--" in stripped:
        return True
    if "dash dash" in stripped.lower() or "double dash" in stripped.lower() or "minus minus" in stripped.lower():
        return True
    if stripped.startswith("http://") or stripped.startswith("https://"):
        return True
    return bool(
        re.match(
            r"^(?:sudo\s+)?(?:python3?|pip3?|yt-dlp|brew|conda|ffmpeg|git|curl|wget|npm|node|docker|kubectl|ollama|source|make)\b",
            stripped,
            flags=re.IGNORECASE,
        )
    )


def _normalize_technical_terms(text: str) -> str:
    """Apply conservative, high-confidence fixes for common Whisper technical errors.

    This runs before placeholder protection so corrected technical tokens can be preserved.
    """
    normalized = text

    # Common tool/name spacing issues
    normalized = re.sub(r"\by\s*t\s*-?\s*dlp\b", "yt-dlp", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bff\s*mpeg\b", "ffmpeg", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bol\s*lama\b", "ollama", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bl\s*m\s*studio\b", "LM Studio", normalized, flags=re.IGNORECASE)

    # Common OpenAI model name spacing (high confidence)
    normalized = re.sub(r"\bgpt\s*[- ]?4\s*o\b", "gpt-4o", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bgpt\s*[- ]?4\s*o\s*mini\b", "gpt-4o-mini", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bgpt\s*[- ]?4\s*turbo\b", "gpt-4-turbo", normalized, flags=re.IGNORECASE)

    # Turn spoken dash phrases into actual flag prefixes (only when clearly flag-like)
    normalized = re.sub(r"\b(?:dash dash|double dash|minus minus)\s+([A-Za-z][\w-]*)\b", r"--\1", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdash\s+([A-Za-z])\b", r"-\1", normalized, flags=re.IGNORECASE)

    return normalized


def _lift_inline_commands_to_fences(text: str) -> str:
    """Moves inline command snippets into their own fenced blocks.

    This improves retrieval quality and helps ensure commands/flags are preserved exactly.
    """
    lines = text.splitlines()
    out = []
    in_fence = False

    # Require at least one flag/path/url indicator so we don't fence ordinary sentences.
    cmd_start = re.compile(
        r"(?<!\w)(?:sudo\s+)?(?:python3?|pip3?|yt-dlp|ffmpeg|brew|conda|ollama|git|curl|wget|npm|node|docker|kubectl|make)\b",
        flags=re.IGNORECASE,
    )
    indicator = re.compile(r"(--[\w-]+|\s-[A-Za-z]\b|https?://|\b[\w./~:-]+\.(?:py|txt|md|json|yaml|yml|mp3|m4a|wav|mp4|pdf)\b)")

    def is_commandish_token(tok: str) -> bool:
        t = tok.strip().strip("\"'`.,;:)")
        if not t:
            return False
        if t.startswith("-"):
            return True
        if "--" in t:
            return True
        if "/" in t or "\\" in t:
            return True
        if ":" in t:
            return True
        if re.search(r"\.[A-Za-z0-9]{1,5}$", t):
            return True
        if t.lower() in {"yt-dlp", "ffmpeg", "pip", "pip3", "python", "python3", "brew", "conda", "ollama", "git", "curl", "wget", "npm", "node", "docker", "kubectl", "make"}:
            return True
        if re.fullmatch(r"gpt-[0-9][A-Za-z0-9._-]*", t, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"llama\d+(?::[0-9a-zA-Z._-]+)?", t, flags=re.IGNORECASE):
            return True
        return False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line.rstrip())
            continue
        if in_fence or not stripped:
            out.append(line.rstrip())
            continue

        if _is_codeish_line(line):
            out.append(line.rstrip())
            continue

        # Look for a command start inside the line.
        m = cmd_start.search(line)
        if not m or not indicator.search(line[m.start():]):
            out.append(line.rstrip())
            continue

        # Take the remainder of the line as a candidate command snippet, then trim trailing punctuation.
        candidate = line[m.start():].strip()
        trailing = ""
        while candidate and candidate[-1] in ".,;:)":
            trailing = candidate[-1] + trailing
            candidate = candidate[:-1]
            candidate = candidate.rstrip()

        # Only lift if it actually looks like a multi-token command.
        tokens = candidate.split()
        if len(tokens) < 2:
            out.append(line.rstrip())
            continue

        # Fence only up to the last "command-ish" token so we don't accidentally include prose.
        last_idx = -1
        for idx, tok in enumerate(tokens):
            if is_commandish_token(tok):
                last_idx = idx
        if last_idx < 1:
            out.append(line.rstrip())
            continue

        command_part = " ".join(tokens[: last_idx + 1]).strip()
        suffix_part = " ".join(tokens[last_idx + 1 :]).strip()

        prefix = line[:m.start()].rstrip()
        if prefix:
            out.append(prefix)
        out.append("```")
        out.append(command_part)
        out.append("```")
        if suffix_part:
            out.append(suffix_part)
        if trailing:
            out.append(trailing)

    return "\n".join(out).strip() + "\n"


def _fence_codeish_lines(text: str) -> str:
    """Wraps command-like lines in fenced code blocks so they can be preserved exactly."""
    lines = text.splitlines()
    out_lines = []
    block = []

    def flush_block():
        nonlocal block
        if not block:
            return
        out_lines.append("```")
        out_lines.extend(block)
        out_lines.append("```")
        block = []

    for line in lines:
        if _is_codeish_line(line):
            block.append(line.rstrip())
            continue
        flush_block()
        out_lines.append(line.rstrip())

    flush_block()
    return "\n".join(out_lines).strip() + "\n"


def _extract_fenced_code_blocks(markdown_text: str):
    """Replaces fenced code blocks with placeholders and returns (text, blocks)."""
    lines = markdown_text.splitlines()
    out_lines = []
    blocks = []
    in_block = False
    current_block_lines = []

    for line in lines:
        if line.strip().startswith("```"):
            if not in_block:
                in_block = True
                current_block_lines = ["```"]
            else:
                current_block_lines.append("```")
                blocks.append("\n".join(current_block_lines))
                placeholder = f"[[CODE_BLOCK_{len(blocks)}]]"
                out_lines.append(placeholder)
                in_block = False
                current_block_lines = []
            continue

        if in_block:
            current_block_lines.append(line)
        else:
            out_lines.append(line)

    # If the input had an unclosed fence, treat it as plain text.
    if in_block:
        out_lines.extend(current_block_lines)

    return "\n".join(out_lines).strip() + "\n", blocks


def _extract_inline_code_placeholders(text: str):
    """Replaces inline code-ish tokens with placeholders and returns (text, items).

    The items are restored later as markdown inline code spans using backticks.
    """
    items = []

    def add_item(raw: str) -> str:
        items.append(raw)
        return f"[[INLINE_CODE_{len(items)}]]"

    # Only transform non-code-block placeholder segments.
    segments = re.split(r"(\[\[CODE_BLOCK_\d+\]\])", text)
    out_segments = []

    flag_re = re.compile(r"(?<!\w)(--[A-Za-z0-9][\w-]*)(?:=([^\s`]+))?")
    short_flag_re = re.compile(r"(?<!\w)(-[a-zA-Z])(?!\w)")
    cmd_re = re.compile(
        r"(?<!\w)(yt-dlp|ffmpeg|python3?|pip3?|brew|conda|ollama|git|curl|wget|npm|node|docker|kubectl|make)(?!\w)",
        flags=re.IGNORECASE,
    )
    model_re = re.compile(
        r"(?<!\w)(gpt-[0-9][A-Za-z0-9._-]*|llama\d+(?::[0-9a-zA-Z._-]+)?|mixtral(?::[0-9a-zA-Z._-]+)?|gemma\d+(?::[0-9a-zA-Z._-]+)?|phi\d+(?::[0-9a-zA-Z._-]+)?)(?!\w)",
        flags=re.IGNORECASE,
    )
    path_like_re = re.compile(r"(?<!\w)([\w./~:-]+\.(?:py|txt|md|json|yaml|yml|mp3|m4a|wav|mp4|pdf))(?!\w)")

    for seg in segments:
        if re.fullmatch(r"\[\[CODE_BLOCK_\d+\]\]", seg or ""):
            out_segments.append(seg)
            continue

        updated = seg

        def flag_sub(m):
            if m.group(2) is not None:
                return add_item(f"{m.group(1)}={m.group(2)}")
            return add_item(m.group(1))

        updated = flag_re.sub(flag_sub, updated)
        updated = short_flag_re.sub(lambda m: add_item(m.group(1)), updated)
        updated = cmd_re.sub(lambda m: add_item(m.group(1)), updated)
        updated = model_re.sub(lambda m: add_item(m.group(1)), updated)
        updated = path_like_re.sub(lambda m: add_item(m.group(1)), updated)

        out_segments.append(updated)

    return "".join(out_segments).strip() + "\n", items


def _restore_inline_code_placeholders(markdown_text: str, items) -> str:
    restored = markdown_text
    for idx, raw in enumerate(items, 1):
        placeholder = f"[[INLINE_CODE_{idx}]]"
        restored = restored.replace(placeholder, f"`{raw}`")
    return restored


def _restore_fenced_code_blocks(markdown_text: str, blocks) -> str:
    restored = markdown_text
    missing = []
    for idx, block in enumerate(blocks, 1):
        placeholder = f"[[CODE_BLOCK_{idx}]]"
        if placeholder in restored:
            restored = restored.replace(placeholder, block)
        else:
            missing.append(block)

    if missing:
        if "## Commands" not in restored:
            restored = restored.rstrip() + "\n\n## Commands\n"
        restored = restored.rstrip() + "\n\n" + "\n\n".join(missing) + "\n"

    # Light normalization
    restored = re.sub(r"\n{4,}", "\n\n\n", restored)
    return restored.strip() + "\n"


def _split_text_preserve_paragraphs(text: str, max_words: int = 1800):
    """Splits text into chunks by paragraphs to keep markdown structure intact."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    current_words = 0

    for para in paragraphs:
        words = para.split()
        word_count = len(words)

        # If a single paragraph is huge (common for raw Whisper), split it into smaller pieces.
        if word_count > max_words:
            for i in range(0, word_count, max_words):
                piece = " ".join(words[i : i + max_words]).strip()
                if piece:
                    # Flush any accumulated chunk before appending a large split paragraph.
                    if current:
                        chunks.append("\n\n".join(current).strip() + "\n")
                        current = []
                        current_words = 0
                    chunks.append(piece + "\n")
            continue

        if current and (current_words + word_count) > max_words:
            chunks.append("\n\n".join(current).strip() + "\n")
            current = []
            current_words = 0
        current.append(para)
        current_words += word_count

    if current:
        chunks.append("\n\n".join(current).strip() + "\n")
    return chunks


def clean_and_reformat_transcript(text: str, model_name: str, title: str | None = None):
    """Cleans a raw Whisper transcript and returns valid Markdown."""
    text = _normalize_technical_terms(text)
    text = _lift_inline_commands_to_fences(text)
    protected = _fence_codeish_lines(text)
    placeholder_text, code_blocks = _extract_fenced_code_blocks(protected)
    placeholder_text, inline_items = _extract_inline_code_placeholders(placeholder_text)

    chunks = _split_text_preserve_paragraphs(placeholder_text, max_words=1800)
    cleaned_chunks = []

    request_timeout_s = int(os.environ.get("LLM_TIMEOUT_SECONDS", "300"))
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "3"))

    base_instruction = (
        "Clean and reformat this raw Whisper transcript into proper Markdown optimized for RAG/document retrieval.\n\n"
        "Hard requirements:\n"
        "- Output VALID Markdown only (no preface, no commentary).\n"
        "- Use '#' for the title, '##' for main sections, '###' for subsections.\n"
        "- Split into logical sections; keep paragraphs reasonably short.\n"
        "- Remove filler and repeated speech; keep technical meaning.\n"
        "- Correct obvious Whisper transcription errors ONLY when you are confident.\n"
        "- Do NOT invent information or add facts not present.\n\n"
        "Technical preservation rules (MUST follow):\n"
        "- Preserve all technical information exactly (tools, names, versions, paths, URLs, numbers).\n"
        "- Preserve commands and flags exactly.\n"
        "- Placeholders are immutable technical tokens:\n"
        "  - [[CODE_BLOCK_N]] is a fenced code block containing commands/flags.\n"
        "  - [[INLINE_CODE_N]] is an inline command/flag/path token.\n"
        "  You MUST NOT edit, delete, rename, wrap, or reorder these placeholders.\n"
        "- Output placement rules for placeholders:\n"
        "  - Put each [[CODE_BLOCK_N]] on its own line, surrounded by blank lines.\n"
        "  - Do NOT put [[CODE_BLOCK_N]] inside bullets, tables, blockquotes, or backticks.\n"
        "  - Keep [[INLINE_CODE_N]] inline in normal sentences (not in fenced blocks).\n\n"
        "Formatting guidance:\n"
        "- Use bullet lists for steps/checklists where appropriate.\n"
        "- Use tables only if clearly helpful; keep them simple.\n"
    )

    if title:
        base_instruction += f"\nTitle to use (exactly): {title}\n"

    for i, chunk in enumerate(chunks, 1):
        print(f"\nCleaning chunk {i}/{len(chunks)}...")
        if i == 1:
            chunk_instruction = (
                base_instruction
                + "\nChunking instructions:\n"
                + "- This is chunk 1: include the single H1 title at the top.\n"
                + "- Then create '##' and '###' headings as needed.\n"
            )
        else:
            chunk_instruction = (
                base_instruction
                + "\nChunking instructions:\n"
                + "- This is a continuation chunk: do NOT repeat the H1 title.\n"
                + "- Continue with appropriate '##' / '###' headings.\n"
            )

        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": f"{chunk_instruction}\n\nTranscript chunk:\n\n{chunk}"}
                    ],
                    timeout=request_timeout_s,
                    temperature=0,
                )
                cleaned_chunks.append(response.choices[0].message.content)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                err = str(exc).lower()
                is_timeout = "timed out" in err or "timeout" in err or type(exc).__name__.lower().endswith("timeouterror")
                if attempt >= max_retries or not is_timeout:
                    break
                backoff_s = 2 * attempt
                print(Fore.YELLOW + f"Request timed out for chunk {i}. Retrying ({attempt}/{max_retries}) after {backoff_s}s..." + Style.RESET_ALL)
                time.sleep(backoff_s)

        if last_exc is not None:
            raise last_exc

    cleaned = "\n\n".join(cleaned_chunks).strip() + "\n"

    # Ensure we have an H1 for the final output.
    if not cleaned.lstrip().startswith("#"):
        cleaned = (f"# {title}\n\n" if title else "# Transcript\n\n") + cleaned.lstrip()

    cleaned = _restore_fenced_code_blocks(cleaned, code_blocks)
    cleaned = _restore_inline_code_placeholders(cleaned, inline_items)
    return cleaned

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

def remove_keywords_from_text(text, keywords):
    """
    Removes a list of keywords from text using case-insensitive matching.
    """
    cleaned_text = text
    for keyword in keywords:
        cleaned_text = re.sub(re.escape(keyword), "", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text

def check_server_running(base_url, backend):
    """
    Checks if the LM Studio or Ollama server is running by sending a GET request to /models endpoint.
    Returns True if running, False otherwise.
    """
    try:
        # LM Studio uses /v1/models, Ollama uses /v1/models as well
        endpoint = f"{base_url}/v1/models"
        response = requests.get(endpoint, timeout=5)
        if response.status_code == 200:
            # If we get a 200 response, the server is running
            # Don't check for loaded models - just verify server is accessible
            return True
        else:
            print(Fore.RED + f"{backend} server responded with status code {response.status_code}." + Style.RESET_ALL)
            return False
    except Exception as e:
        print(Fore.RED + f"Could not connect to {backend} server at {base_url}: {e}" + Style.RESET_ALL)
        return False

def transcribe_local_audio():
    """
    Allows the user to select an audio file from the 'local' folder and transcribes it,
    saving the transcript in the 'transcripts' folder.
    """
    local_dir = os.path.join(os.path.dirname(__file__), "audio")
    if not os.path.exists(local_dir):
        print(Fore.RED + "No 'audio' directory found." + Style.RESET_ALL)
        return
    audio_files = [f for f in os.listdir(local_dir) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.mp4'))]
    if not audio_files:
        print("No audio files found in the 'audio' directory.")
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
            client.models.list()
        else:
            client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(Fore.RED + f"\nCould not connect to local backend ({backend}). Falling back to OpenAI Cloud." + Style.RESET_ALL)
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        backend = "OpenAI Cloud"
        default_model = "gpt-4o"

    # List available models
    available_models = []
    if backend == "OpenAI Cloud":
        # Show popular OpenAI models
        print(Fore.CYAN + "\nPopular OpenAI models:" + Style.RESET_ALL)
        popular_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
        for idx, model in enumerate(popular_models, 1):
            print(f" {idx}. {model}")
            available_models.append(model)
        print()
    else:
        # List models from local backend
        try:
            print(Fore.CYAN + "\nAvailable models on this backend:" + Style.RESET_ALL)
            models = client.models.list()
            for idx, m in enumerate(models.data, 1):
                print(f" {idx}. {m.id}")
                available_models.append(m.id)
            print()
        except Exception as e:
            print(Fore.RED + f"Could not list models: {e}" + Style.RESET_ALL)

    print("\n")
    user_input = input(Fore.YELLOW + f"Enter the model name or number (or press Enter to use default: {default_model}): " + Style.RESET_ALL).strip()
    print("\n")
    
    # Check if user entered a number to select from the list
    if user_input.isdigit() and available_models:
        model_idx = int(user_input) - 1
        if 0 <= model_idx < len(available_models):
            model_name = available_models[model_idx]
        else:
            print(Fore.YELLOW + f"Invalid selection. Using default: {default_model}" + Style.RESET_ALL)
            model_name = default_model
    else:
        model_name = user_input if user_input else default_model
    
    print(Fore.GREEN + f"Using model: {model_name}" + Style.RESET_ALL)

    # Check if local backend server is running first
    if backend in ["LM Studio", "Ollama"]:
        if not check_server_running(base_url, backend):
            print(Fore.RED + f"\n{backend} server is not running. Please start it and try again." + Style.RESET_ALL)
            return
        else:
            print(Fore.GREEN + f"{backend} server is running and ready!" + Style.RESET_ALL)

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
        print("6. Clean & reformat a transcript (Markdown)")
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
            file_choice = input("Select a file number to clean/reformat (or 'q' to return): ").strip()
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
                    print("Cleaning & reformatting transcript into Markdown (this may take a moment)...")
                    base_name = os.path.splitext(os.path.basename(transcript_file))[0]
                    cleaned_md = clean_and_reformat_transcript(text, model_name, title=base_name)
                    print("\nCleaned Markdown:\n", cleaned_md)
                    save_choice = input("\nWould you like to save this cleaned Markdown to the summaries directory? (y/n): ").strip().lower()
                    if save_choice == "y":
                        summaries_dir = os.path.join(os.path.dirname(__file__), "summaries")
                        if not os.path.exists(summaries_dir):
                            os.makedirs(summaries_dir)
                        cleaned_file = os.path.join(summaries_dir, f"{base_name}_cleaned.md")
                        with open(cleaned_file, "w", encoding="utf-8") as f:
                            f.write(cleaned_md)
                        print(f"Cleaned Markdown saved to: {cleaned_file}")
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
    direct_images = glob.glob(os.path.join(ocr_root, '*.png')) + \
                    glob.glob(os.path.join(ocr_root, '*.PNG')) + \
                    glob.glob(os.path.join(ocr_root, '*.jpg')) + \
                    glob.glob(os.path.join(ocr_root, '*.JPG')) + \
                    glob.glob(os.path.join(ocr_root, '*.jpeg')) + \
                    glob.glob(os.path.join(ocr_root, '*.JPEG')) + \
                    glob.glob(os.path.join(ocr_root, '*.bmp')) + \
                    glob.glob(os.path.join(ocr_root, '*.BMP')) + \
                    glob.glob(os.path.join(ocr_root, '*.heic')) + \
                    glob.glob(os.path.join(ocr_root, '*.HEIC'))
    
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
    image_files = glob.glob(os.path.join(chosen_folder, '*.png')) + \
                  glob.glob(os.path.join(chosen_folder, '*.PNG')) + \
                  glob.glob(os.path.join(chosen_folder, '*.jpg')) + \
                  glob.glob(os.path.join(chosen_folder, '*.JPG')) + \
                  glob.glob(os.path.join(chosen_folder, '*.jpeg')) + \
                  glob.glob(os.path.join(chosen_folder, '*.JPEG')) + \
                  glob.glob(os.path.join(chosen_folder, '*.bmp')) + \
                  glob.glob(os.path.join(chosen_folder, '*.BMP')) + \
                  glob.glob(os.path.join(chosen_folder, '*.heic')) + \
                  glob.glob(os.path.join(chosen_folder, '*.HEIC'))
    
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

def ocr_main():
    """
    Processes images in the OCR directory using AI vision model (via Ollama or LM Studio).
    Supports both direct images in src/ocr/ or organized in subfolders under src/ocr/.
    Falls back to Tesseract OCR if AI vision is not available.
    """
    # Check for vision models in Ollama or LM Studio
    use_ai_ocr = False
    vision_model = None
    ai_base_url = None
    
    print(Fore.CYAN + "\nChecking for AI vision models..." + Style.RESET_ALL)
    
    # Try Ollama first
    try:
        response = requests.get("http://localhost:11434/v1/models", timeout=3)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [m.get('model') or m.get('name') for m in models_data.get('data', [])]
            print(Fore.CYAN + f"Ollama models found: {', '.join(available_models) if available_models else 'None'}" + Style.RESET_ALL)
            vision_models = [m for m in available_models if 'vl' in m.lower() or 'vision' in m.lower() or 'llava' in m.lower()]
            if vision_models:
                use_ai_ocr = True
                vision_model = vision_models[0]
                ai_base_url = "http://localhost:11434"
                print(Fore.GREEN + f"✓ AI OCR enabled using Ollama vision model: {vision_model}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.YELLOW + f"Ollama not accessible: {e}" + Style.RESET_ALL)
    
    # Try LM Studio if Ollama not found
    if not use_ai_ocr:
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=3)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [m.get('id') or m.get('model') for m in models_data.get('data', [])]
                print(Fore.CYAN + f"LM Studio models found: {', '.join(available_models) if available_models else 'None'}" + Style.RESET_ALL)
                vision_models = [m for m in available_models if 'vl' in m.lower() or 'vision' in m.lower() or 'qwen' in m.lower()]
                if vision_models:
                    use_ai_ocr = True
                    vision_model = vision_models[0]
                    ai_base_url = "http://localhost:1234"
                    print(Fore.GREEN + f"✓ AI OCR enabled using LM Studio vision model: {vision_model}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.YELLOW + f"LM Studio not accessible: {e}" + Style.RESET_ALL)
    
    if not use_ai_ocr:
        print(Fore.RED + "\n⚠ NO AI VISION MODELS DETECTED!" + Style.RESET_ALL)
        print(Fore.YELLOW + "Please ensure you have a vision model running:" + Style.RESET_ALL)
        print(Fore.YELLOW + "  • Ollama (port 11434): Models with 'vl', 'vision', or 'llava' in name" + Style.RESET_ALL)
        print(Fore.YELLOW + "  • LM Studio (port 1234): Models with 'vl', 'vision', or 'qwen' in name" + Style.RESET_ALL)
        print(Fore.YELLOW + "\nExamples: qwen2.5-vl-3b, qwen3-vl:8b, llava, etc." + Style.RESET_ALL)
        print(Fore.CYAN + "\nFalling back to traditional Tesseract OCR (lower quality)." + Style.RESET_ALL)
        
        proceed = input(Fore.YELLOW + "\nContinue with Tesseract OCR? (y/n): " + Style.RESET_ALL).strip().lower()
        if proceed != "y":
            print("Returning to main menu.")
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
    direct_images = glob.glob(os.path.join(ocr_root, '*.png')) + \
                    glob.glob(os.path.join(ocr_root, '*.PNG')) + \
                    glob.glob(os.path.join(ocr_root, '*.jpg')) + \
                    glob.glob(os.path.join(ocr_root, '*.JPG')) + \
                    glob.glob(os.path.join(ocr_root, '*.jpeg')) + \
                    glob.glob(os.path.join(ocr_root, '*.JPEG')) + \
                    glob.glob(os.path.join(ocr_root, '*.bmp')) + \
                    glob.glob(os.path.join(ocr_root, '*.BMP')) + \
                    glob.glob(os.path.join(ocr_root, '*.heic')) + \
                    glob.glob(os.path.join(ocr_root, '*.HEIC')) + \
                    glob.glob(os.path.join(ocr_root, '*.heif')) + \
                    glob.glob(os.path.join(ocr_root, '*.HEIF'))
    
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
        image_files = glob.glob(os.path.join(chosen_folder, '*.png')) + \
                     glob.glob(os.path.join(chosen_folder, '*.PNG')) + \
                     glob.glob(os.path.join(chosen_folder, '*.jpg')) + \
                     glob.glob(os.path.join(chosen_folder, '*.JPG')) + \
                     glob.glob(os.path.join(chosen_folder, '*.jpeg')) + \
                     glob.glob(os.path.join(chosen_folder, '*.JPEG')) + \
                     glob.glob(os.path.join(chosen_folder, '*.bmp')) + \
                     glob.glob(os.path.join(chosen_folder, '*.BMP')) + \
                     glob.glob(os.path.join(chosen_folder, '*.heic')) + \
                     glob.glob(os.path.join(chosen_folder, '*.HEIC')) + \
                     glob.glob(os.path.join(chosen_folder, '*.heif')) + \
                     glob.glob(os.path.join(chosen_folder, '*.HEIF'))
        if not image_files:
            print(Fore.RED + "No image files found in selected folder." + Style.RESET_ALL)
            return
        image_files.sort()  # Sort files by filename
        folder_name = os.path.basename(chosen_folder) if chosen_folder != ocr_root else "OCR root"
        print(f"\nProcessing {len(image_files)} images in '{folder_name}' (sorted by filename)...")
        print(Fore.CYAN + f"Your system: 12 cores, 32GB RAM, RTX 3050 6GB" + Style.RESET_ALL)
        print(Fore.CYAN + f"Estimated capacity: 50-100+ images per batch" + Style.RESET_ALL)

        keywords_input = input(
            Fore.YELLOW
            + "Enter keyword(s) to remove from OCR results (comma-separated, or press Enter to skip): "
            + Style.RESET_ALL
        ).strip()
        keywords_to_remove = [k.strip() for k in keywords_input.split(",") if k.strip()]
        if keywords_to_remove:
            print(
                Fore.CYAN
                + f"Will remove {len(keywords_to_remove)} keyword(s) from each image OCR result before saving."
                + Style.RESET_ALL
            )

        ocr_txt_files = []
        
        for idx, img_path in enumerate(image_files, 1):
            print(Fore.MAGENTA + f"\n[{idx}/{len(image_files)}] Processing: {os.path.basename(img_path)}" + Style.RESET_ALL)
            try:
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
                
                # Convert to RGB first (in case of RGBA), then to grayscale
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                img = img.convert('L')
                
                # Advanced preprocessing for better OCR
                from PIL import ImageEnhance, ImageFilter, ImageOps
                
                # Resize if image is too small (upscale for better OCR)
                width, height = img.size
                if width < 1800 or height < 1800:
                    scale = max(1800 / width, 1800 / height)
                    new_size = (int(width * scale), int(height * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2.5)
                
                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(2.0)
                
                # Apply adaptive thresholding for better text extraction
                # This binarizes the image (black text on white background)
                img = ImageOps.autocontrast(img, cutoff=2)
                
                # Perform OCR based on available method
                print(Fore.CYAN + f"Running OCR on {os.path.basename(img_path)}..." + Style.RESET_ALL)
                
                if use_ai_ocr:
                        # Use AI vision model for OCR
                        try:
                            import base64
                            from io import BytesIO
                            
                            # Convert image to base64 for API
                            buffer = BytesIO()
                            # Convert back to RGB for JPEG encoding
                            img_rgb = img.convert('RGB')
                            img_rgb.save(buffer, format='JPEG', quality=95)
                            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            
                            # Determine which API to use based on where we found the model
                            vision_response = None
                            
                            # Try Ollama format first (port 11434)
                            try:
                                ollama_response = requests.post(
                                    "http://localhost:11434/api/generate",
                                    json={
                                        "model": vision_model,
                                        "prompt": "Extract all text from this image. Provide only the text content, maintaining the original formatting and structure. Do not add any commentary or descriptions.",
                                        "images": [img_base64],
                                        "stream": False
                                    },
                                    timeout=120
                                )
                                
                                if ollama_response.status_code == 200:
                                    result = ollama_response.json()
                                    best_text = result.get('response', '')
                                    print(Fore.GREEN + f"✓ AI OCR completed (Ollama)" + Style.RESET_ALL)
                                    vision_response = True
                            except:
                                pass
                            
                            # If Ollama failed, try LM Studio format (port 1234)
                            if not vision_response:
                                try:
                                    lmstudio_response = requests.post(
                                        "http://localhost:1234/v1/chat/completions",
                                        json={
                                            "model": vision_model,
                                            "messages": [
                                                {
                                                    "role": "user",
                                                    "content": [
                                                        {
                                                            "type": "text",
                                                            "text": "Extract all text from this image. Provide only the text content, maintaining the original formatting and structure. Do not add any commentary or descriptions."
                                                        },
                                                        {
                                                            "type": "image_url",
                                                            "image_url": {
                                                                "url": f"data:image/jpeg;base64,{img_base64}"
                                                            }
                                                        }
                                                    ]
                                                }
                                            ],
                                            "max_tokens": 2000,
                                            "temperature": 0.1
                                        },
                                        timeout=120
                                    )
                                    
                                    if lmstudio_response.status_code == 200:
                                        result = lmstudio_response.json()
                                        best_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                                        print(Fore.GREEN + f"✓ AI OCR completed (LM Studio)" + Style.RESET_ALL)
                                        vision_response = True
                                except Exception as e:
                                    pass
                            
                            if not vision_response:
                                print(Fore.YELLOW + f"⚠ AI OCR failed, falling back to Tesseract" + Style.RESET_ALL)
                                best_text = pytesseract.image_to_string(img, config='--psm 3 --oem 1')
                        except Exception as e:
                            print(Fore.YELLOW + f"⚠ AI OCR error: {e}, using Tesseract" + Style.RESET_ALL)
                            best_text = pytesseract.image_to_string(img, config='--psm 3 --oem 1')
                else:
                    # Use traditional Tesseract OCR with PSM 3
                    best_text = pytesseract.image_to_string(img, config='--psm 3 --oem 1')

                if keywords_to_remove:
                    best_text = remove_keywords_from_text(best_text, keywords_to_remove)
                
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(best_text)
                ocr_txt_files.append(txt_path)
                print(Fore.GREEN + f"Saved OCR text to: {txt_path}" + Style.RESET_ALL)
                
                # Clear memory after processing each image
                del img
                if 'img_rgb' in locals():
                    del img_rgb
                if 'buffer' in locals():
                    del buffer
                if 'img_base64' in locals():
                    del img_base64
                import gc
                gc.collect()
                
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
            
            # Get AI summary for verification
            verify = input(Fore.YELLOW + f"\nWould you like AI to verify the combined text captured all {len(image_files)} images correctly? (y/n): " + Style.RESET_ALL).strip().lower()
            
            if verify == 'y':
                print(Fore.CYAN + "\nGenerating AI verification summary..." + Style.RESET_ALL)
                
                # Select backend for verification
                base_url, api_key, backend, model_name = select_backend_and_model()
                
                # Check if server is running
                if base_url:
                    if not check_server_running(base_url, backend):
                        print(Fore.RED + f"{backend} server is not running. Skipping verification." + Style.RESET_ALL)
                        verify = 'n'
                
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

                        # Initialize OpenAI client
                        if base_url:
                            client = openai.OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
                        else:
                            client = openai.OpenAI(api_key=api_key)
                        
                        # Get AI verification
                        print(Fore.CYAN + f"Using {backend} with model: {model_name}" + Style.RESET_ALL)
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that verifies OCR results for completeness and quality."},
                                {"role": "user", "content": verification_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=1000
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
                            print(Fore.YELLOW + "This might be due to:" + Style.RESET_ALL)
                            print(Fore.YELLOW + "  • LM Studio server not running properly" + Style.RESET_ALL)
                            print(Fore.YELLOW + "  • No model loaded in LM Studio" + Style.RESET_ALL)
                            print(Fore.YELLOW + "  • Model doesn't support the request format" + Style.RESET_ALL)
                        
                    except Exception as e:
                        import traceback
                        print(Fore.RED + f"\nError during AI verification: {e}" + Style.RESET_ALL)
                        print(Fore.YELLOW + "Debug info:" + Style.RESET_ALL)
                        print(Fore.YELLOW + traceback.format_exc() + Style.RESET_ALL)
            
            # Cleanup individual files
            print(Fore.YELLOW + f"\n{'='*60}" + Style.RESET_ALL)
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
                all_images = glob.glob(os.path.join(chosen_folder, '*.png')) + \
                             glob.glob(os.path.join(chosen_folder, '*.PNG')) + \
                             glob.glob(os.path.join(chosen_folder, '*.jpg')) + \
                             glob.glob(os.path.join(chosen_folder, '*.JPG')) + \
                             glob.glob(os.path.join(chosen_folder, '*.jpeg')) + \
                             glob.glob(os.path.join(chosen_folder, '*.JPEG')) + \
                             glob.glob(os.path.join(chosen_folder, '*.bmp')) + \
                             glob.glob(os.path.join(chosen_folder, '*.BMP')) + \
                             glob.glob(os.path.join(chosen_folder, '*.heic')) + \
                             glob.glob(os.path.join(chosen_folder, '*.HEIC')) + \
                             glob.glob(os.path.join(chosen_folder, '*.heif')) + \
                             glob.glob(os.path.join(chosen_folder, '*.HEIF'))
                
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
