import pytesseract # type: ignore
from PIL import Image # type: ignore
import glob
import os
import requests # type: ignore
import shutil
import platform

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

import yt_dlp # type: ignore
import whisper # type: ignore
import torch # type: ignore
import openai # type: ignore
import subprocess
from colorama import Fore, Style, init # type: ignore

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
    Returns True if running, False otherwise.
    """
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
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
    print("\n" + "="*60)
    print("         YouTube Downloader and Transcription Tools")
    print("="*60 + "\n")

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
            print(Fore.GREEN + f"Combined OCR text saved to: {combined_path}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
