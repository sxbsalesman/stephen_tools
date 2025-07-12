import os
import yt_dlp
import whisper
import torch
import openai
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

init(autoreset=True)

def select_backend_and_model():
    print("\nChoose which LLM backend to use:")
    print("1. OpenAI Cloud (default)")
    print("2. LM Studio (local)")
    print("3. Ollama (local)")

    choice = input("Enter 1, 2, or 3 (or press Enter for OpenAI Cloud): ").strip()

    if choice == "2":
        base_url = "http://localhost:1234/v1"
        api_key = "sk-local"
        backend = "LM Studio"
        model_name = "Mistral-7B-Instruct-v0.3-GGUF"
    elif choice == "3":
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"
        backend = "Ollama"
        model_name = "mistral:instruct"
    else:
        base_url = None
        api_key = os.environ.get("OPENAI_API_KEY")
        backend = "OpenAI Cloud"
        model_name = "gpt-4o"

    print(f"Selected backend: {backend}")
    print(f"Default model set: {model_name}")
    return base_url, api_key, backend, model_name

def download_audio(youtube_url, output_path=os.path.join("src", "audio")):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
    if not os.path.isfile(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg not found at expected path: {ffmpeg_path}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': ffmpeg_path,
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)
        audio_file = os.path.splitext(filename)[0] + '.mp3'
        return audio_file

def transcribe_audio(audio_file, transcript_dir=os.path.join("src", "transcripts")):
    if not os.path.exists(transcript_dir):
        os.makedirs(transcript_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    result = model.transcribe(audio_file)

    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    transcript_file = os.path.join(transcript_dir, f"{base_name}.txt")
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    return transcript_file

def list_files(directory, ext=".txt"):
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.lower().endswith(ext)]

def chat_session(model_name):
    print(Fore.MAGENTA + "Starting chat session. Type 'exit' to quit." + Style.RESET_ALL)

    system_prompt = input(Fore.YELLOW + "Enter a system prompt (or press Enter to skip): " + Style.RESET_ALL).strip()

    summary_dir = os.path.join("src", "summaries")
    summary_files = list_files(summary_dir)
    file_content = ""
    if summary_files:
        print("\nAvailable summary files:")
        for idx, fname in enumerate(summary_files, 1):
            print(f"{idx}. {fname}")
        file_choice = input(Fore.YELLOW + "Select a summary file to include as context (or 'q' to skip): " + Style.RESET_ALL).strip()
        if file_choice.lower() != "q" and file_choice != "":
            try:
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(summary_files):
                    with open(os.path.join(summary_dir, summary_files[file_idx]), "r", encoding="utf-8") as f:
                        file_content = f.read()
                    print(Fore.GREEN + f"Loaded file: {summary_files[file_idx]}" + Style.RESET_ALL)
                else:
                    print(Fore.RED + "Invalid selection." + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
    else:
        print(Fore.RED + "No summary files found." + Style.RESET_ALL)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if file_content:
        messages.append({"role": "user", "content": f"Here is the summary context:\n\n{file_content}"})

    while True:
        user_input = input(Fore.YELLOW + "You: " + Style.RESET_ALL)
        if user_input.lower() == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            timeout=120
        )
        bot_message = response.choices[0].message.content
        print(Fore.GREEN + "Bot: " + bot_message + Style.RESET_ALL)
        messages.append({"role": "assistant", "content": bot_message})

def main():
    global client
    base_url, api_key, backend, default_model = select_backend_and_model()

    try:
        if base_url:
            client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=5)
            client.models.list()
        else:
            client = openai.OpenAI(api_key=api_key)
    except Exception:
        print(Fore.RED + f"\nCould not connect to local backend ({backend}). Falling back to OpenAI Cloud." + Style.RESET_ALL)
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        backend = "OpenAI Cloud"
        default_model = "gpt-4o"

    print("\n")
    user_input = input(Fore.YELLOW + f"Enter the model name to use (or press Enter to use default: {default_model}): " + Style.RESET_ALL).strip()
    model_name = user_input if user_input else default_model
    print(Fore.GREEN + f"Using model: {model_name}" + Style.RESET_ALL)

    while True:
        print(Fore.CYAN + "\nChoose an option:" + Style.RESET_ALL)
        print("1. Download a YouTube audio file")
        print("2. Transcribe an audio file")
        print("3. Start a chat session")
        print("4. Exit")
        choice = input(Fore.YELLOW + "Enter your choice (1/2/3/4): " + Style.RESET_ALL).strip()

        if choice == "1":
            url = input(Fore.YELLOW + "Enter the YouTube URL: " + Style.RESET_ALL).strip()
            try:
                audio_file = download_audio(url)
                print(Fore.GREEN + f"Downloaded audio: {audio_file}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)
        elif choice == "2":
            audio_dir = os.path.join("src", "audio")
            audio_files = list_files(audio_dir, ext=".mp3")
            if not audio_files:
                print(Fore.RED + "No audio files found." + Style.RESET_ALL)
                continue
            print("\nAvailable audio files:")
            for idx, fname in enumerate(audio_files, 1):
                print(f"{idx}. {fname}")
            file_choice = input(Fore.YELLOW + "Select a file number to transcribe (or 'q' to return): " + Style.RESET_ALL).strip()
            if file_choice.lower() == "q":
                continue
            try:
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(audio_files):
                    audio_path = os.path.join(audio_dir, audio_files[file_idx])
                    transcript_file = transcribe_audio(audio_path)
                    print(Fore.GREEN + f"Transcript saved: {transcript_file}" + Style.RESET_ALL)
                else:
                    print(Fore.RED + "Invalid selection." + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)
        elif choice == "3":
            chat_session(model_name)
        elif choice == "4":
            print(Fore.CYAN + "Goodbye!" + Style.RESET_ALL)
            break
        else:
            print(Fore.RED + "Invalid choice." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
