import os
import requests # type: ignore

# Always use the ffmpeg in the project directory
ffmpeg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ffmpeg', 'bin', 'ffmpeg.exe'))
os.environ["FFMPEG_BINARY"] = ffmpeg_path
os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ["PATH"]

# print("Using ffmpeg at:", ffmpeg_path)

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
        print("\nYou selected OpenAI Cloud.")
        api_key = input("Enter your OpenAI API key (or press Enter to use the .env/environment variable): ").strip()
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        backend = "OpenAI Cloud"
        model_name = "gpt-4o"

    print(f"Selected backend: {backend}")
    print(f"Default model set: {model_name}")

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
    ffmpeg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ffmpeg', 'bin'))
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'ffmpeg_location': ffmpeg_dir,
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
    ffmpeg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ffmpeg', 'bin'))
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'ffmpeg_location': ffmpeg_dir,
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

def chat_session(model_name):
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

    # List available models if not using the default (OpenAI Cloud)
    if backend != "OpenAI Cloud":
        try:
            print(Fore.CYAN + "\nAvailable models on this backend:" + Style.RESET_ALL)
            models = client.models.list()
            for m in models.data:
                print(" -", m.id)
            print()
        except Exception as e:
            print(Fore.RED + f"Could not list models: {e}" + Style.RESET_ALL)

    print("\n")
    user_input = input(Fore.YELLOW + f"Enter the model name to use (or press Enter to use default: {default_model}): " + Style.RESET_ALL).strip()
    print("\n")
    model_name = user_input if user_input else default_model
    print(Fore.GREEN + f"Using model: {model_name}" + Style.RESET_ALL)

    if isinstance(model_name, str):
        if "instruct" not in model_name.lower() and "chat" not in model_name.lower():
            print(Fore.YELLOW + "Warning: You are not using an 'instruct' or 'chat' model. "
                  "Responses may not follow instructions as expected." + Style.RESET_ALL)

    # Initial greeting and instructions from the assistant
    response = client.chat.completions.create(
        model="Mistral-7B-Instruct-v0.3-GGUF",
        messages=[{
            "role": "user",
            "content": (
                "You are an assistant for a YouTube audio downloader and transcription tool. "
                "Greet the user and briefly explain that you can help download YouTube audio, transcribe it, "
                "and summarize transcripts using local AI models."
            )
        }]
    )
    print(response.choices[0].message.content)

    if backend in ["LM Studio", "Ollama"]:
        if not check_server_running(base_url, backend):
            print(Fore.RED + f"\n{backend} server is not running. Please start it and try again." + Style.RESET_ALL)
            return  # or sys.exit(1)
        else:
            print(Fore.GREEN + f"{backend} server is running and ready!" + Style.RESET_ALL)

    while True:
        print(Fore.CYAN + "\nChoose an option:" + Style.RESET_ALL)
        print("1. Download a YouTube file for processing")
        print("2. Select an audio file to transcribe")
        print("3. Combine transcript files")
        print("4. Start a chat session")
        print("5. Summarize a transcript file")
        print("6. Download a SoundCloud audio file")
        print("7. Delete a file")
        print("8. Exit")
        choice = input(Fore.YELLOW + "Enter your choice (1/2/3/4/5/6/7/8): " + Style.RESET_ALL).strip()

        if choice == "1":
            youtube_url = input(Fore.YELLOW + "Enter the YouTube Video URL: " + Style.RESET_ALL).strip()
            try:
                print("Downloading audio...")
                audio_file = download_audio(youtube_url)
                print(Fore.GREEN + f"Audio downloaded to: {audio_file}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)

        elif choice == "2":
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

        elif choice == "3":
            combine_transcript_files()
        elif choice == "4":
            chat_session(model_name)
        elif choice == "5":
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

        elif choice == "6":
            soundcloud_url = input(Fore.YELLOW + "Enter the SoundCloud URL: " + Style.RESET_ALL).strip()
            try:
                print("Downloading SoundCloud audio...")
                audio_file = download_soundcloud_audio(soundcloud_url)
                print(Fore.GREEN + f"Audio downloaded to: {audio_file}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
        elif choice == "7":
            delete_file_menu()
        elif choice == "8":
            print(Fore.CYAN + "Goodbye!" + Style.RESET_ALL)
            break
        else:
            print(Fore.RED + "Invalid choice. Please enter a valid option." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
