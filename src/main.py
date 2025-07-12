import os

# Always use the ffmpeg in the project directory
ffmpeg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ffmpeg', 'bin', 'ffmpeg.exe'))
os.environ["FFMPEG_BINARY"] = ffmpeg_path
os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ["PATH"]

# print("Using ffmpeg at:", ffmpeg_path)

import yt_dlp
import whisper
import torch
import openai
import subprocess
from colorama import Fore, Style, init

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

def download_audio(youtube_url, output_path="audio"):
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

def transcribe_audio(audio_file, transcript_dir="transcripts"):
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
    if not os.path.exists(audio_dir):
        print(Fore.RED + "No audio directory found." + Style.RESET_ALL)
        return []
    return [f for f in os.listdir(audio_dir) if f.lower().endswith(('.mp3', '.wav', '.m4a'))]

def list_transcript_files(transcript_dir="transcripts"):
    transcript_path = os.path.join(os.path.dirname(__file__), transcript_dir)
    if not os.path.exists(transcript_path):
        print(Fore.RED + "No transcripts directory found." + Style.RESET_ALL)
        return []
    return [f for f in os.listdir(transcript_path) if f.lower().endswith('.txt')]

def chat_session(model_name):
    print(Fore.MAGENTA + "Starting chat session. Type 'exit' to quit." + Style.RESET_ALL)

    system_prompt = input(Fore.YELLOW + "Enter a system prompt (or press Enter to skip): " + Style.RESET_ALL).strip()

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

def main():
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
                timeout=5
            )
            client.models.list()
        else:
            client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(Fore.RED + f"\nCould not connect to local backend ({backend}). Falling back to OpenAI Cloud." + Style.RESET_ALL)
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        backend = "OpenAI Cloud"
        default_model = "gpt-4o"

    print("\n")
    user_input = input(Fore.YELLOW + f"Enter the model name to use (or press Enter to use default: {default_model}): " + Style.RESET_ALL).strip()
    print("\n")
    model_name = user_input if user_input else default_model
    print(Fore.GREEN + f"Using model: {model_name}" + Style.RESET_ALL)

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

    while True:
        print(Fore.CYAN + "\nChoose an option:" + Style.RESET_ALL)
        print("1. Download a YouTube file for processing")
        print("2. Select an audio file to transcribe")
        print("3. Start a chat session")
        print("4. Summarize a transcript file")
        print("5. Exit")
        choice = input(Fore.YELLOW + "Enter your choice (1/2/3/4/5): " + Style.RESET_ALL).strip()

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
            chat_session(model_name)

        elif choice == "4":
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

        elif choice == "5":
            print(Fore.CYAN + "Goodbye!" + Style.RESET_ALL)
            break

        else:
            print(Fore.RED + "Invalid choice. Please enter a valid option." + Style.RESET_ALL)

if __name__ == "__main__":
    main()

