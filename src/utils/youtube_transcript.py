def get_transcript(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def parse_transcript(transcript):
    parsed_text = []
    for entry in transcript:
        parsed_text.append(entry['text'])
    return ' '.join(parsed_text)