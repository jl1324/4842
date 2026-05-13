import os
import webbrowser
from datetime import datetime
from dotenv import load_dotenv
import os

# Check available transcription backends
try:
    import whisperx

    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

if not WHISPERX_AVAILABLE and not OPENAI_AVAILABLE:
    raise EnvironmentError(
        "No transcription backend found. Install whisperx or openai."
    )


load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Create output folders if they don't exist
os.makedirs("reports", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)


def get_audio_path():
    """Prompt user to input audio file path"""
    print("\n" + "=" * 50)
    print("  CIU Analysis Pipeline")
    print("=" * 50)
    print("\nEnter the path to your audio file.")
    print("Format: C:\\Users\\name\\audio.wav  or  ./audio.wav")
    print()

    while True:
        path = input("Audio file path: ").strip().strip('"')
        if os.path.exists(path):
            return path
        else:
            print(f"  File not found: {path}")
            print("  Please check the path and try again.\n")


def transcribe_audio(audio_path):
    print("\n[1/3] Transcribing audio...")
    if WHISPERX_AVAILABLE:
        # WhisperX transcription
        model = whisperx.load_model("base", device="cpu")
        result = model.transcribe(audio_path)
        transcript = " ".join([seg["text"] for seg in result["segments"]])
    else:
        # Fallback to OpenAI Whisper API
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open(audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
        transcript = result.text
    print("      Done!")
    return transcript


def format_transcript(transcript):
    """Format transcript with one sentence per line"""
    formatted = (
        transcript.replace(". ", ".\n").replace("? ", "?\n").replace("! ", "!\n")
    )
    return formatted.strip()


def identify_cius(transcript):
    """Step 2: Identify CIUs using GPT"""
    print("\n[2/3] Identifying CIUs...")

    prompt = f"""You are a speech-language pathology assistant applying Nicholas & Brookshire (1993) CIU scoring rules to a Cookie Theft picture description task.

The Cookie Theft picture (Goodglass et al., 2001) depicts:
- A boy standing on a stool stealing cookies from a jar in a cupboard
- The stool is falling over
- A girl standing nearby watching the boy
- A woman drying dishes at a sink
- Water overflowing from the sink onto the floor
- A kitchen setting with a window, curtains, and cupboards

A Correct Information Unit (CIU) is any word that is:
- Intelligible in context
- Accurate in relation to the Cookie Theft picture
- Relevant and informative (not filler, repetition, or error)

NOT a CIU:
- Fillers: um, uh, you know, like
- Repeated words (only first instance counts)
- False starts and revisions
- Inaccurate or irrelevant content not relating to the picture
- Neologisms or unintelligible words
- Punctuation marks

For each word in the transcript below, label it as CIU or NON-CIU.
Ignore punctuation marks entirely — do not include them in your output.
Preserve the sentence structure by adding a blank line between sentences.

Return your response as a list of pairs, one per line, in this exact format:
word | CIU
word | NON-CIU

Add a blank line between each sentence to preserve structure.

Transcript:
{transcript}"""

    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )

    print("      Done!")
    return response.choices[0].message.content


def generate_html(transcript, ciu_output, audio_filename):
    """Step 3: Generate colour-coded interactive HTML output"""

    # Parse output — split into sentences by blank lines
    sentence_blocks = ciu_output.strip().split("\n\n")

    all_sentences_html = []
    ciu_count = 0
    total_count = 0
    word_index = 0

    for block in sentence_blocks:
        lines = [l.strip() for l in block.strip().split("\n") if "|" in l]
        if not lines:
            continue

        word_spans = []
        for line in lines:
            parts = line.split("|")
            if len(parts) != 2:
                continue
            word = parts[0].strip()
            label = parts[1].strip()

            total_count += 1
            is_ciu = label == "CIU"
            if is_ciu:
                ciu_count += 1

            css_class = "ciu" if is_ciu else "non-ciu"
            word_spans.append(
                f'<span class="{css_class}" id="word-{word_index}" onclick="toggleWord({word_index})">{word}</span>'
            )
            word_index += 1

        if word_spans:
            all_sentences_html.append(f'<p class="sentence">{" ".join(word_spans)}</p>')

    ciu_percentage = (ciu_count / total_count * 100) if total_count > 0 else 0

    # Format raw transcript
    formatted_transcript = format_transcript(transcript)
    transcript_html = "".join(
        f"<p>{line}</p>" for line in formatted_transcript.split("\n")
    )

    # Load template and replace placeholders
    with open("template.html", "r") as f:
        html = f.read()

    html = html.replace("{{AUDIO_FILENAME}}", audio_filename)
    html = html.replace("{{DATE}}", datetime.now().strftime("%d/%m/%Y %H:%M"))
    html = html.replace("{{TOTAL_COUNT}}", str(total_count))
    html = html.replace("{{CIU_COUNT}}", str(ciu_count))
    html = html.replace("{{CIU_PERCENT}}", f"{ciu_percentage:.1f}")
    html = html.replace("{{TRANSCRIPT_HTML}}", transcript_html)
    html = html.replace("{{ANNOTATION_HTML}}", "".join(all_sentences_html))

    return html


def get_input():
    """Prompt user for audio file or existing transcript"""
    print("\n" + "=" * 50)
    print("  CIU Analysis Pipeline")
    print("=" * 50)

    backend = "WhisperX" if WHISPERX_AVAILABLE else "OpenAI Whisper"
    print(f"\nTranscription backend: {backend}")
    print("\nWhat would you like to input?")
    print(f"  1. Audio file (will transcribe using {backend})")
    print("  2. Existing transcript file (skip transcription)")
    print()

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        while True:
            path = input("Audio file path: ").strip().strip('"')
            if os.path.exists(path):
                return "audio", path
            print(f"  File not found: {path}\n")

    elif choice == "2":
        while True:
            path = input("Transcript file path: ").strip().strip('"')
            if os.path.exists(path):
                return "transcript", path
            print(f"  File not found: {path}\n")
    else:
        print("Invalid choice, defaulting to audio input.")
        return get_input()


def run_pipeline():
    input_type, path = get_input()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if input_type == "audio":
        audio_filename = os.path.basename(path)
        transcript = transcribe_audio(path)
    else:
        audio_filename = os.path.basename(path)
        with open(path, "r") as f:
            transcript = f.read()
        print("\n[1/3] Transcript loaded from file!")

    # Save formatted transcript
    formatted = format_transcript(transcript)
    transcript_file = os.path.join("transcripts", f"transcript_{timestamp}.txt")
    with open(transcript_file, "w") as f:
        f.write(formatted)
    print(f"\n      Transcript saved: {transcript_file}")

    # Step 2: Identify CIUs
    ciu_output = identify_cius(transcript)

    # Step 3: Generate HTML
    print("\n[3/3] Generating report...")
    html = generate_html(transcript, ciu_output, audio_filename)

    html_file = os.path.join("reports", f"ciu_report_{timestamp}.html")
    with open(html_file, "w") as f:
        f.write(html)
    print(f"      Report saved: {html_file}")

    webbrowser.open(f"file://{os.path.abspath(html_file)}")
    print("\n  Done! Report opened in browser.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_pipeline()
