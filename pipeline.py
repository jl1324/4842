import os
import openai
import webbrowser
from datetime import datetime
from dotenv import load_dotenv
import os

import http.server
import socketserver
import threading


def serve():
    reports_dir = os.path.abspath("reports")
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
        *args, directory=reports_dir, **kwargs
    )
    with socketserver.TCPServer(("", 8080), handler) as httpd:
        httpd.serve_forever()


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


def transcribe_audio(audio_path):
    print("\n[1/3] Transcribing audio...")
    if WHISPERX_AVAILABLE:
        model = whisperx.load_model("base", device="cpu")
        result = model.transcribe(audio_path)
        transcript = " ".join([seg["text"] for seg in result["segments"]])
    else:
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
    """Step 2: Identify CIUs using GPT-4o with full Nicholas & Brookshire (1993) rules"""
    print("\n[2/3] Identifying CIUs...")

    prompt = f"""You are a speech-language pathology assistant applying the Nicholas & Brookshire (1993) CIU scoring rules to a Cookie Theft picture description task.

The Cookie Theft picture (Goodglass et al., 2001) depicts:
- A boy standing on a stool stealing cookies from a jar in a cupboard
- The stool is tipping over / falling
- A girl standing nearby watching the boy
- A woman drying dishes at a sink
- Water overflowing from the sink onto the floor
- A kitchen setting with a window, curtains, and cupboards

---

DEFINITION:
A Correct Information Unit (CIU) is any word that is intelligible in context, accurate in relation to the Cookie Theft picture, relevant, and informative.

---

WORDS THAT COUNT AS CIUs:
- Ungrammatical words, IF they still meet the CIU criteria
- Paraphasias that would be intelligible as the target word in context (e.g. "school" for "stool")
- Only the FINAL attempt in a series of attempts to correct a sound or word error
- Informal terms that convey picture information (e.g. "yup", "nope", "uh-huh", "un-uh")
- Embellishments that add to the events portrayed or express a moral
- Words expressing legitimate uncertainty
- Auxiliary verbs and main verbs counted as two separate CIUs
- Contractions counted as two separate CIUs
- Each word separated by hyphens counted as a separate CIU

---

WORDS THAT DO NOT COUNT AS CIUs:
- The word "and" — NEVER a CIU
- Unintelligible or irrelevant words
- Grammatically incorrect words that cause misunderstanding or uncertainty
- Repeated attempts to produce a word (only the final attempt counts)
- Dead ends, false starts, or revisions that are uninformative
- Repetition of words or ideas that don't add new information and aren't needed for cohesion or grammar
- First use of a pronoun before the antecedent has been established
- Vague or nonspecific words that aren't needed for grammatical completeness (when a more specific word could have been used)
- Conjunctions, qualifiers, and modifiers used indiscriminately as fillers: "so", "then", "I think that", "it looks like", "apparently", "of course", "sort of"
- Filler words, interjections, and tag questions: "um", "uh", "you know", "like", "right?"
- Commentary about the task, the patient's own performance, or personal experience

---

CONFIDENCE LABELS:
For each word, assign one of the following labels based on how certain you are it is a CIU:

- CONFIDENT     → Clearly a CIU with no ambiguity
- MEDIUM-CONFIDENCE → Could be a CIU, but requires clinical judgement (e.g. borderline relevance, mild ambiguity)
- NOT-CONFIDENT → Clearly not a CIU (filler, repetition, irrelevant, false start, etc.)

---

INSTRUCTIONS:
- Label every word in the transcript below.
- Do NOT include punctuation marks in your output — skip them entirely.
- Preserve sentence structure by adding a blank line between sentences.
- Return one word per line in this exact format:

word | LABEL

where LABEL is one of: CONFIDENT, MEDIUM-CONFIDENCE, NOT-CONFIDENT

Add a blank line between each sentence.

---

Transcript:
{transcript}"""

    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )

    print("      Done!")
    return response.choices[0].message.content


def generate_html(transcript, ciu_output, audio_filename):
    """Step 3: Generate colour-coded interactive HTML output"""

    # Map GPT labels to CSS classes
    label_to_class = {
        "CONFIDENT": "confident",
        "MEDIUM-CONFIDENCE": "medium-confidence",
        "NOT-CONFIDENT": "not-confident",
        "REPEATED": "repeated",
    }

    # Parse output — split into sentences by blank lines
    sentence_blocks = ciu_output.strip().split("\n\n")

    all_sentences_html = []
    confident_count = 0
    medium_count = 0
    not_ciu_count = 0
    repeated_count = 0
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
            label = parts[1].strip().upper()

            if word_index > 0 and word.lower() == prev_word.lower():
                label = "REPEATED"
            prev_word = word

            css_class = label_to_class.get(label, "not-confident")

            total_count += 1
            if css_class == "confident":
                confident_count += 1
            elif css_class == "medium-confidence":
                medium_count += 1
            elif css_class == "not-confident":
                not_ciu_count += 1
            elif css_class == "repeated":
                repeated_count += 1

            word_spans.append(
                f'<span class="{css_class}" id="word-{word_index}" onclick="toggleWord({word_index})">{word}</span>'
            )
            word_index += 1

        if word_spans:
            all_sentences_html.append(f'<p class="sentence">{" ".join(word_spans)}</p>')

    ciu_percentage = (confident_count / total_count * 100) if total_count > 0 else 0

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
    html = html.replace("{{CIU_COUNT}}", str(confident_count))
    html = html.replace("{{CIU_PERCENT}}", f"{ciu_percentage:.1f}")
    html = html.replace("{{NOT_CIU_COUNT}}", str(not_ciu_count))
    html = html.replace("{{REPEATED_COUNT}}", str(repeated_count))
    html = html.replace("{{PENDING_COUNT}}", str(medium_count))
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

    threading.Thread(target=serve, daemon=True).start()
    print(
        f"\n  Open in your browser: http://35.189.169.32:8080/{os.path.basename(html_file)}"
    )
    input("  Press Enter to exit...")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_pipeline()
