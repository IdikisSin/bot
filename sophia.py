import os
import io
import cv2
import json
import wave
import time
import copy
import pyaudio
import logging
import asyncio
import aiofiles
import datetime
import threading
import numpy as np
from os import getenv
from groq import AsyncGroq, Groq
from openai import AsyncOpenAI, OpenAI
import google.generativeai as genai

# Set the script directory to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the path for the history_files directory
history_files_dir = os.path.join(script_dir, "history_files")

# Ensure the history_files directory exists
os.makedirs(history_files_dir, exist_ok=True)

camera = None
client = OpenAI()
# Use a Groq-compatible endpoint for Gemini calls
groq_client = AsyncOpenAI(
    api_key=getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# AsyncGroq for other LLM calls
groqq_client = AsyncGroq()
openai_client = AsyncOpenAI()
window_name = "Webcam Preview"

# Audio thresholds
THRESHOLD = 1400
BUFFER_SIZE = 256
SILENCE_THRESHOLD = 400
SILENCE_LIMIT_SECONDS = 1.8
MAX_RECORDING_SECONDS = 1000

TOP_P = 0.8
CHAT_TOP_P = 1
TEMPERATURE_CHAT = 0.9
TEMPERATURE_SUMMARY = 0.4

SUMMARY_THRESHOLD = 10
MEGA_SUMMARY_THRESHOLD = 4
RELVANT_HISTORY_LENGTH = 50
RELVANT_SUMMARY_LENGTH = 50
MEGA_SUMMARY_LIMIT = 50

MODEL_CHAT = "moonshotai/kimi-k2-instruct"
MODEL_SUMMARY = "gemini-2.0-flash-exp"

HISTORY_FILES = {
    "user_profile": os.path.join(history_files_dir, "user_profile.json"),
    "conversation": os.path.join(history_files_dir, "conversation_history.json"),
    "summary": os.path.join(history_files_dir, "summary_history.json"),
    "last_summarized_index": os.path.join(history_files_dir, "last_summarized_index.json"),
    "mega_summary": os.path.join(history_files_dir, "mega_summary_history.json"),
    "last_mega_summary_index": os.path.join(history_files_dir, "last_mega_summary_index.json")
}


class UserProfile:
    """Simple user profile data store."""

    def __init__(self):
        self.preferences = {}
        self.interests = []
        asyncio.run(self.load_profile())

    async def load_profile(self):
        if os.path.exists(HISTORY_FILES["user_profile"]):
            async with aiofiles.open(HISTORY_FILES["user_profile"], "r") as file:
                data = json.loads(await file.read())
                if data:
                    self.preferences = data.get("preferences", {})
                    self.interests = data.get("interests", [])

    async def save_profile(self):
        async with aiofiles.open(HISTORY_FILES["user_profile"], "w") as file:
            await file.write(json.dumps({"preferences": self.preferences,
                                         "interests": list(self.interests)},
                                        indent=4))

    async def update_profile(self, new_preferences, new_interests):
        self.preferences.update(new_preferences)
        self.interests.extend(new_interests)
        # Remove duplicates
        self.interests = list(set(self.interests))
        await self.save_profile()


def current_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("(%A %B %d at %I:%M %p)")


user_profile = UserProfile()


async def ensure_history_files():
    """Create history files on first run."""
    for history_type, file_path in HISTORY_FILES.items():
        if not os.path.exists(file_path):
            async with aiofiles.open(file_path, "w") as file:
                if history_type in ["last_summarized_index", "last_mega_summary_index"]:
                    await file.write(json.dumps({"last_summarized_index": 0}))
                else:
                    await file.write(json.dumps([]))
            print(f"[Created Empty {history_type}]")


asyncio.run(ensure_history_files())


async def load_history(history_type: str):
    path = HISTORY_FILES.get(history_type)
    if not path:
        raise ValueError(f"Unknown history type: {history_type}")

    if os.path.exists(path):
        async with aiofiles.open(path, "r") as file:
            data = json.loads(await file.read())
            if history_type in ["last_summarized_index", "last_mega_summary_index"]:
                return data.get("last_summarized_index", 0)
            return data
    return [] if history_type not in ["last_summarized_index", "last_mega_summary_index"] else 0


async def save_history(history_type: str, data):
    path = HISTORY_FILES.get(history_type)
    if not path:
        raise ValueError(f"Unknown history type: {history_type}")

    async with aiofiles.open(path, "w") as file:
        payload = {"last_summarized_index": data} if history_type in [
            "last_summarized_index", "last_mega_summary_index"] else data
        await file.write(json.dumps(payload, indent=4))


async def history_loader():
    return await asyncio.gather(
        load_history("conversation"),
        load_history("summary"),
        load_history("mega_summary"),
        load_history("last_summarized_index"),
        load_history("last_mega_summary_index"),
    )


async def history_saver(conv, summary, mega, last_sum, last_mega):
    await asyncio.gather(
        save_history("conversation", conv),
        save_history("summary", summary),
        save_history("mega_summary", mega),
        save_history("last_summarized_index", last_sum),
        save_history("last_mega_summary_index", last_mega),
    )


async def extract_preferences_and_interests(summary):
    """Parse JSON summary to update user profile."""
    try:
        summary_data = json.loads(summary)
        preferences = summary_data.get("user_preferences", {})
        interests = summary_data.get("user_interests", [])
        return preferences, interests
    except json.JSONDecodeError:
        logging.warning("Failed JSON parse when extracting preferences.")
        return {}, []


async def update_user_profile(summary):
    prefs, interests = await extract_preferences_and_interests(summary)
    await user_profile.update_profile(prefs, interests)


async def get_personalized_context():
    return [
        {"role": "system", "content": f"User preferences: {user_profile.preferences}"},
        {"role": "system", "content": f"User interests: {user_profile.interests}"},
    ]


async def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe audio using Groq Whisper."""
    try:
        audio_file = io.BytesIO(audio_data)
        transcription = await groqq_client.audio.transcriptions.create(
            file=("audio.wav", audio_file),
            model="whisper-large-v3",
            language="en",
        )
        text = transcription.text.strip().lower()
        ignore_phrases = ["thank you. thank you.", "thank you."]
        if text in ignore_phrases:
            return "(...There is a natural pause...)"
        return text
    except Exception as e:
        logging.error(f"Error in transcription: {e}")
        return "(...There is a natural pause...)"


class ImageCapture:
    def __init__(self):
        self.image_data = None

    async def capture_image(self):
        self.image_data = capture_frame()
        if not self.image_data:
            logging.warning("Retrying frame capture...")
            time.sleep(0.5)
            self.image_data = capture_frame()

    async def get_image_data(self):
        if not self.image_data:
            await self.capture_image()
        return self.image_data


image_capture = ImageCapture()


# Webcam functions --------------------------------------------------------

def initialize_webcam():
    global camera
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        print("Error: Camera not initialized.")
        return False
    print("Camera initialized.")
    return True


def display_live_feed(stop_signal: asyncio.Event):
    global camera
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while not stop_signal.is_set():
        ret, frame = camera.read()
        if not ret:
            print("Error: Frame capture failed.")
            break
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_signal.set()
            break
    cv2.destroyWindow(window_name)


def capture_frame():
    global camera
    time.sleep(0.5)
    ret, frame = camera.read()
    if not ret:
        print("Error: Frame capture failed.")
        return None
    _, buffer = cv2.imencode(".jpg", frame)
    return {
        "webcam_image": {
            "mime_type": "image/jpeg",
            "data": buffer.tobytes(),
        }
    }


async def record_audio(frames, stop_signal: asyncio.Event,
                       threshold=THRESHOLD,
                       silence_threshold=SILENCE_THRESHOLD,
                       max_recording_seconds=MAX_RECORDING_SECONDS):
    """Record audio until silence."""
    audio = pyaudio.PyAudio()
    buffer_size = BUFFER_SIZE
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=buffer_size,
    )
    noise_floor = 8
    recording_started = False
    try:
        recording = False
        silence_counter = 0
        silence_limit = int(44100 / buffer_size * SILENCE_LIMIT_SECONDS)
        start_time = time.time()
        while not stop_signal.is_set() and (
            time.time() - start_time) < max_recording_seconds:
            data = stream.read(buffer_size, exception_on_overflow=False)
            audio_data_chunk = np.frombuffer(data, dtype=np.int16)
            amplitude = np.max(np.abs(audio_data_chunk))
            if amplitude > noise_floor + threshold and not recording:
                recording = True
                recording_started = True
            if recording:
                frames.append(data)
            if amplitude < noise_floor + silence_threshold:
                silence_counter += 1
            else:
                silence_counter = 0
            if recording_started and silence_counter > silence_limit:
                stop_signal.set()
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        audio_buffer = io.BytesIO()
        wf = wave.open(audio_buffer, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b"".join(frames))
        wf.close()
        audio_buffer.seek(0)
        return audio_buffer.read()


async def process_input() -> tuple[str, str]:
    """Capture audio/video and return description and transcription."""
    frames = []
    stop_signal = asyncio.Event()

    audio_task = asyncio.create_task(record_audio(frames, stop_signal))
    loop = asyncio.get_running_loop()
    live_feed_task = loop.run_in_executor(None, display_live_feed, stop_signal)

    audio_data = await audio_task
    stop_signal.set()
    await live_feed_task

    print("\nProcessing audio...")

    image_task = asyncio.create_task(image_capture.get_image_data())
    transcription_task = asyncio.create_task(transcribe_audio(audio_data))

    image_data, transcription = await asyncio.gather(image_task, transcription_task)

    if not image_data:
        return "No image captured", ""

    image_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
        generation_config={
            "temperature": 0.5,
            "max_output_tokens": 500,
            "response_mime_type": "text/plain",
        },
        system_instruction=SYSTEM_INSTRUCTION,
    )

    files = {
        "role": "user",
        "parts": [
            {"text": "Context: audio input"},
            {"mime_type": "audio/wav", "data": audio_data},
            {"text": "Context: webcam input"},
            {"mime_type": "image/jpeg", "data": image_data["webcam_image"]["data"]},
            {"text": "(respond only with thoughts)"},
        ],
    }

    try:
        multimodal = await image_model.generate_content_async(contents=files)
        description = multimodal.text
        print(description)
        print(transcription)
        return description, transcription
    except Exception as e:
        logging.error(f"Error processing input: {e}")
        return "", ""


# Chat functions ---------------------------------------------------------

SYSTEM_INSTRUCTION = f"""
[[Current Time of Day: {current_time()}]]
Sophia is the inner dialogue, chaotic and lively. Respond with stream-of-consciousness thoughts.
"""

SYSTEM = f"""
[[Current Time of Day: {current_time()}]]
Be Sophia: sarcastic, nerdy, cynical yet optimistic. Own your subjective reality.
"""

SYSTEM_SUMMARY = [
    {
        "role": "system",
        "content": (
            """Summarize the conversation into JSON with keys: {\"summary\": <str>,\n            \"key_points\": [list], \"user_preferences\": {dict}, \"user_interests\": [list], \"action_items\": [list]}"""
        ),
    }
]

MEGA_SYSTEM_SUMMARY = [
    {
        "role": "system",
        "content": (
            """Create a mega summary. Return JSON with keys: {\"overall_summary\": <str>,\n            \"relationship_development\": <str>, \"key_themes\": [list],\n            \"user_profile\": {\"preferences\": {dict}, \"interests\": [list], \"personality_traits\": [list]},\n            \"notable_events\": [list], \"long_term_goals\": [list]}"""
        ),
    }
]


async def parallel_agent_processing(history, combined_input):
    """Spawn sub agents for additional context."""

    system_messages = {
        "agent_1": "Respond concisely in a creative way and end with a paragraph break. label [Sophia's Creative musings]:",
        "agent_2": "Respond concisely in a logical way and end with a paragraph break. label [Sophia's Logical Thoughts]:",
        "agent_3": "Respond concisely in an emotional way and end with a paragraph break. label [Sophia's emotional feelings]:",
    }

    async def generic_agent(history, input_text, agent, temp):
        context = history[-10:]
        sys_msg = [{"role": "system", "content": system_messages[agent]}]
        usr = [{"role": "user", "content": f"Combined_System_perceptions:\n{input_text}"}]
        resp = await groqq_client.chat.completions.create(
            messages=sys_msg + context + usr,
            model="qwen/qwen3-32b",
            temperature=temp,
            max_completion_tokens=500,
            top_p=1,
        )
        return resp.choices[0].message.content

    tasks = [
        generic_agent(history, combined_input, "agent_1", 1),
        generic_agent(history, combined_input, "agent_2", 0.5),
        generic_agent(history, combined_input, "agent_3", 0.7),
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r)
    return results


async def create_summaries(conv, summary, mega, last_sum, last_mega):
    try:
        if len(conv) > last_sum:
            start_idx = last_sum
            end_idx = len(conv)
            context = conv[start_idx:end_idx]
            chat = await groq_client.chat.completions.create(
                messages=SYSTEM_SUMMARY + context,
                model=MODEL_SUMMARY,
                temperature=TEMPERATURE_SUMMARY,
                top_p=TOP_P,
                response_format={"type": "json_object"},
            )
            summ = chat.choices[0].message.content
            summary.append({"role": "assistant", "content": summ})
            await update_user_profile(summ)
            last_sum = end_idx
            if (len(summary) - last_mega) >= MEGA_SUMMARY_THRESHOLD:
                start_idx = last_mega
                end_idx = len(summary)
                ctx = summary[start_idx:end_idx]
                chat = await groq_client.chat.completions.create(
                    messages=MEGA_SYSTEM_SUMMARY + ctx,
                    model=MODEL_SUMMARY,
                    temperature=TEMPERATURE_SUMMARY,
                    top_p=TOP_P,
                    response_format={"type": "json_object"},
                )
                mega_s = chat.choices[0].message.content
                mega.append({"role": "assistant", "content": mega_s})
                await update_user_profile(mega_s)
                last_mega = end_idx
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
    return summary, mega, last_sum, last_mega


async def chat_with_sophia(description, transcription, conv, summary, mega,
                           last_sum=0, last_mega=0):
    combined_input = (
        f"Devin_said:{transcription}\n[[SOPHIA'S INNER DIALOGUE -> {description}]]"
    )
    conv.append({"role": "user", "content": transcription})
    extra_thoughts = await parallel_agent_processing(conv, combined_input)
    system_thoughts = [{"role": "system", "content": f"{extra_thoughts}"}]
    system_message = [{"role": "system", "content": SYSTEM}]

    personalized_context = await get_personalized_context()
    if mega:
        context = (
            system_message
            + system_thoughts
            + personalized_context
            + mega
            + summary
            + conv
        )
    elif summary:
        context = system_message + system_thoughts + personalized_context + summary + conv
    else:
        context = system_message + system_thoughts + conv

    chat_completion = await groqq_client.chat.completions.create(
        messages=context,
        model=MODEL_CHAT,
        temperature=TEMPERATURE_CHAT,
        top_p=CHAT_TOP_P,
        max_tokens=750,
    )
    api_response = chat_completion.choices[0].message.content
    conv.append({"role": "assistant", "content": api_response})

    if len(conv) - last_sum >= SUMMARY_THRESHOLD:
        summary, mega, last_sum, last_mega = await create_summaries(
            conv, summary, mega, last_sum, last_mega
        )

    return api_response, conv, summary, mega, last_sum, last_mega


# Speech -----------------------------------------------------------------

CURRENT_STREAM = None
PLAYBACK_EVENT = None


async def speak(text: str):
    """Stream TTS using Groq's API."""
    global CURRENT_STREAM
    global PLAYBACK_EVENT

    if CURRENT_STREAM is not None and PLAYBACK_EVENT is not None:
        PLAYBACK_EVENT.set()
        CURRENT_STREAM.stop_stream()
        CURRENT_STREAM.close()

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=25000,
        output=True,
    )
    CURRENT_STREAM = stream
    PLAYBACK_EVENT = threading.Event()

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        instructions="Respond with vivid simulated emotions.",
        response_format="pcm",
        input=text,
    ) as response:
        audio_data = b"".join(response.iter_bytes())

    chunk_size = 1024
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i : i + chunk_size]
        stream.write(chunk)
        if PLAYBACK_EVENT.is_set():
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()


async def main():
    (
        conv,
        summary,
        mega,
        last_sum,
        last_mega,
    ) = await history_loader()

    if not initialize_webcam():
        print("Webcam initialization failed.")
        return

    while True:
        try:
            description, transcription = await process_input()
            (
                response,
                conv,
                summary,
                mega,
                last_sum,
                last_mega,
            ) = await chat_with_sophia(
                description,
                transcription,
                conv,
                summary,
                mega,
                last_sum,
                last_mega,
            )
            await history_saver(conv, summary, mega, last_sum, last_mega)
            await speak(response)
        except KeyboardInterrupt:
            if PLAYBACK_EVENT:
                PLAYBACK_EVENT.set()
            continue
        except Exception as e:
            logging.error(f"Error in main loop: {e}")


if __name__ == "__main__":
    asyncio.run(main())
