"""Audio recording and transcription utilities."""
import asyncio
import io
import logging
import time
import wave
import threading
import numpy as np
import pyaudio
import google.generativeai as genai
from .config import (
    RATE,
    CHANNELS,
    FORMAT,
    BUFFER_SIZE,
    THRESHOLD,
    SILENCE_THRESHOLD,
    SILENCE_LIMIT_SECONDS,
    MAX_RECORDING_SECONDS,
    NOISE_FLOOR,
    SAFETY_SETTINGS,
    GENERATION_CONFIG_1,
    SYSTEM_PROMPT,
)


def record_audio(frames, stop_signal: threading.Event, event: asyncio.Event, loop: asyncio.AbstractEventLoop) -> bytes:
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=BUFFER_SIZE,
    )

    recording_started = False
    recording = False
    silence_counter = 0
    silence_limit = int(RATE / BUFFER_SIZE * SILENCE_LIMIT_SECONDS)
    start_time = time.time()

    try:
        while not stop_signal.is_set() and (time.time() - start_time) < MAX_RECORDING_SECONDS:
            data = stream.read(BUFFER_SIZE, exception_on_overflow=False)
            audio_data_chunk = np.frombuffer(data, dtype=np.int16)
            amplitude = np.max(np.abs(audio_data_chunk))

            if amplitude > NOISE_FLOOR + THRESHOLD and not recording:
                recording = True
                recording_started = True
                logging.info("Recording started.")
                loop.call_soon_threadsafe(event.set)

            if recording:
                frames.append(data)

            if amplitude < NOISE_FLOOR + SILENCE_THRESHOLD:
                silence_counter += 1
            else:
                silence_counter = 0

            if recording_started and silence_counter > silence_limit:
                logging.info("Silence detected. Stopping recording.")
                stop_signal.set()
    except Exception as exc:
        logging.error("Error during recording: %s", exc)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    audio_buffer.seek(0)
    return audio_buffer.read()


async def transcribe_audio(audio_data: bytes) -> str:
    transcription_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG_1,
        system_instruction=SYSTEM_PROMPT,
    )
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": "[Transcribe the Speech in the provided audio. If no Audio Provided Prompt with '(a brief pause in the conversation...)]"},
                {"mime_type": "audio/wav", "data": audio_data},
            ],
        }
    ]
    try:
        result = await transcription_model.generate_content_async(contents=contents)
        return result.text
    except Exception as exc:
        logging.error("Error generating transcription: %s", exc)
        return ""


def speak(text: str, groq_client):
    logging.info("Starting TTS stream...")
    stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=50000, output=True)
    stream.start_stream()
    with groq_client.audio.speech.with_streaming_response.create(
        model="playai-tts", voice="Cheyenne-PlayAI", response_format="wav", input=text
    ) as response:
        for chunk in response.iter_bytes(chunk_size=64):
            stream.write(chunk)
    stream.stop_stream()
    stream.close()
