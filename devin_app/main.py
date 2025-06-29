import asyncio
import datetime
import threading
import logging
from colorama import Fore
from groq import AsyncGroq, Groq
import google.generativeai as genai

from .vision import initialize_webcam, display_live_feed, capture_image_from_webcam
from .audio import record_audio, transcribe_audio, speak
from .history import load_history, save_history
from .config import SYSTEM_INSTRUCTION, SAFETY_SETTINGS, GENERATION_CONFIG


def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_model():
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG,
        system_instruction=SYSTEM_INSTRUCTION,
    )


async def process_input(event: asyncio.Event, image_data: dict, conversation_history, groq_client) -> str:
    if not image_data:
        logging.error("No image data provided.")
        return "No image captured..."

    frames = []
    stop_signal = threading.Event()
    loop = asyncio.get_running_loop()
    current_time = get_current_time()

    audio_data = await asyncio.to_thread(record_audio, frames, stop_signal, event, loop)
    transcription = await transcribe_audio(audio_data)
    logging.info("Transcription: %s", transcription)

    conversation_history.append(
        {"role": "user", "parts": [{"text": transcription}, {"text": f"(timestamp:{current_time})"}]}
    )

    contents = [
        {
            "role": "user",
            "parts": [
                {"text": "[Respond to this audio as if I were hearing it with a sense of true perceptual awareness and auditory understanding of the subjective experience.]"},
                {"mime_type": "audio/wav", "data": audio_data},
                {"text": f"Current dateTime:{current_time}"},
            ],
        },
        {
            "role": "user",
            "parts": [
                {"text": "[Respond to these images as if I were seeing them with a sense of true perceptual awareness and visual understanding of the subjective experience.]"},
                {"inline_data": image_data["webcam_image"]},
                {"inline_data": image_data["screenshot"]},
                {"text": f"Current dateTime:{current_time}"},
            ],
        },
    ] + conversation_history

    model = build_model()
    try:
        result = await model.generate_content_async(contents=contents)
        description = result.text
        conversation_history.append({"role": "model", "parts": [{"text": description}]})
        save_history(conversation_history)
        return description
    except Exception as exc:
        logging.error("Error generating response: %s", exc)
        return "Error generating response."


async def capture_and_process_image() -> dict:
    image_data = await asyncio.to_thread(capture_image_from_webcam)
    if not image_data:
        logging.error("Failed to capture image. Skipping processing.")
        return {}
    logging.info("Image capture success.")
    return image_data


async def run_app():
    if not initialize_webcam():
        logging.error("Failed to initialize webcam.")
        return

    groq = Groq()
    groq_client = AsyncGroq()

    conversation_history = load_history()

    stop_signal = threading.Event()
    feed_thread = threading.Thread(target=display_live_feed, args=(stop_signal,))
    feed_thread.start()

    try:
        while True:
            logging.info(f"{Fore.LIGHTYELLOW_EX}\nStarting image capture and processing...\n{'_' * 50}\n")
            image_data = await capture_and_process_image()

            logging.info(f"{Fore.LIGHTYELLOW_EX}Listening...{'_' * 50}")
            audio_started_event = asyncio.Event()

            logging.info(f"{Fore.LIGHTYELLOW_EX}Starting audio processing...{'_' * 50}")
            task = asyncio.create_task(process_input(audio_started_event, image_data, conversation_history, groq))
            await audio_started_event.wait()
            response = await task
            print(f"Assistant: {response}")
            speak(response, groq)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        stop_signal.set()
    finally:
        feed_thread.join()
        logging.info("Program terminated.")


def main():
    asyncio.run(run_app())


if __name__ == "__main__":
    main()
