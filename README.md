# Devin AI Application

This repository contains a small example application that records audio, captures webcam images, and uses generative AI services to respond. The project has been organised into a Python package located in `devin_app`.

## Requirements

The code depends on a few external packages:

- `pyaudio`
- `opencv-python`
- `colorama`
- `google-generativeai`
- `groq`
- `Pillow`

You will also need valid credentials for the Google Generative AI API and Groq services.

Install dependencies with pip:

```bash
pip install pyaudio opencv-python colorama google-generativeai groq Pillow
```

## Running the app

Execute the application using Python:

```bash
python -m devin_app.main
```

The program will open your webcam, listen for audio input, and respond using text-to-speech.
