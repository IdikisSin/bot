# Sophia

This repository contains **sophia.py**, a single script that implements a
multimodal conversation agent. It uses various APIs (Groq, OpenAI, Gemini) for
speech recognition, language generation and text‑to‑speech output. Webcam and
microphone input are used to capture user interactions.

Running the script requires a Python environment with the dependencies listed at
the top of `sophia.py` and appropriate API credentials.

```bash
python sophia.py
```

The script will open the webcam, listen for speech and respond using TTS while
building conversation summaries for long‑term context.
