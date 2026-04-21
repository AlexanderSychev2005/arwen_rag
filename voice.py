import asyncio
import os
import re

import edge_tts
import pygame
from groq import Groq

from config import ARWEN_VOICE, GROQ_API_KEY

pygame.mixer.init()
groq_client = Groq(api_key=GROQ_API_KEY)


async def speak(text: str) -> None:
    """
    Synthesizes speech from text using Edge-TTS and plays it asynchronously
    using a producer-consumer queue to minimize latency.
    """
    cleaned_text = text.replace("\n", " ")
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", cleaned_text)
        if len(s.strip()) > 2
    ]
    audio_queue = asyncio.Queue()

    async def producer() -> None:
        for i, sentence in enumerate(sentences):
            file_name = f"arwen_chunk_{i}.mp3"
            try:
                communicate = edge_tts.Communicate(sentence, ARWEN_VOICE)
                await communicate.save(file_name)
                await audio_queue.put(file_name)
            except Exception as e:
                print(f"[TTS Error]: Failed generating audio: {e}")
        await audio_queue.put(None)

    async def consumer() -> None:
        while True:
            file_name = await audio_queue.get()
            if file_name is None:
                break
            try:
                pygame.mixer.music.load(file_name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.05)
                pygame.mixer.music.unload()
                os.remove(file_name)
            except Exception as e:
                print(f"[Playback Error]: {e}")

    await asyncio.gather(producer(), consumer())


def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes the recorded audio file using Groq's fast Whisper API.
    """
    with open(audio_file_path, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file_path, file.read()),
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en",
        )
    return transcription.strip()
