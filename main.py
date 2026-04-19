import asyncio

from brain import get_llm_response
from ears import ArwenEars
from memory import save_to_memory
from voice import speak, transcribe_audio


async def main_loop() -> None:
    """
    The main asynchronous event loop for the voice assistant.
    Handles recording, transcription, LLM processing, memory saving, and TTS playback.
    """
    print("\n[Arwen] The light of the Evenstar shines for you. I am ready.")
    ears = ArwenEars(silence_duration=1.5, vad_threshold=0.5)

    while True:
        try:
            # 1. Listen
            audio_file = ears.listen_and_record("user_input.wav")

            # 2. Transcribe
            print("[Arwen] Listening to your words...")
            user_text = transcribe_audio(audio_file)

            if len(user_text) < 2:
                continue

            print(f"You: {user_text}")

            # 3. Think
            bot_response = get_llm_response(user_text)
            print(f"Arwen: {bot_response}")

            # 4. Remember
            save_to_memory(user_text, bot_response)

            # 5. Speak
            await speak(bot_response)

        except KeyboardInterrupt:
            print("\n[Arwen] Namárië. Farewell.")
            break
        except Exception as e:
            print(f"\n[System Error]: {e}")


if __name__ == "__main__":
    asyncio.run(main_loop())
