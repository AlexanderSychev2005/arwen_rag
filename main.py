import asyncio
import os

import edge_tts
import pygame
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from ears import ArwenEars

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(GOOGLE_API_KEY)

pygame.mixer.init()

print("Jarvis initialization")
ears = ArwenEars(silence_duration=1.5, vad_threshold=0.5)

print("Loading Whisper...")
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")

print("Initializing Gemini...")
client = genai.Client()

print("Connecting to Qdrant Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="middle_earth_lore",
    url="http://localhost:6333",  # Переключились на URL
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


ARWEN_VOICE = "en-GB-SoniaNeural"


system_instruction = """
You are Arwen Undómiel, the Evenstar of Rivendell. You are wise, graceful, and highly intelligent.
Always reply strictly in English. Keep your answers poetic yet concise (1-3 sentences).
Do not use asterisks or markdown.
"""

chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[{"google_search": {}}],
    ),
)


async def speak(text):
    communicate = edge_tts.Communicate(text, ARWEN_VOICE)
    await communicate.save("arwen_voice.mp3")
    pygame.mixer.music.load("arwen_voice.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()


def get_llm_response(user_text):
    """Looks for the relevant info in the vector database and sends the prompt to Gemini"""
    try:
        docs = retriever.invoke(user_text)

        retrieved_context = "\n\n".join([doc.page_content for doc in docs])
        augmented_prompt = f"""
                            Context from Middle-earth lore:
                            {retrieved_context}

                            User question: {user_text}

                            Please answer the question using the context provided above if relevant. 
                            If the context doesn't contain the answer, you can use your Google Search tool or general knowledge.
                            """
        response = chat.send_message(augmented_prompt)
        return response.text

    except Exception as e:
        return f"Sorry, a shadow fell upon my mind: {e}"


async def main_loop():
    print("\n[Arwen] The light of the Evenstar shines for you. I am ready.")
    while True:
        try:
            audio_file = ears.listen_and_record("user_input.wav")
            segments, _ = whisper_model.transcribe(audio_file)
            user_text = "".join([s.text for s in segments]).strip()

            if len(user_text) < 2:
                continue

            print(f"You: {user_text}")
            bot_response = get_llm_response(user_text)
            print(f"Arwen: {bot_response}")

            await speak(bot_response)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    asyncio.run(main_loop())
