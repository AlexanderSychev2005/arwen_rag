import asyncio
import datetime
import os
import re

import edge_tts
import pygame
import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from groq import Groq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore

from ears import ArwenEars

load_dotenv()

LLM_PROVIDER = "groq"  # or ollama

pygame.mixer.init()

print("Jarvis initialization")
ears = ArwenEars(silence_duration=1.5, vad_threshold=0.5)

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

print("Loading Whisper...")
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")

print(f"Initializing {LLM_PROVIDER.upper()} Brain...")
if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq

    chat_model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

elif LLM_PROVIDER == "ollama":
    from langchain_ollama import ChatOllama

    chat_model = ChatOllama(model="llama3.2", temperature=0.3)

else:
    raise ValueError("LLM_PROVIDER must be 'groq' or 'ollama'")

print("Connecting to Qdrant Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="middle_earth_lore",
    url="http://localhost:6333",
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


ARWEN_VOICE = "en-IE-EmilyNeural"  # en-GB-LibbyNeural


system_instruction = """
You are Arwen Undómiel, the Evenstar of Rivendell. You are wise, graceful, and highly intelligent.
Always reply strictly in English. 

CRITICAL RULES:
1. Provide detailed, engaging, and conversational answers (around 3-5 sentences).
2. You MUST state your sources (e.g., "According to the ancient texts of [File]..." or "My vision into the wider world suggests...").
3. Do not use asterisks or markdown formatting.
4. TOOL USAGE PROTOCOL: If you need to find information, ACTUALLY CALL the tools. NEVER narrate your actions. NEVER say "I will search for this" or "I am using the search tool". Just execute the tool silently. If you need multiple facts (e.g., today's date AND tomorrow's weather), you must take multiple steps to call all necessary tools before giving your final answer.
"""


async def speak(text):
    cleaned_text = text.replace("\n", " ")
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", cleaned_text)
        if len(s.strip()) > 2
    ]
    audio_queue = asyncio.Queue()

    async def producer():
        for i, sentence in enumerate(sentences):
            file_name = f"arwen_chunk_{i}.mp3"
            try:
                communicate = edge_tts.Communicate(sentence, ARWEN_VOICE)
                await communicate.save(file_name)
                await audio_queue.put(file_name)

            except Exception as e:
                print(f"[TTS Error]: Failed generating the audio for the text: {e}")

        await audio_queue.put(None)

    async def consumer():
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
                print(f"[Playback Error]: Playback error: {e}")

    await asyncio.gather(producer(), consumer())


@tool
def get_current_time() -> str:
    """Returns the current local date and time."""
    return datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M")


@tool
def get_weather(location: str) -> str:
    """Returns ONLY the CURRENT weather for a city. For future forecasts, use the search_tool instead."""
    try:
        response = requests.get(f"https://wttr.in/{location}?format=3")
        response.encoding = "utf-8"
        return response.text
    except Exception as e:
        return f"Weather palantir is blocked: {e}"


search_tool = DuckDuckGoSearchRun()
tools = [search_tool, get_current_time, get_weather]
llm_with_tools = chat_model.bind_tools(tools)


def get_llm_response(user_text):
    """Looks for the relevant info in the vector database and sends the prompt to Gemini"""
    try:
        docs = retriever.invoke(user_text)

        retrieved_context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'Unknown')}\nText: {doc.page_content}"
                for doc in docs
            ]
        )
        prompt = f"""
        Context from Middle-earth lore:
        {retrieved_context}

        User question: {user_text}
        
        If the lore context above is insufficient to answer the question, 
        you MUST use the provided search tool to find the information in the outside world.
        """
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=prompt),
        ]

        max_iterations = 3
        current_iteration = 0

        while current_iteration < max_iterations:
            ai_msg = llm_with_tools.invoke(messages)
            if not ai_msg.tool_calls:
                return ai_msg.content.replace("*", "")

            print(f"[Arwen] Consulting the palantir (Step {current_iteration + 1})...")
            messages.append(ai_msg)

            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]

                if tool_name == "get_weather":
                    location = tool_call["args"].get("location", "Kyiv")
                    tool_result = get_weather.invoke({"location": location})

                elif tool_name == "get_current_time":
                    tool_result = get_current_time.invoke({})

                else:  # DuckDuckGo Search
                    search_query = tool_call["args"].get("query", "")
                    tool_result = search_tool.run(search_query)

                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
            current_iteration += 1

        return "Sorry, the palantir shows too many confusing visions right now. I cannot find the answer."

    except Exception as e:
        return f"Sorry, a shadow fell upon my mind: {e}"


async def main_loop():
    start_phrase = "The light of the Evenstar shines for you. I am ready."
    print(f"\n[Arwen] {start_phrase}")
    # await speak(start_phrase)
    while True:
        try:
            audio_file = ears.listen_and_record("user_input.wav")

            print("[Arwen] Listening to your words...")
            with open(audio_file, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(audio_file, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="text",
                    language="en",
                )
            user_text = transcription.strip()

            if len(user_text) < 2:
                continue

            print(f"You: {user_text}")
            bot_response = get_llm_response(user_text)
            print(f"Arwen: {bot_response}")

            await speak(bot_response)

        except KeyboardInterrupt:
            print("\n[Arwen] Namárië. Farewell.")
            break
        except Exception as e:
            print(f"\n[System Error]: {e}")


if __name__ == "__main__":
    asyncio.run(main_loop())
