import datetime

import requests
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from config import OPENWEATHER_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

tavily_search = TavilySearchResults(max_results=2)


@tool
def search_the_web(query: str) -> str:
    """
    Searches the internet for modern-world facts, dates, pop culture, or information not found in Middle-earth lore.
    """
    try:
        results = tavily_search.invoke({"query": query})
        if isinstance(results, str):
            return results

        return "\n".join(
            [f"Source: {r.get('url')}\nContent: {r.get('content')}" for r in results]
        )
    except Exception as e:
        return f"Search failed: {e}"


@tool
def send_telegram_message(message: str) -> str:
    """
    Sends a text message to the user's Telegram.
    CRITICAL: ONLY use this tool if the user EXPLICITLY says words like "Telegram", "send me", or "message me".
    If the user just asks a question (e.g., "Tell me about..."), DO NOT use this tool. Just answer them normally.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return "Error: Telegram credentials are missing in the configuration."

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"🧝🏻‍♀️ *Message from Arwen:*\n\n{message}",
            "parse_mode": "Markdown",
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            return "SUCCESS: Message was successfully sent to the user's Telegram. You can now speak to the user."
        else:
            return f"Failed to send message: {response.text}"
    except Exception as e:
        return f"Telegram palantir is blocked: {e}"


@tool
def get_current_time() -> str:
    """Returns the current local date and time. Use this to know what 'today' or 'now' is."""
    return datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M")


@tool
def get_weather(location: str) -> str:
    """Returns the current weather AND tomorrow's forecast for a specific city. Use this for ALL weather questions."""
    try:
        if not OPENWEATHER_API_KEY:
            return "Error: OpenWeather API key is missing."

        current_url: str = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
        curr_data: dict = requests.get(current_url).json()

        if curr_data.get("cod") != 200:
            return f"Could not find weather for {location}."

        curr_temp: float = curr_data["main"]["temp"]
        curr_desc: str = curr_data["weather"][0]["description"]

        forecast_url: str = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
        cast_data: dict = requests.get(forecast_url).json()

        tomorrow_data: dict = cast_data["list"][8]
        tom_temp: float = tomorrow_data["main"]["temp"]
        tom_desc: str = tomorrow_data["weather"][0]["description"]
        tom_date: str = tomorrow_data["dt_txt"]

        return (
            f"Current weather in {location}: {curr_desc}, {curr_temp}°C. \n"
            f"Forecast for tomorrow ({tom_date}): {tom_desc}, around {tom_temp}°C."
        )
    except Exception as e:
        return f"Weather palantir is blocked: {e}"


tools_list: list = [
    search_the_web,
    get_current_time,
    get_weather,
    send_telegram_message,
]
