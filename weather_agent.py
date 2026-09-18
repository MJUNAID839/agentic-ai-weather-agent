"""
Day 1 Agentic AI Project — Weather Agent (ReAct pattern)
Author: Muhammad Junaid Asghar

Demonstrates the core agent loop: Thought -> Action -> Observation -> Final Answer.
No framework used on purpose — this hand-built loop teaches the underlying
mechanics that frameworks like CrewAI/LangGraph automate later.

Setup:
    pip install requests google-generativeai

    Get a free Gemini API key: https://aistudio.google.com/apikey
    Then set it as an environment variable:
        export GEMINI_API_KEY="your_key_here"      (Mac/Linux)
        setx GEMINI_API_KEY "your_key_here"          (Windows)
"""

import os
import json
import re
import time
import requests
from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv

# ---------- SETUP ----------
load_dotenv()  # reads GEMINI_API_KEY from your .env file
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-flash-lite-latest"  # lighter model, less prone to free-tier overload


def call_model(prompt: str, max_retries: int = 4) -> str:
    """
    Calls the model with automatic retry on temporary overload (503 errors).
    Waits a bit longer each time it retries.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=prompt
            )
            return response.text.strip()
        except genai_errors.ServerError as e:
            wait_seconds = attempt * 5
            print(f" Model overloaded (attempt {attempt}/{max_retries}). "
                  f"Waiting {wait_seconds}s before retrying...")
            time.sleep(wait_seconds)
    raise RuntimeError("Model still unavailable after several retries. Try again in a few minutes.")


# ---------- TOOL: get_weather ----------
def get_weather(city: str, day: str = "today") -> dict:
    """
    Tool the agent can call. Uses Open-Meteo (no API key required).
    day: "today" -> returns current live conditions
         "tomorrow" -> returns tomorrow's forecast (max/min temp, expected rain)
    """
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_res = requests.get(geo_url, params={"name": city, "count": 1}).json()

    if "results" not in geo_res:
        return {"error": f"Could not find location: {city}"}

    lat = geo_res["results"][0]["latitude"]
    lon = geo_res["results"][0]["longitude"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,rain,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 2,  # today + tomorrow
        "timezone": "auto",
    }
    weather_res = requests.get(weather_url, params=params).json()

    if day == "tomorrow":
        daily = weather_res.get("daily", {})
        return {
            "city": city,
            "day": "tomorrow",
            "temp_max_C": daily.get("temperature_2m_max", [None, None])[1],
            "temp_min_C": daily.get("temperature_2m_min", [None, None])[1],
            "expected_precipitation_mm": daily.get("precipitation_sum", [None, None])[1],
        }
    else:
        current = weather_res.get("current", {})
        return {
            "city": city,
            "day": "today",
            "temperature_C": current.get("temperature_2m"),
            "precipitation_mm": current.get("precipitation"),
            "rain_mm": current.get("rain"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
        }


# ---------- THE AGENT LOOP (ReAct) ----------
def run_agent(user_question: str):
    print(f"\n USER QUESTION: {user_question}\n")

    # Step 1: THOUGHT — ask the LLM to reason about what it needs
    thought_prompt = f"""
You are a reasoning agent. A user asked: "{user_question}"

You have access to one tool:
- get_weather(city: str, day: str) -> returns weather data for that city.
  day must be exactly "today" (current live conditions) or "tomorrow" (forecast).
  Use "tomorrow" if the question refers to a future day; otherwise use "today".

Think step by step about what you need to answer this question.
If you need weather data, respond ONLY with valid JSON in this exact format:
{{"action": "get_weather", "city": "<city name>", "day": "<today or tomorrow>"}}

If you don't need any tool, respond ONLY with:
{{"action": "none"}}
"""
    thought_response = call_model(thought_prompt)
    print(f" THOUGHT + ACTION DECISION:\n{thought_response}\n")

    # Robust extraction: find the first {...} block instead of assuming
    # the whole response is clean JSON.
    match = re.search(r"\{.*\}", thought_response, re.DOTALL)
    if match:
        try:
            decision = json.loads(match.group())
        except json.JSONDecodeError:
            decision = {"action": "none"}
    else:
        decision = {"action": "none"}

    # Step 2: ACTION — call the tool if the agent decided it needs one
    observation = None
    if decision.get("action") == "get_weather":
        city = decision["city"]
        day = decision.get("day", "today")
        print(f" ACTION: Calling get_weather('{city}', day='{day}')")
        observation = get_weather(city, day)
        print(f" OBSERVATION: {observation}\n")

    # Step 3: FINAL REASONING — generate the grounded final answer
    final_prompt = f"""
User asked: "{user_question}"

Here is the real, current data you retrieved (do not contradict this data,
do not invent additional data):
{json.dumps(observation)}

Using ONLY this data, give a clear, helpful, natural-language answer to the
user's question. Mention the actual numbers you used to reach your conclusion.
"""
    final_answer = call_model(final_prompt)
    print(f" FINAL ANSWER:\n{final_answer}\n")


# ---------- RUN IT ----------
if __name__ == "__main__":
    print("🌦️  Agentic Weather Assistant — type 'quit' to exit\n")
    while True:
        user_question = input("Ask a weather question: ").strip()
        if user_question.lower() in ("quit", "exit"):
            break
        if user_question:
            run_agent(user_question)