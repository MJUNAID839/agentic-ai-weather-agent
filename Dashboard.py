"""
Day 1 Agentic AI Project — Weather Agent Dashboard (Streamlit)
Author: Muhammad Junaid Asghar

A visual dashboard for the ReAct weather agent. Shows each step of the
Thought -> Action -> Observation -> Final Answer loop live in the browser.

Setup:
    pip install streamlit requests google-genai python-dotenv

Run with:
    streamlit run dashboard.py
    (NOT `python dashboard.py` — Streamlit apps must be launched with `streamlit run`)
"""

import os
import json
import re
import time
import requests
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv

# ---------- SETUP ----------
load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-flash-lite-latest"


def call_model(prompt: str, max_retries: int = 4) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text.strip()
        except genai_errors.ServerError:
            time.sleep(attempt * 5)
    raise RuntimeError("Model still unavailable after several retries. Try again in a few minutes.")


def get_weather(city: str, day: str = "today") -> dict:
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
        "forecast_days": 2,
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


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Agentic Weather Assistant", page_icon="🌦️", layout="centered")

st.title("🌦️ Agentic Weather Assistant")
st.caption("A small ReAct-pattern agent — Thought → Action → Observation → Final Answer")

user_question = st.text_input(
    "Ask a weather-related question:",
    placeholder="e.g. Should I carry an umbrella in Lahore today?",
)

if st.button("Run Agent", type="primary") and user_question:
    # ---------- THOUGHT ----------
    with st.status("Agent is reasoning...", expanded=True) as status:
        st.write("🤔 **Thought:** Deciding whether a tool call is needed...")

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
        st.code(thought_response if thought_response else "(empty response)", language="json")

        # Robust extraction: find the first {...} block instead of assuming
        # the whole response is clean JSON (models don't always follow
        # formatting instructions perfectly).
        match = re.search(r"\{.*\}", thought_response, re.DOTALL)
        if match:
            try:
                decision = json.loads(match.group())
            except json.JSONDecodeError:
                decision = {"action": "none"}
        else:
            decision = {"action": "none"}

        # ---------- ACTION + OBSERVATION ----------
        observation = None
        if decision.get("action") == "get_weather":
            city = decision["city"]
            day = decision.get("day", "today")
            st.write(f"⚡ **Action:** Calling `get_weather('{city}', day='{day}')`")
            observation = get_weather(city, day)
            st.write("👁️ **Observation:**")
            st.json(observation)

        # ---------- FINAL ANSWER ----------
        st.write("✅ **Generating final grounded answer...**")
        final_prompt = f"""
User asked: "{user_question}"

Here is the real, current data you retrieved (do not contradict this data,
do not invent additional data):
{json.dumps(observation)}

Using ONLY this data, give a clear, helpful, natural-language answer to the
user's question. Mention the actual numbers you used to reach your conclusion.
"""
        final_answer = call_model(final_prompt)
        status.update(label="Done!", state="complete", expanded=True)

    st.success(final_answer)