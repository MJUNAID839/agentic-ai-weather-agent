# 🌦️ Agentic Weather Assistant

**Day 1 of learning Agentic AI** — a small agent built using the **ReAct pattern** (Reasoning + Acting), demonstrating how an LLM can reason about what it needs, call a real tool, and generate a grounded answer instead of guessing.

## Why this project?

LLMs are frozen at training time — they have no live internet access and can't know today's actual weather. Ask one directly, and it will either refuse or hallucinate a plausible-sounding but fake answer.

This project solves that by giving the LLM (Gemini) access to a real weather tool. Instead of guessing, the agent explicitly reasons about what data it needs, retrieves real data, and only then answers — grounded in fact, not fabrication.

## How it works — the ReAct loop

```
🤔 Thought       → Model decides if it needs live weather data
⚡ Action        → Calls get_weather(city, day) — a real API call to Open-Meteo
👁️ Observation   → Model sees the real, live result
✅ Final Answer  → Model answers using ONLY the real data retrieved
```

The agent also decides *which* day's data it needs — `"today"` for current live conditions, or `"tomorrow"` for a real forecast — based on how the question is phrased.

## Two versions included

| File | Description |
|---|---|
| `weather_agent.py` | Terminal version — type a question, see the full reasoning trace printed live |
| `Dashboard.py` | Streamlit web dashboard — same agent, visual UI with step-by-step reasoning display |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/MJUNAID839/agentic-ai-weather-agent.git
cd agentic-ai-weather-agent

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install requests google-genai python-dotenv streamlit

# 4. Add your Gemini API key
# Create a .env file in the project root with:
# GEMINI_API_KEY=your_key_here
# (Get a free key at https://aistudio.google.com/apikey)
```

## Running it

**Terminal version:**
```bash
python weather_agent.py
```

**Dashboard version:**
```bash
streamlit run Dashboard.py
```

## Tech stack

- **Gemini API** (`google-genai`) — reasoning and language generation
- **Open-Meteo API** — free, no-key-required weather data (current + forecast)
- **Streamlit** — dashboard UI
- Hand-built ReAct loop — no agent framework used intentionally, to understand the underlying mechanics before moving to frameworks like LangGraph/CrewAI

## What I learned building this

- Why LLMs need external tools to answer questions requiring current, real-world data
- The ReAct pattern: Thought → Action → Observation → Final Answer
- Handling real-world messiness: deprecated packages, API rate limits/retries, and LLMs not always following output-format instructions perfectly (robust JSON parsing was needed)
- The difference between a tool that just displays data (like a weather widget) and an agent that reasons across data to make a decision

## Next steps

- Add a second tool so the agent has to choose between multiple options
- Move to a proper agent framework (LangGraph / CrewAI)
- Add memory so the agent can handle multi-turn conversations about weather plans

---

*Part of my Agentic AI learning journey — follow along on [LinkedIn](#).*