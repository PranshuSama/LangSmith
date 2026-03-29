import os
import json
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from dotenv import load_dotenv

os.environ['LANGCHAIN_PROJECT'] = 'ReAct Agent'

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city using Open-Meteo (free, no API key).
  """
  city_name = city.strip()
  geocode_url = "https://geocoding-api.open-meteo.com/v1/search"

  try:
    geocode_resp = requests.get(
      geocode_url,
      params={"name": city_name, "count": 1, "language": "en", "format": "json"},
      timeout=6,
    )
    geocode_data = geocode_resp.json()
  except requests.Timeout:
    return json.dumps({"success": False, "error": "weather_api_timeout"})
  except requests.RequestException as exc:
    return json.dumps({"success": False, "error": f"weather_api_request_failed: {exc}"})
  except ValueError:
    return json.dumps({"success": False, "error": "weather_api_invalid_json"})

  results = geocode_data.get("results") if isinstance(geocode_data, dict) else None
  if not results:
    return json.dumps({"success": False, "error": "city_not_found", "retryable": False})

  place = results[0]
  lat = place.get("latitude")
  lon = place.get("longitude")

  if lat is None or lon is None:
    return json.dumps({"success": False, "error": "invalid_geocode_result", "retryable": False})

  forecast_url = "https://api.open-meteo.com/v1/forecast"
  try:
    weather_resp = requests.get(
      forecast_url,
      params={"latitude": lat, "longitude": lon, "current": "temperature_2m,weather_code"},
      timeout=6,
    )
    weather_data = weather_resp.json()
  except requests.Timeout:
    return json.dumps({"success": False, "error": "weather_api_timeout", "retryable": False})
  except requests.RequestException as exc:
    return json.dumps({"success": False, "error": f"weather_api_request_failed: {exc}", "retryable": False})
  except ValueError:
    return json.dumps({"success": False, "error": "weather_api_invalid_json", "retryable": False})

  current = weather_data.get("current") if isinstance(weather_data, dict) else None
  units = weather_data.get("current_units") if isinstance(weather_data, dict) else None

  if not current:
    return json.dumps({"success": False, "error": "weather_api_failed", "retryable": False})

  return json.dumps({
    "success": True,
    "city": place.get("name", city_name),
    "country": place.get("country", ""),
    "temperature": current.get("temperature_2m"),
    "temperature_unit": (units or {}).get("temperature_2m", "C"),
    "weather_code": current.get("weather_code"),
  })

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

SYSTEM_PROMPT = (
  "You are a fast assistant. Use tools with minimum steps. "
  "For birthplace+temperature questions: "
  "1) Call duckduckgo_search exactly once to find birthplace city. "
  "2) Call get_weather_data exactly once with that city. "
  "3) If weather tool returns any non-retryable error (retryable=false), stop immediately and answer without more tool calls. "
  "Do not call the same tool repeatedly. Keep answers concise."
)

agent_executor = create_agent(
  model=llm,
  tools=[search_tool, get_weather_data],
  system_prompt=SYSTEM_PROMPT,
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke(
  {
    "messages": [("user", "Identify the birthplace city of Kalpana Chawla (search) and give its current temperature")]
  },
  config={"recursion_limit": 6},
)
print(response)

if response.get("messages"):
  print(response["messages"][-1].content)