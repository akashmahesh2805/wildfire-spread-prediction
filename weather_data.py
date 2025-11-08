import requests
import pandas as pd

# define LA region bounds
north, south, east, west = 34.25, 34.15, -118.05, -118.25
start_date, end_date = "2020-01-01", "2025-06-30"

url = "https://archive-api.open-meteo.com/v1/era5"
params = {
    "latitude": (north + south) / 2,
    "longitude": (east + west) / 2,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
}

response = requests.get(url, params=params)
data = response.json()
df_weather = pd.DataFrame(data["hourly"])
df_weather["time"] = pd.to_datetime(df_weather["time"])
df_weather.to_csv("weather_data.csv", index=False)
