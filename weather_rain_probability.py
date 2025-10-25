import requests
import json
from datetime import datetime
import sys

def get_coordinates(city, state, country_code, api_key):
    geo_url = "http://api.openweathermap.org/geo/1.0/direct"
    
    if state:
        location_query = f"{city},{state},{country_code}"
    else:
        location_query = f"{city},{country_code}"
    
    params = {
        'q': location_query,
        'limit': 1,
        'appid': api_key
    }
    
    try:
        response = requests.get(geo_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data:
            return data[0]['lat'], data[0]['lon'], data[0]['name']
        else:
            print(f"Error: Could not find coordinates for {location_query}")
            return None, None, None
    except Exception as e:
        print(f"Error getting coordinates: {e}")
        return None, None, None

def get_weather_forecast(lat, lon, target_datetime, api_key):
    forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
    
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(forecast_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        target_dt = datetime.fromisoformat(target_datetime)
        
        forecasts = data.get('list', [])
        if not forecasts:
            print("No forecast data available")
            return None
        
        closest_forecast = min(forecasts, 
            key=lambda f: abs(datetime.fromtimestamp(f['dt']) - target_dt))
        
        forecast_time = datetime.fromtimestamp(closest_forecast['dt'])
        time_diff = abs((forecast_time - target_dt).total_seconds() / 3600)
        
        weather_data = {
            'forecast_time': forecast_time.strftime('%Y-%m-%d %H:%M'),
            'time_diff_hours': time_diff,
            'temp': closest_forecast['main']['temp'],
            'feels_like': closest_forecast['main']['feels_like'],
            'temp_min': closest_forecast['main']['temp_min'],
            'temp_max': closest_forecast['main']['temp_max'],
            'pressure': closest_forecast['main']['pressure'],
            'humidity': closest_forecast['main']['humidity'],
            'weather': closest_forecast['weather'][0]['description'],
            'weather_main': closest_forecast['weather'][0]['main'],
            'clouds': closest_forecast['clouds']['all'],
            'wind_speed': closest_forecast['wind']['speed'],
            'wind_deg': closest_forecast['wind'].get('deg', 0),
            'rain_3h': closest_forecast.get('rain', {}).get('3h', 0),
            'pop': closest_forecast.get('pop', 0) * 100,
            'visibility': closest_forecast.get('visibility', 10000)
        }
        
        return weather_data
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except Exception as e:
        print(f"Error processing forecast: {e}")
        return None

def update_circuit_config(gp_name, rain_prob):
    config_file = 'circuit_config.json'
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if gp_name not in config:
            print(f"\nWarning: '{gp_name}' not found in circuit_config.json")
            print("Available circuits:")
            for name in sorted(config.keys()):
                print(f"  - {name}")
            return False
        
        old_prob = config[gp_name]['rain_prob']
        config[gp_name]['rain_prob'] = rain_prob / 100
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nUpdated {gp_name}:")
        print(f"  rain_prob: {old_prob:.2f} -> {rain_prob/100:.2f} ({rain_prob:.1f}%)")
        
        return True
        
    except FileNotFoundError:
        print(f"\nError: {config_file} not found in current directory")
        return False
    except Exception as e:
        print(f"\nError updating config: {e}")
        return False

def main():
    print("="*80)
    print("WEATHER RAIN PROBABILITY FETCHER")
    print("Using OpenWeatherMap API 2.5 (FREE TIER)")
    print("="*80)
    
    api_key = input("\nEnter your OpenWeatherMap API key: ").strip()
    if not api_key:
        print("Error: API key is required")
        print("\nGet a FREE API key at: https://openweathermap.org/api")
        sys.exit(1)
    
    print("\n" + "-"*80)
    print("RACE LOCATION")
    print("-"*80)
    city = input("City (e.g., Austin, Monaco, Monza): ").strip()
    state = input("Region/State (optional, e.g., Texas): ").strip()
    country_code = input("Country code (e.g., US, MC, IT): ").strip().upper()
    
    if not city or not country_code:
        print("Error: City and country code are required")
        sys.exit(1)
    
    print("\n" + "-"*80)
    print("RACE TIMING")
    print("-"*80)
    race_date = input("Race date (YYYY-MM-DD): ").strip()
    race_time = input("Local race time (HH:MM, 24-hour format): ").strip()
    
    target_datetime = f"{race_date}T{race_time}:00"
    
    try:
        target_dt = datetime.fromisoformat(target_datetime)
        days_ahead = (target_dt - datetime.now()).days
        
        if days_ahead > 5:
            print(f"\nWarning: Race is {days_ahead} days ahead")
            print("Free API only provides 5-day forecasts - data may be less accurate")
        elif days_ahead < 0:
            print(f"\nWarning: Race date is in the past")
    except ValueError:
        print("Error: Invalid date/time format")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("FETCHING WEATHER DATA")
    print("="*80)
    
    print(f"Step 1: Getting coordinates for {city}, {country_code}...")
    lat, lon, location_name = get_coordinates(city, state, country_code, api_key)
    
    if lat is None:
        sys.exit(1)
    
    print(f"Location found: {location_name}")
    print(f"Coordinates: {lat:.4f}, {lon:.4f}")
    
    print(f"\nStep 2: Getting weather forecast for {target_datetime}...")
    weather = get_weather_forecast(lat, lon, target_datetime, api_key)
    
    if weather is None:
        sys.exit(1)
    
    print("\n" + "="*80)
    print("WEATHER FORECAST")
    print("="*80)
    print(f"Location: {location_name}")
    print(f"Target time: {target_datetime}")
    print(f"Forecast time: {weather['forecast_time']}")
    print(f"Time difference: {weather['time_diff_hours']:.1f} hours")
    print("\n" + "-"*80)
    print("CONDITIONS")
    print("-"*80)
    print(f"Weather: {weather['weather']}")
    print(f"Temperature: {weather['temp']:.1f} C (feels like {weather['feels_like']:.1f} C)")
    print(f"Temperature range: {weather['temp_min']:.1f} - {weather['temp_max']:.1f} C")
    print(f"Humidity: {weather['humidity']}%")
    print(f"Cloud cover: {weather['clouds']}%")
    print(f"Wind speed: {weather['wind_speed']:.1f} m/s")
    print(f"Wind direction: {weather['wind_deg']} degrees")
    print(f"Visibility: {weather['visibility']} m")
    print(f"Pressure: {weather['pressure']} hPa")
    
    print("\n" + "-"*80)
    print("RAIN ANALYSIS")
    print("-"*80)
    print(f"Probability of precipitation: {weather['pop']:.2f}%")
    if weather['rain_3h'] > 0:
        print(f"Expected rainfall (3h): {weather['rain_3h']:.1f} mm")
    else:
        print("No rain expected")
    
    rain_prob = weather['pop']
    
    print("\n" + "="*80)
    print("UPDATE CIRCUIT CONFIGURATION")
    print("="*80)
    
    gp_name = input("\nEnter Grand Prix name (e.g., 'United States'): ").strip()
    
    if gp_name:
        update_choice = input(f"\nUpdate {gp_name} with rain probability {rain_prob:.2f}%? (y/n): ").strip().lower()
        
        if update_choice == 'y':
            if update_circuit_config(gp_name, rain_prob):
                print("\nConfiguration updated successfully")
            else:
                print("\nFailed to update configuration")
        else:
            print("\nConfiguration not updated")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Rain probability for {location_name}: {rain_prob:.2f}%")
    print(f"Weather: {weather['weather']}")
    print(f"Temperature: {weather['temp']:.1f} C")
    print("="*80)

if __name__ == "__main__":
    main()