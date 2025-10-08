import requests
import json

TRAVEL_API_KEY = "e492bc2f98e84f3ec1927fb6a3f5b7b8"
TRAVEL_API_URL = "https://api.aviationstack.com/v1/flights"

all_flights = []
offset = 0
per_page = 100  # Max allowed per request

while True:
    params = {
        "access_key": TRAVEL_API_KEY,
        "limit": per_page,
        "offset": offset
    }
    
    try:
        response = requests.get(TRAVEL_API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            flights = data.get("data", [])
            if not flights:
                break  # No more flights to fetch
            all_flights.extend(flights)
            offset += per_page
            print(f"Fetched {len(all_flights)} flights so far...")
        else:
            print(f"API returned an error. Status code: {response.status_code}")
            print(response.text)
            break
    except Exception as e:
        print("Error connecting to the API:", e)
        break

# Save all flights to JSON
with open("flight_data.json", "w", encoding="utf-8") as f:
    json.dump(all_flights, f, ensure_ascii=False, indent=4)

print(f"All available flight data saved to flight_data.json. Total flights fetched: {len(all_flights)}")
