# travel_api.py
import os
from serpapi import GoogleSearch


class TravelAPIClient:
    """
    Wrapper for SerpAPI Google Flights engine.
    Provides methods to search for flights.
    """

    def __init__(self):
        # Load API key from environment or fallback hardcoded (not recommended in prod)
        self.api_key = os.getenv(
            "TRAVEL_API_KEY",
            "4e13fc78d701d94e599450ed9fc78ea7918338ba40c140883dde235148e2f974"
        )

    def search_flights(self, origin: str, destination: str, outbound_date: str, return_date: str):
        """
        Search flights between origin and destination using SerpAPI.
        
        Args:
            origin (str): Departure airport IATA code (e.g., "PEK").
            destination (str): Arrival airport IATA code (e.g., "AUS").
            outbound_date (str): Outbound date in YYYY-MM-DD format.
            return_date (str): Return date in YYYY-MM-DD format.

        Returns:
            dict: JSON response from SerpAPI.
        """
        params = {
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": outbound_date,
            "return_date": return_date,
            "currency": "USD",
            "hl": "en",
            "api_key": self.api_key,
        }

        search = GoogleSearch(params)
        return search.get_dict()

    def search_example(self):
        """
        Example fixed query for testing the API client.
        """
        return self.search_flights(
            origin="PEK",
            destination="AUS",
            outbound_date="2025-09-30",
            return_date="2025-10-06"
        )


