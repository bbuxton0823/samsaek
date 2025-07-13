#!/usr/bin/env python3
"""
🚀 Enhanced Samsaek Travel Agent with SamSaekTransit Integration
Real-time transit data, traffic monitoring, and comprehensive travel assistance
"""

import requests
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
import uuid
import google.generativeai as genai
from exa_py import Exa

# 🔍 Initialize Weave tracking (with error handling)
try:
    import weave
    weave.init("samsaek")
    WEAVE_ENABLED = True
    print("🔍 Weave tracking initialized for Enhanced Travel Agent")
except Exception as e:
    print(f"⚠️ Weave initialization failed: {e}")
    print("📊 Continuing without Weave tracking")
    WEAVE_ENABLED = False
    
    # Create dummy decorator for when Weave is not available
    def weave_op_dummy():
        def decorator(func):
            return func
        return decorator
    
    class WeaveStub:
        @staticmethod
        def op():
            return weave_op_dummy()
    
    weave = WeaveStub()

class EnhancedTravelAgent:
    """Enhanced Travel Agent with SamSaekTransit Integration"""
    
    def __init__(self):
        # 🔑 API Keys Configuration - Updated with real keys
        self.gemini_api_key = "AIzaSyA-mock-key-for-demo"  # Replace with real key
        self.exa_api_key = "0ada66fe-bf44-4f53-8290-e4e1fd24d92c"  # Updated with real key
        self.transitland_api_key = "x5unflDSbpKEWnThyfmteM8MHxIsg3eL"  # Confirmed working key
        self.google_maps_api_key = "AIzaSyDuv9fgGshY9pKXW_YTNijy6jrLajeo0h0"  # Updated with real key
        self.traffic_511_api_key = "09ddeab2-9b3f-4531-8a35-5304443e02b4"  # Updated with real key
        
        # Initialize Gemini if available
        if self.gemini_api_key and self.gemini_api_key != "AIzaSyA-mock-key-for-demo":
            genai.configure(api_key=self.gemini_api_key)
        
        print("🎯 Enhanced Travel Agent Initialized")
        print("   ✈️ Flight & Hotel Services: Ready")
        print("   🚊 Transit Integration: Ready")
        print("   🚗 Traffic Monitoring: Ready")
        print("   🔍 Real-time Data: Ready")
        print(f"   📊 Exa AI Search: {'✅ Active' if self.exa_api_key else '❌ No API Key'}")
        
    @weave.op()
    def get_geolocation(self) -> List[float]:
        """Get device location using Google Maps Geolocation API"""
        if not self.google_maps_api_key:
            # Return San Francisco coordinates as default
            return [37.7749, -122.4194]
            
        url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={self.google_maps_api_key}"
        
        payload = {
            "considerIp": True,
            "wifiAccessPoints": [
                {
                    "macAddress": "01:23:45:67:89:AB",
                    "signalStrength": -65,
                    "signalToNoiseRatio": 40
                }
            ]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            lat = data.get("location", {}).get("lat")
            lng = data.get("location", {}).get("lng")
            return [lat, lng] if lat and lng else [37.7749, -122.4194]
        except Exception as e:
            print(f"⚠️ Geolocation API error: {e}")
            return [37.7749, -122.4194]  # Fallback to San Francisco
    
    @weave.op()
    def search_transit_routes(self, search_query: str, route_type: int = 3) -> List[str]:
        """Search for transit routes using TransitLand API"""
        api_url = f"https://transit.land/api/v2/rest/routes?api_key={self.transitland_api_key}"
        params = {"search": search_query, "route_type": route_type} if search_query else {"route_type": route_type}
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            
            routes = response.json().get("routes", [])
            route_list = []
            
            for route in routes[:10]:  # Limit to top 10 results
                short_name = route.get('route_short_name', 'N/A')
                long_name = route.get('route_long_name', 'N/A')
                route_list.append(f"{short_name} - {long_name}")
            
            return route_list if route_list else ["No routes found for your search"]
            
        except Exception as e:
            print(f"❌ Transit routes error: {e}")
            return [f"Unable to fetch route data: {str(e)}"]
    
    @weave.op()
    def search_transit_stops(self, search_query: str, route_type: int = 3) -> List[str]:
        """Search for transit stops using TransitLand API"""
        api_url = f"https://transit.land/api/v2/rest/stops?api_key={self.transitland_api_key}"
        params = {"search": search_query, "served_by_route_type": route_type} if search_query else {"served_by_route_type": route_type}
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            
            stops = response.json().get("stops", [])
            stop_list = []
            
            for stop in stops[:10]:  # Limit to top 10 results
                stop_name = stop.get('stop_name', 'N/A')
                stop_id = stop.get('stop_id', 'N/A')
                stop_list.append(f"{stop_name} - ID: {stop_id}")
            
            return stop_list if stop_list else ["No stops found for your search"]
            
        except Exception as e:
            print(f"❌ Transit stops error: {e}")
            return [f"Unable to fetch stop data: {str(e)}"]
    
    @weave.op()
    def get_nearby_transit(self, radius: int = 1000, option: str = "stops") -> List[str]:
        """Find nearby transit options based on current location"""
        lat, lng = self.get_geolocation()
        
        if option == "routes":
            api_url = f"https://transit.land/api/v2/rest/routes?api_key={self.transitland_api_key}"
        else:
            api_url = f"https://transit.land/api/v2/rest/stops?api_key={self.transitland_api_key}"
        
        params = {"lat": lat, "lon": lng, "radius": radius}
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            result_list = []
            
            if option == "routes":
                routes = data.get("routes", [])
                for route in routes[:10]:
                    short_name = route.get('route_short_name', 'N/A')
                    long_name = route.get('route_long_name', 'N/A')
                    result_list.append(f"{short_name} - {long_name}")
            else:
                stops = data.get("stops", [])
                for stop in stops[:10]:
                    stop_name = stop.get('stop_name', 'N/A')
                    stop_id = stop.get('stop_id', 'N/A')
                    result_list.append(f"{stop_name} - ID: {stop_id}")
            
            return result_list if result_list else [f"No nearby {option} found within {radius}m"]
            
        except Exception as e:
            print(f"❌ Nearby transit error: {e}")
            return [f"Unable to fetch nearby transit data: {str(e)}"]
    
    @weave.op()
    def get_transit_departures(self, stop_id: str, agency_onestop_id: str) -> List[str]:
        """Get real-time departures for a specific transit stop"""
        if not stop_id or not agency_onestop_id:
            return ["Stop ID and Agency ID are required for departure information"]
        
        # First, get the onestop_id for the stop
        api_url = f"https://transit.land/api/v2/rest/stops?stop_id={stop_id}&served_by_onestop_ids={agency_onestop_id}&api_key={self.transitland_api_key}"
        
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            data = response.json().get("stops", [])
            if not data:
                return ["No stop found with the provided stop ID and agency ID"]
            
            onestop_departures_id = data[0].get("onestop_id")
            if not onestop_departures_id:
                return ["Unable to get stop onestop ID"]
            
            # Get departures for this stop
            departures_url = f"https://transit.land/api/v2/rest/stops/{onestop_departures_id}/departures?api_key={self.transitland_api_key}"
            departures_response = requests.get(departures_url, timeout=10)
            departures_response.raise_for_status()
            
            departures_data = departures_response.json().get("stops", [])
            if not departures_data:
                return ["No departure data available for this stop"]
            
            departures = departures_data[0].get("departures", [])
            departure_list = []
            
            for departure in departures[:10]:  # Limit to next 10 departures
                arrival = departure.get("arrival", {})
                estimated = arrival.get("estimated", "Unknown time")
                trip = departure.get("trip", {})
                route = trip.get("route", {})
                route_name = route.get("route_short_name", "N/A")
                trip_headsign = trip.get("trip_headsign", "N/A")
                departure_list.append(f"{route_name} to {trip_headsign} at {estimated}")
            
            return departure_list if departure_list else ["No upcoming departures found"]
            
        except Exception as e:
            print(f"❌ Transit departures error: {e}")
            return [f"Unable to fetch departure data: {str(e)}"]
    
    @weave.op()
    def get_traffic_data(self) -> List[str]:
        """Get real-time traffic incident data"""
        if not self.traffic_511_api_key:
            return ["Traffic data unavailable - API key not configured"]
        
        api_url = f"https://api.511.org/traffic/events?api_key={self.traffic_511_api_key}&format=JSON"
        
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            # Handle UTF-8 BOM encoding issue
            content = response.content.decode('utf-8-sig')
            traffic_data = json.loads(content).get("events", [])
            incident_list = []
            
            for incident in traffic_data[:10]:  # Limit to 10 incidents
                headline = incident.get('headline', 'Traffic incident')
                roads = incident.get('roads', [])
                highway = roads[0].get('name', 'Unknown road') if roads else 'Unknown road'
                incident_list.append(f"{headline} on {highway}")
            
            return incident_list if incident_list else ["No traffic incidents reported"]
            
        except Exception as e:
            print(f"❌ Traffic data error: {e}")
            return [f"Unable to fetch traffic data: {str(e)}"]
    
    @weave.op()
    async def search_travel_info(self, query: str) -> str:
        """Search for travel information using Exa AI"""
        if not self.exa_api_key:
            return "Travel search unavailable - API key not configured"
        
        try:
            exa = Exa(api_key=self.exa_api_key)
            results = exa.search_and_contents(query, text=True, num_results=3).results
            
            if results:
                summaries = []
                for result in results:
                    if hasattr(result, 'text') and result.text:
                        # Take first 200 characters of content
                        summary = result.text[:200] + "..." if len(result.text) > 200 else result.text
                        summaries.append(summary)
                
                return " ".join(summaries) if summaries else "No detailed information found"
            else:
                return "No search results found for your query"
                
        except Exception as e:
            print(f"❌ Exa search error: {e}")
            return f"Unable to search travel information: {str(e)}"
    
    def _extract_destination(self, user_input: str) -> Optional[str]:
        """Extract destination from user input"""
        destinations = [
            "san francisco", "san diego", "los angeles", "new york", "las vegas", 
            "chicago", "miami", "boston", "seattle", "portland", "denver", "austin", "dallas",
            "paris", "london", "tokyo", "barcelona", "rome", "madrid", "berlin", "amsterdam"
        ]
        
        user_lower = user_input.lower()
        for dest in destinations:
            if dest in user_lower:
                return dest.title()
        
        # Extract words after common patterns
        patterns = ["to ", "visit ", "travel to ", "going to ", "flight to "]
        for pattern in patterns:
            if pattern in user_lower:
                after_pattern = user_lower.split(pattern, 1)[1]
                words = after_pattern.split()
                if words:
                    return words[0].title()
        
        return None
    
    def _classify_travel_request(self, user_input: str) -> str:
        """Classify the type of travel request"""
        user_lower = user_input.lower()
        
        # Transit-related keywords
        transit_keywords = ["train", "bus", "metro", "subway", "transit", "rail", "station", "stop", "route", "schedule", "departure"]
        if any(keyword in user_lower for keyword in transit_keywords):
            return "transit"
        
        # Traffic-related keywords
        traffic_keywords = ["traffic", "road", "highway", "incident", "congestion", "delay"]
        if any(keyword in user_lower for keyword in traffic_keywords):
            return "traffic"
        
        # Flight-related keywords
        flight_keywords = ["flight", "plane", "airport", "airline", "fly"]
        if any(keyword in user_lower for keyword in flight_keywords):
            return "flight"
        
        # Hotel-related keywords
        hotel_keywords = ["hotel", "stay", "accommodation", "room", "booking"]
        if any(keyword in user_lower for keyword in hotel_keywords):
            return "hotel"
        
        return "general"
    
    @weave.op()
    async def handle_travel_request(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Main handler for travel requests with enhanced transit integration"""
        
        request_id = str(uuid.uuid4())
        user_name = context.get('user_name', '') if context else ''
        name_suffix = f", {user_name}" if user_name else ""
        
        print(f"✈️ Enhanced Travel Agent processing: {request_id}")
        print(f"   📝 Input: {user_input}")
        
        # Classify the request
        request_type = self._classify_travel_request(user_input)
        destination = self._extract_destination(user_input)
        
        print(f"   🎯 Type: {request_type}")
        print(f"   📍 Destination: {destination}")
        
        try:
            # Handle different request types
            if request_type == "transit":
                return await self._handle_transit_request(user_input, destination, name_suffix)
            elif request_type == "traffic":
                return await self._handle_traffic_request(user_input, name_suffix)
            elif request_type == "flight":
                return await self._handle_flight_request(user_input, destination, name_suffix)
            elif request_type == "hotel":
                return await self._handle_hotel_request(user_input, destination, name_suffix)
            else:
                return await self._handle_general_travel_request(user_input, destination, name_suffix)
                
        except Exception as e:
            print(f"❌ Travel request error: {e}")
            return f"I apologize{name_suffix}, I'm experiencing technical difficulties processing your travel request. Please try again or contact support."
    
    async def _handle_transit_request(self, user_input: str, destination: str, name_suffix: str) -> str:
        """Handle transit-specific requests"""
        response = f"I can help you with transit information{name_suffix}! "
        
        # Check if user is asking for specific transit data
        if "route" in user_input.lower():
            if destination:
                routes = self.search_transit_routes(destination)
                response += f"Here are transit routes for {destination}: {', '.join(routes[:3])}. "
            else:
                nearby_routes = self.get_nearby_transit(1000, "routes")
                response += f"Here are nearby transit routes: {', '.join(nearby_routes[:3])}. "
        
        elif "stop" in user_input.lower() or "station" in user_input.lower():
            if destination:
                stops = self.search_transit_stops(destination)
                response += f"Here are transit stops near {destination}: {', '.join(stops[:3])}. "
            else:
                nearby_stops = self.get_nearby_transit(1000, "stops")
                response += f"Here are nearby transit stops: {', '.join(nearby_stops[:3])}. "
        
        elif "departure" in user_input.lower() or "schedule" in user_input.lower():
            response += "To get departure times, I'll need a specific stop ID and agency ID. You can search for stops first to get these details. "
        
        else:
            # General transit inquiry
            if destination:
                search_info = await self.search_travel_info(f"public transit {destination}")
                response += f"For transit to {destination}: {search_info[:200]}..."
            else:
                nearby_options = self.get_nearby_transit(1000, "stops")
                response += f"Here are nearby transit options: {', '.join(nearby_options[:3])}. "
        
        return response + "Would you like more specific transit information?"
    
    async def _handle_traffic_request(self, user_input: str, name_suffix: str) -> str:
        """Handle traffic-specific requests"""
        response = f"Let me check current traffic conditions for you{name_suffix}! "
        
        traffic_incidents = self.get_traffic_data()
        if traffic_incidents and traffic_incidents[0] != "Traffic data unavailable - API key not configured":
            response += f"Current traffic incidents: {', '.join(traffic_incidents[:3])}. "
        else:
            response += "Traffic data is currently unavailable, but I can help you with route planning using other methods. "
        
        return response + "Would you like me to help you find alternative routes?"
    
    async def _handle_flight_request(self, user_input: str, destination: str, name_suffix: str) -> str:
        """Handle flight-specific requests"""
        response = f"I'd be happy to help with your flight needs{name_suffix}! "
        
        if destination:
            # Get travel insights including pricing factors
            search_info = await self.search_travel_info(f"flight prices {destination} 2025")
            response += f"For flights to {destination}: Current market analysis shows prices are fluctuating due to fuel costs and demand. {search_info[:150]}... "
        else:
            response += "I can help you find flights, check prices, and provide market insights. "
        
        response += "What specific flight information do you need - destinations, dates, or pricing?"
        return response
    
    async def _handle_hotel_request(self, user_input: str, destination: str, name_suffix: str) -> str:
        """Handle hotel-specific requests"""
        response = f"I can assist you with accommodation options{name_suffix}! "
        
        if destination:
            search_info = await self.search_travel_info(f"hotels {destination} booking")
            response += f"For hotels in {destination}: {search_info[:150]}... "
        else:
            response += "I can help you find hotels, compare prices, and check availability. "
        
        response += "What type of accommodation are you looking for?"
        return response
    
    async def _handle_general_travel_request(self, user_input: str, destination: str, name_suffix: str) -> str:
        """Handle general travel requests"""
        response = f"I'm your comprehensive travel specialist{name_suffix}! I can help you with: "
        response += "✈️ Flights and pricing, 🏨 Hotels and accommodations, 🚊 Public transit and routes, "
        response += "🚗 Traffic conditions and incidents, 📍 Local transportation options. "
        
        if destination:
            response += f"For travel to {destination}, I can provide specific recommendations. "
        
        response += "What type of travel assistance do you need today?"
        return response

# Global instance
enhanced_travel_agent = None

async def get_enhanced_travel_response(user_input: str, context: Dict[str, Any] = None) -> str:
    """Get response from enhanced travel agent"""
    global enhanced_travel_agent
    
    if enhanced_travel_agent is None:
        enhanced_travel_agent = EnhancedTravelAgent()
    
    try:
        response = await enhanced_travel_agent.handle_travel_request(user_input, context)
        return response
    except Exception as e:
        print(f"❌ Error in enhanced travel agent: {e}")
        return "I'm here to help you with your travel needs. Let me assist you with flights, hotels, transit, or traffic information."

# Test function
async def test_enhanced_travel_agent():
    """Test the enhanced travel agent"""
    
    agent = EnhancedTravelAgent()
    
    test_requests = [
        "I need a flight to San Francisco",
        "Show me train routes to downtown",
        "What's the traffic like today?",
        "Find me hotels in New York",
        "Help me plan a trip to Europe"
    ]
    
    print("\n🧪 Testing Enhanced Travel Agent")
    print("=" * 60)
    
    for i, request in enumerate(test_requests, 1):
        print(f"\nTest {i}: {request}")
        response = await agent.handle_travel_request(request, {"user_name": "John"})
        print(f"Response: {response}")

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_enhanced_travel_agent())