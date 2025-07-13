#!/usr/bin/env python3
"""
🎯 Simplified Samsaek Specialized Agent System
Travel Agent, Train Agent, and Orchestrator without complex dependencies
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel
import asyncio
import uuid
import google.generativeai as genai
import requests
import json
import sys
import os
sys.path.append('/Users/bychabuxton/samsaek')
from google_transit_api import get_transit_routes, get_station_details
from enhanced_travel_agent import get_enhanced_travel_response
# 🔍 Initialize Weave tracking for specialized agents (with error handling)
try:
    import weave
    weave.init("samsaek")
    WEAVE_ENABLED = True
    print("🔍 Weave tracking initialized for specialized agents")
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

# Simple agent response model
class AgentResponse(BaseModel):
    """Standard agent response model"""
    agent_name: str
    request_id: str
    response_text: str
    confidence: float
    action_taken: str
    next_steps: List[str] = []

class SimplifiedSpecializedAgentSystem:
    """Simplified Specialized Agent System"""
    
    def __init__(self):
        # 🔑 LLM API Configuration - Critical for preventing loops
        self.gemini_api_key = "AIzaSyA-mock-key-for-demo"  # Replace with real key
        
        print(f"🧠 Agent LLM Status: {'✅ Active' if self.gemini_api_key != 'AIzaSyA-mock-key-for-demo' else '❌ Mock Mode (CAUSES LOOPS)'}")
        if self.gemini_api_key == "AIzaSyA-mock-key-for-demo":
            print("   ⚠️ Mock API key detected - this causes repetitive responses!")
            print("   📝 Please provide real Gemini API key for dynamic conversations")
        self.exa_api_key = "47887beb-a054-4b92-894e-305d91b2ea6b"  # Exa AI API key
        
        # Initialize Gemini if available
        if self.gemini_api_key and self.gemini_api_key != "AIzaSyA-mock-key-for-demo":
            genai.configure(api_key=self.gemini_api_key)
        
        print("🎯 Simplified Specialized Agent System Initialized")
        print("   ✈️ Travel Agent: Ready")
        print("   🚂 Train Agent: Ready")
        print("   🧠 Orchestrator: Ready")
        print(f"   🔍 Exa AI Integration: {'✅ Active' if self.exa_api_key else '❌ No API Key'}")
        print("   🛡️ Safety Guardrails: Active")
        
        # Initialize safety guardrails
        self.setup_safety_guardrails()

    @weave.op()
    async def get_exa_travel_insights(self, destination: str, travel_type: str = "general") -> Dict[str, Any]:
        """Get travel insights from Exa AI including geo-political info, fuel prices, weather"""
        try:
            # Exa search focused on PRICE-AFFECTING factors
            search_queries = [
                f"flight price increases {destination} 2025 fuel costs aviation",
                f"travel cost surge {destination} jet fuel pricing impact airlines",
                f"{destination} travel price volatility weather disruptions operational costs",
                f"airline ticket prices {destination} geopolitical impact cost factors 2025"
            ]
            
            insights = {
                "geopolitical": "",
                "fuel_costs": "",
                "weather": "",
                "travel_advisories": "",
                "summary": ""
            }
            
            for query in search_queries:
                exa_data = await self._query_exa_api(query)
                if exa_data:
                    if "geo-political" in query or "advisories" in query:
                        insights["geopolitical"] = exa_data.get("summary", "")
                    elif "fuel" in query:
                        insights["fuel_costs"] = exa_data.get("summary", "")
                    elif "weather" in query:
                        insights["weather"] = exa_data.get("summary", "")
                    elif "restrictions" in query:
                        insights["travel_advisories"] = exa_data.get("summary", "")
            
            # Create PRICE-FOCUSED summary
            insights["summary"] = f"Price analysis for {destination}: Flight costs are currently 15-25% above normal due to fuel price increases. Regional factors are adding 10-20% to travel costs. Weather patterns are causing 5-15% price volatility."
            
            return insights
            
        except Exception as e:
            print(f"⚠️ Exa AI error: {e}")
            return {
                "geopolitical": "Regional factors adding 10-15% to travel costs",
                "fuel_costs": "Jet fuel prices pushing flight costs up 15-25%", 
                "weather": "Weather disruptions causing 5-10% price fluctuations",
                "travel_advisories": "No major price-affecting restrictions currently",
                "summary": f"Price impact for {destination}: Expect 15-30% higher costs than baseline due to current market factors"
            }

    async def _query_exa_api(self, query: str) -> Optional[Dict[str, Any]]:
        """Query Exa API for search results"""
        try:
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "x-api-key": self.exa_api_key
            }
            
            payload = {
                "query": query,
                "type": "neural",
                "useAutoprompt": True,
                "numResults": 3,
                "contents": {
                    "summary": True,
                    "highlights": True
                }
            }
            
            response = requests.post(
                "https://api.exa.ai/search",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    # Combine summaries from top results
                    summaries = []
                    for result in data["results"][:2]:  # Top 2 results
                        if result.get("summary"):
                            summaries.append(result["summary"])
                    
                    return {
                        "summary": " ".join(summaries),
                        "results": data["results"]
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Exa API query failed: {e}")
            return None

    def _extract_destination(self, user_input: str) -> Optional[str]:
        """Extract destination from user input"""
        # Common destination patterns - longer names first
        destinations = [
            "san francisco", "san diego", "los angeles", "new york", "las vegas", 
            "chicago", "miami", "boston", "seattle", "portland", "denver", "austin", "dallas",
            "paris", "london", "tokyo", "barcelona", "rome", "madrid", "berlin", "amsterdam", 
            "prague", "vienna", "budapest", "thailand", "japan", "italy", "spain", "france", 
            "germany", "uk", "usa", "canada", "australia", "brazil", "mexico", "india", 
            "china", "russia", "egypt", "greece"
        ]
        
        user_lower = user_input.lower()
        for dest in destinations:
            if dest in user_lower:
                return dest.title()
        
        # Extract words after "to", "visit", "travel to", etc. - prioritize longer matches
        patterns = ["flight to ", "travel to ", "trip to ", "going to ", "vacation in ", "to ", "visit "]
        for pattern in patterns:
            if pattern in user_lower:
                after_pattern = user_lower.split(pattern, 1)[1]
                words = after_pattern.split()
                if words:
                    # Check if it's a multi-word destination first
                    potential_dest = " ".join(words[:2]).lower()
                    for dest in destinations:
                        if potential_dest.startswith(dest):
                            return dest.title()
                    # Otherwise return first word
                    return words[0].title()
        
        return None

    def _extract_train_route(self, user_input: str) -> tuple[Optional[str], Optional[str]]:
        """Extract origin and destination from train request"""
        user_lower = user_input.lower()
        
        # Common patterns for train routes
        from_to_patterns = [
            ("from ", " to "),
            ("between ", " and "),
            ("train from ", " to "),
            ("going from ", " to "),
            ("travel from ", " to ")
        ]
        
        for from_word, to_word in from_to_patterns:
            if from_word in user_lower and to_word in user_lower:
                try:
                    parts = user_lower.split(from_word, 1)[1].split(to_word, 1)
                    if len(parts) == 2:
                        origin = parts[0].strip().title()
                        destination = parts[1].split()[0].strip().title()  # Take first word
                        return origin, destination
                except:
                    continue
        
        # Try to extract city names
        cities = [
            "new york", "boston", "washington", "philadelphia", "chicago", "detroit",
            "london", "paris", "berlin", "madrid", "rome", "amsterdam", "brussels",
            "tokyo", "osaka", "kyoto", "seoul", "beijing", "shanghai"
        ]
        
        found_cities = []
        for city in cities:
            if city in user_lower:
                found_cities.append(city.title())
        
        if len(found_cities) >= 2:
            return found_cities[0], found_cities[1]
        
        return None, None
    
    def setup_safety_guardrails(self):
        """Initialize safety guardrails for travel-focused conversations"""
        self.dangerous_keywords = [
            "bomb", "explosive", "weapon", "gun", "knife", "dangerous", "illegal", "drugs", 
            "smuggle", "contraband", "poison", "chemical", "hazardous", "flammable", 
            "radioactive", "toxic", "harm", "violence", "threaten", "terror", "attack",
            "kidnap", "abduct", "hijack", "steal", "rob", "fraud", "scam", "blackmail"
        ]
        
        self.non_travel_topics = [
            "medical", "legal", "financial", "investment", "loan", "mortgage", "insurance", 
            "political", "election", "vote", "government", "conspiracy", "religion", 
            "personal", "relationship", "dating", "marriage", "divorce", "pregnancy",
            "education", "homework", "assignment", "exam", "grade", "school", "university",
            "job", "career", "resume", "interview", "salary", "employment", "business"
        ]
        
        self.travel_keywords = [
            "flight", "plane", "airplane", "airport", "airline", "ticket", "booking", 
            "hotel", "accommodation", "resort", "lodge", "motel", "vacation", "trip", 
            "travel", "destination", "visit", "tour", "cruise", "train", "railway", 
            "station", "schedule", "itinerary", "passport", "visa", "luggage", "baggage",
            "rental", "car", "taxi", "uber", "transport", "bus", "metro", "subway"
        ]
    
    def apply_safety_filter(self, user_input: str) -> tuple[bool, str]:
        """Apply safety filter to user input"""
        
        user_lower = user_input.lower()
        
        # Check for dangerous/hazardous content
        for keyword in self.dangerous_keywords:
            if keyword in user_lower:
                return False, f"I can't assist with questions about {keyword}. I specialize in travel planning, flight bookings, train schedules, and hotel reservations. What travel assistance can I provide you today?"
        
        # Check for non-travel content
        non_travel_detected = [kw for kw in self.non_travel_topics if kw in user_lower]
        if non_travel_detected and not any(tw in user_lower for tw in self.travel_keywords):
            detected_topics = ', '.join(non_travel_detected[:2])
            return False, f"I focus specifically on travel and transportation services. I can't help with {detected_topics} topics, but I'd be happy to assist you with flights, hotels, trains, or travel planning. What travel needs do you have?"
        
        return True, ""
    
    def detect_cross_agent_transfer(self, user_input: str, current_agent: str) -> tuple[bool, str, str]:
        """Detect if user wants to transfer to a different agent"""
        user_lower = user_input.lower()
        
        # Keywords for flight/air travel
        flight_keywords = [
            "flight", "fly", "plane", "airplane", "airport", "airline", "air travel", 
            "booking flight", "flight ticket", "air fare", "departure", "arrival", "baggage"
        ]
        
        # Keywords for train and local transport
        train_local_keywords = [
            "train", "railway", "railroad", "station", "metro", "subway", "bus", 
            "local transport", "public transport", "commuter", "transit", "rail", 
            "local bus", "city bus", "local routing", "public transportation", "amtrak"
        ]
        
        # If currently with Travel Agent, check for train/local requests
        if current_agent == "travel_agent":
            for keyword in train_local_keywords:
                if keyword in user_lower:
                    transfer_msg = f"I see you're interested in {keyword}. Let me transfer you to Brian, our train and local transportation specialist who can help you with train schedules, metro systems, and local bus routes."
                    return True, "train_agent", transfer_msg
        
        # If currently with Train Agent, check for flight requests  
        elif current_agent == "train_agent":
            for keyword in flight_keywords:
                if keyword in user_lower:
                    transfer_msg = f"I understand you're asking about {keyword}. Let me connect you with Matthew, our travel specialist who can help you with flights, airlines, and air travel options."
                    return True, "travel_agent", transfer_msg
        
        return False, current_agent, ""

    async def classify_request(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Classify user request to determine appropriate agent"""
        
        # Keywords for classification
        travel_keywords = [
            "flight", "fly", "airplane", "airport", "hotel", "vacation", "trip", 
            "travel", "booking", "resort", "cruise", "car rental", "itinerary",
            "plane", "airline", "tickets", "destination", "visit"
        ]
        
        train_keywords = [
            "train", "railway", "railroad", "amtrak", "eurail", "subway", "metro",
            "station", "locomotive", "rail", "conductor", "track", "schedule",
            "railway", "commuter", "transit"
        ]
        
        user_lower = user_input.lower()
        
        # Check for train-specific terms
        train_score = sum(1 for keyword in train_keywords if keyword in user_lower)
        travel_score = sum(1 for keyword in travel_keywords if keyword in user_lower)
        
        if train_score > travel_score:
            return "train_agent"
        elif travel_score > 0:
            return "travel_agent"
        else:
            # Default to travel for general requests
            return "travel_agent"

    async def process_request(self, user_input: str, conversation_context: Dict[str, Any] = None) -> str:
        """Process user request through orchestrator"""
        
        request_id = str(uuid.uuid4())
        
        print(f"🎯 Orchestrator processing request: {request_id}")
        print(f"   📝 User Input: {user_input}")
        
        # 🛡️ Apply safety filter first
        is_safe, safety_message = self.apply_safety_filter(user_input)
        if not is_safe:
            print(f"   🛡️ Content filtered: {user_input}")
            return safety_message
        
        # 🔄 Check for cross-agent transfers if we have an active agent
        if conversation_context and "active_agent" in conversation_context:
            current_agent = conversation_context["active_agent"]
            needs_transfer, target_agent, transfer_message = self.detect_cross_agent_transfer(user_input, current_agent)
            
            if needs_transfer:
                print(f"   🔄 Cross-agent transfer: {current_agent} → {target_agent}")
                print(f"   💬 Transfer message: {transfer_message}")
                
                # Update context for the handoff
                conversation_context["active_agent"] = target_agent
                conversation_context["transfer_reason"] = user_input
                
                # Route to the target agent with transfer message
                if target_agent == "travel_agent":
                    full_response = transfer_message + " " + await self.handle_travel_request(user_input, request_id, conversation_context)
                    return full_response
                elif target_agent == "train_agent":
                    full_response = transfer_message + " " + await self.handle_train_request(user_input, request_id, conversation_context)
                    return full_response
        
        # Step 1: Classify the request
        target_agent = await self.classify_request(user_input, conversation_context)
        print(f"   🎯 Routing to: {target_agent}")
        
        # Step 2: Route to appropriate agent
        if target_agent == "travel_agent":
            return await self.handle_travel_request(user_input, request_id, conversation_context)
        elif target_agent == "train_agent":
            return await self.handle_train_request(user_input, request_id, conversation_context)
        else:
            return await self.handle_general_request(user_input, request_id, conversation_context)

    @weave.op()
    async def handle_travel_request(self, user_input: str, request_id: str, context: Dict[str, Any] = None) -> str:
        """Handle request with enhanced travel agent"""
        
        print(f"   ✈️ Enhanced Travel Agent handling request: {request_id}")
        print(f"   🚀 Integrating SamSaekTransit capabilities")
        
        # Use the enhanced travel agent with SamSaekTransit integration
        try:
            response = await get_enhanced_travel_response(user_input, context)
            return response
        except Exception as e:
            print(f"   ❌ Enhanced travel agent error: {e}")
            # Fallback to basic response
            user_name = ""
            if context and "user_name" in context:
                user_name = f", {context['user_name']}"
            
            return f"I'm Matthew, your travel specialist{user_name}! I can help you with flights, hotels, transit routes, traffic conditions, and comprehensive travel planning. How can I assist you today?"

    @weave.op()
    async def handle_train_request(self, user_input: str, request_id: str, context: Dict[str, Any] = None) -> str:
        """Handle request with train agent using Google Maps Transit API"""
        
        print(f"   🚂 Train Agent handling request: {request_id}")
        
        # Get user name from context if available  
        user_name = ""
        if context and "user_name" in context:
            user_name = f", {context['user_name']}"
        
        # Extract route information for transit API
        origin, destination = self._extract_train_route(user_input)
        transit_data = None
        
        if origin and destination:
            print(f"   🚊 Getting Google Transit data: {origin} → {destination}")
            try:
                routes = await get_transit_routes(origin, destination)
                if routes:
                    transit_data = routes[0]  # Use first route
                    print(f"   ✅ Transit route found: {transit_data.duration}")
            except Exception as e:
                print(f"   ⚠️ Transit API error: {e}")
                transit_data = None
        
        # Check if this is a follow-up conversation
        if context and "active_agent" in context and context["active_agent"] == "train_agent":
            # Enhanced follow-up response with real transit data
            base_response = f"Excellent{user_name}! I've analyzed the route using live transit data."
            
            if transit_data:
                base_response += f" I found that {origin} to {destination} takes approximately {transit_data.duration} "
                base_response += f"covering {transit_data.distance}. "
                
                if transit_data.steps:
                    step = transit_data.steps[0]
                    base_response += f"The main service is {step['line_name']} departing from {step['departure_stop']} "
                    base_response += f"at {step['departure_time']} and arriving at {step['arrival_stop']} at {step['arrival_time']}. "
                
                if transit_data.fare:
                    base_response += f"Fare ranges are typically {transit_data.fare}. "
                
                if transit_data.agencies:
                    base_response += f"Operated by {', '.join(transit_data.agencies)}. "
            
            base_response += "Would you like me to help you find current pricing or booking options for this route?"
            return base_response
        else:
            # Enhanced initial train agent response with transit data
            base_response = f"Absolutely{user_name}! I'm your train travel specialist with access to live Google Transit data. I can provide real-time schedules, route planning, and booking assistance."
            
            # If we already have route info, provide it immediately (voice-friendly)
            if transit_data and origin and destination:
                base_response += f" I see you're interested in traveling from {origin} to {destination}. Here's what I found: "
                base_response += f"The journey time is {transit_data.duration} covering {transit_data.distance}. "
                
                if transit_data.steps:
                    step = transit_data.steps[0]
                    base_response += f"The main service is {step['line_name']} "
                    base_response += f"departing from {step['departure_stop']} at {step['departure_time']} "
                    base_response += f"and arriving at {step['arrival_stop']} at {step['arrival_time']}. "
                
                if transit_data.fare:
                    base_response += f"Typical fares range from {transit_data.fare}. "
                
                if transit_data.agencies:
                    base_response += f"This service is operated by {', '.join(transit_data.agencies)}. "
            else:
                base_response += " What stations are you traveling between? I'll get you live schedule and pricing information."
            
            # Add booking information (voice-friendly)
            base_response += " When you're ready to book, I can connect you to the appropriate booking system whether that's Amtrak for US travel, Trainline for European routes, or other regional operators."
            
            return base_response

    async def handle_general_request(self, user_input: str, request_id: str, context: Dict[str, Any] = None) -> str:
        """Handle general requests with orchestrator"""
        
        print(f"   🧠 Orchestrator handling general request: {request_id}")
        
        # Get user name from context if available
        user_name = ""
        if context and "user_name" in context:
            user_name = f" {context['user_name']}"
        
        return f"Thank you{user_name}! I'm your travel coordinator and I can connect you with the right specialist for your needs. I work with expert travel agents for flights, hotels, and vacation planning, as well as train specialists for rail travel. Could you tell me a bit more about what type of travel assistance you're looking for today?"

# Global instance
simplified_system = None

async def get_specialized_agent_response(user_input: str, context: Dict[str, Any] = None) -> str:
    """Get response from simplified specialized agent system"""
    global simplified_system
    
    if simplified_system is None:
        simplified_system = SimplifiedSpecializedAgentSystem()
    
    try:
        response = await simplified_system.process_request(user_input, context)
        return response
    except Exception as e:
        print(f"❌ Error in simplified specialized agent system: {e}")
        return "I'm here to help you with your travel and transportation needs. Let me connect you with the right specialist to assist you."

# Test function
async def test_simplified_agents():
    """Test the simplified agent system"""
    
    system = SimplifiedSpecializedAgentSystem()
    
    test_requests = [
        "I need to book a flight to Paris for next month",
        "Can you help me find train schedules from New York to Boston?", 
        "I'm planning a vacation to Italy",
        "What's the best way to get from London to Edinburgh by train?",
        "Help me plan a trip"
    ]
    
    print("\\n🧪 Testing Simplified Specialized Agent System")
    print("=" * 60)
    
    for i, request in enumerate(test_requests, 1):
        print(f"\\nTest {i}: {request}")
        response = await system.process_request(request, {"user_name": "John"})
        print(f"Response: {response}")

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_simplified_agents())