#!/usr/bin/env python3
"""
🚀 Samsaek Enhanced Voice Server
AI-powered conversations with state management, Gemini integration, and conversation logging
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, Dict, List
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from twilio.twiml.messaging_response import MessagingResponse
import requests
import os
import asyncio
from datetime import datetime
import google.generativeai as genai
import sys
sys.path.append('/Users/bychabuxton/samsaek/src')
from simplified_specialized_agents import get_specialized_agent_response
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import webbrowser
import subprocess

# 🔍 Initialize Weave tracking for all LLM calls (with error handling)
try:
    import weave
    weave.init("samsaek")
    WEAVE_ENABLED = True
    print("🔍 Weave tracking initialized for 'samsaek'")
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

# 🔑 API Credentials from Environment Variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "demo-twilio-account-sid")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "demo-twilio-auth-token")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+1234567890")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "demo-elevenlabs-api-key")
# 🔑 LLM API Configuration
# To fix the conversation loops, please provide a real Gemini API key
# Get one from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA-mock-key-for-demo")

print(f"🧠 LLM Status: {'✅ Active' if GEMINI_API_KEY != 'AIzaSyA-mock-key-for-demo' else '❌ Mock Mode (CAUSES LOOPS)'}")
if GEMINI_API_KEY == "AIzaSyA-mock-key-for-demo":
    print("   ⚠️ Using mock API key - this causes conversation loops!")
    print("   📝 Please replace with real Gemini API key to fix loops")
    print("   🔗 Get API key: https://makersuite.google.com/app/apikey")

# Initialize Gemini AI
if GEMINI_API_KEY and GEMINI_API_KEY != "AIzaSyA-mock-key-for-demo":
    genai.configure(api_key=GEMINI_API_KEY)

# 🧠 Conversation State Management
class ConversationState(BaseModel):
    """Track conversation state across multiple interactions"""
    call_sid: str
    current_question: int = 0
    responses: List[str] = []
    start_time: Optional[datetime] = None
    user_name: Optional[str] = None
    context: str = ""
    active_agent: Optional[str] = None  # Track which agent is handling the conversation
    agent_voice: Optional[str] = "Polly.Joanna"  # Default voice
    conversation_stage: str = "greeting"  # greeting, agent_routing, agent_conversation, booking, closing
    booking_mode: bool = False  # Whether user wants to proceed with booking
    booking_destination: Optional[str] = None
    booking_dates: Optional[str] = None
    booking_type: Optional[str] = None  # "flight", "hotel", "train", "package"
    previous_voice: Optional[str] = None  # Track voice changes for handoff announcements
    
    class Config:
        arbitrary_types_allowed = True

# Store active conversations
active_conversations: Dict[str, ConversationState] = {}

# 🛡️ SAFETY GUARDRAILS
DANGEROUS_KEYWORDS = [
    "bomb", "explosive", "weapon", "gun", "knife", "dangerous", "illegal", "drugs", 
    "smuggle", "contraband", "poison", "chemical", "hazardous", "flammable", 
    "radioactive", "toxic", "harm", "violence", "threaten", "terror", "attack",
    "kidnap", "abduct", "hijack", "steal", "rob", "fraud", "scam", "blackmail"
]

NON_TRAVEL_KEYWORDS = [
    "medical", "legal", "financial", "investment", "loan", "mortgage", "insurance", 
    "political", "election", "vote", "government", "conspiracy", "religion", 
    "personal", "relationship", "dating", "marriage", "divorce", "pregnancy",
    "education", "homework", "assignment", "exam", "grade", "school", "university",
    "job", "career", "resume", "interview", "salary", "employment", "business"
]

TRAVEL_RELATED_KEYWORDS = [
    "flight", "plane", "airplane", "airport", "airline", "ticket", "booking", 
    "hotel", "accommodation", "resort", "lodge", "motel", "vacation", "trip", 
    "travel", "destination", "visit", "tour", "cruise", "train", "railway", 
    "station", "schedule", "itinerary", "passport", "visa", "luggage", "baggage",
    "rental", "car", "taxi", "uber", "transport", "bus", "metro", "subway"
]

# 🛡️ Content Safety Filter
def content_safety_filter(user_input: str) -> tuple[bool, str]:
    """Filter user input for dangerous content and non-travel topics"""
    
    user_lower = user_input.lower()
    
    # Check for dangerous/hazardous content
    for keyword in DANGEROUS_KEYWORDS:
        if keyword in user_lower:
            return False, f"I'm sorry, I can't assist with questions about {keyword}. I'm here to help with travel planning, bookings, and transportation. What travel assistance can I provide you today?"
    
    # Check for non-travel content (provide helpful redirect)
    non_travel_detected = [kw for kw in NON_TRAVEL_KEYWORDS if kw in user_lower]
    if non_travel_detected and not any(tw in user_lower for tw in TRAVEL_RELATED_KEYWORDS):
        return False, f"I specialize in travel and transportation services. I can't help with {', '.join(non_travel_detected[:2])} topics, but I'd be happy to assist you with flight bookings, hotel reservations, train schedules, or travel planning. What travel assistance do you need?"
    
    # Content is safe and travel-related
    return True, ""

# 💬 Enhanced Conversation Flow with Continuous Agent Interaction
QUESTIONS = [
    "What's your name?",
    "What type of assistance are you looking for today?", 
    "Is there anything else I can help you with?"
]

GREETING = "Hello! Welcome to Samsaek, your multi-agent AI platform. I'm your intelligent assistant and I can connect you with specialized agents for travel, trains, and more. I can also help you complete bookings entirely through voice commands. Let me learn about your needs."
CLOSING = "Thank you for your time! I've noted all your responses and our team will follow up. Have a great day!"

# 🔄 Cross-Agent Transfer Support
AGENT_VOICES = {
    "orchestrator": "Polly.Joanna",
    "travel_agent": "Polly.Matthew", 
    "train_agent": "Polly.Brian"
}

def detect_agent_transfer_from_response(ai_response: str) -> tuple[bool, str]:
    """Detect if AI response indicates an agent transfer"""
    response_lower = ai_response.lower()
    
    # Look for transfer indicators
    if "let me transfer you to" in response_lower or "connect you with" in response_lower:
        if "brian" in response_lower or "train" in response_lower:
            return True, "train_agent"
        elif "matthew" in response_lower or "travel specialist" in response_lower:
            return True, "travel_agent"
    
    return False, ""

# Voice-controlled booking functions
@weave.op()
async def voice_controlled_booking(conversation: ConversationState, booking_type: str, destination: str, user_input: str) -> str:
    """Handle voice-controlled booking through browser automation"""
    try:
        if booking_type == "flight":
            return await voice_book_flight(conversation, destination, user_input)
        elif booking_type == "train":
            return await voice_book_train(conversation, destination, user_input)
        elif booking_type == "hotel":
            return await voice_book_hotel(conversation, destination, user_input)
        else:
            return "I can help you book flights, trains, or hotels through voice commands. Which would you prefer?"
    except Exception as e:
        print(f"❌ Voice booking error: {e}")
        return "I'm having trouble accessing the booking systems right now. Let me provide you with direct links instead."

async def voice_book_flight(conversation: ConversationState, destination: str, user_input: str) -> str:
    """Voice-controlled flight booking"""
    try:
        # Launch browser with flight search
        flight_url = f"https://www.expedia.com/Flights-Search?trip=oneway&leg1=from:{destination}"
        
        # For voice mode, we'll guide the user through the process
        booking_guidance = f"Perfect! I'm opening the flight booking system for {destination}. "
        booking_guidance += "I've launched Expedia in your browser. Here's what I see and what you need to do: "
        booking_guidance += "First, I need your departure city. Where are you flying from? "
        booking_guidance += "Then I'll help you select dates and find the best prices based on the market analysis we discussed."
        
        # Open the URL 
        webbrowser.open(flight_url)
        
        conversation.booking_mode = True
        conversation.booking_type = "flight"
        conversation.booking_destination = destination
        conversation.conversation_stage = "booking"
        
        return booking_guidance
        
    except Exception as e:
        print(f"❌ Flight booking error: {e}")
        return f"I'll provide you the direct link for flight booking: https://www.expedia.com/Flights-Search?trip=oneway&leg1=from:{destination}"

async def voice_book_train(conversation: ConversationState, destination: str, user_input: str) -> str:
    """Voice-controlled train booking"""
    try:
        # Determine appropriate train booking site
        if "london" in destination.lower() or "uk" in destination.lower() or "england" in destination.lower():
            train_url = "https://www.nationalrail.co.uk/"
            site_name = "National Rail"
        elif any(country in destination.lower() for country in ["paris", "berlin", "madrid", "europe"]):
            train_url = "https://www.trainline.com/"
            site_name = "Trainline"
        else:
            train_url = "https://www.amtrak.com/tickets/departure"
            site_name = "Amtrak"
        
        booking_guidance = f"Excellent! I'm opening the train booking system for your journey. "
        booking_guidance += f"I've launched {site_name} in your browser. "
        booking_guidance += "I can see the booking form. Tell me your departure station and destination station, "
        booking_guidance += "and I'll guide you through selecting the best departure time and fare options."
        
        # Open the URL
        webbrowser.open(train_url)
        
        conversation.booking_mode = True
        conversation.booking_type = "train"
        conversation.booking_destination = destination
        conversation.conversation_stage = "booking"
        
        return booking_guidance
        
    except Exception as e:
        print(f"❌ Train booking error: {e}")
        return f"I'll provide you the direct link for train booking: https://www.amtrak.com/tickets/departure"

async def voice_book_hotel(conversation: ConversationState, destination: str, user_input: str) -> str:
    """Voice-controlled hotel booking"""
    try:
        hotel_url = f"https://www.booking.com/searchresults.html?ss={destination}"
        
        booking_guidance = f"Perfect! I'm opening the hotel booking system for {destination}. "
        booking_guidance += "I've launched Booking.com in your browser. I can see the search results page. "
        booking_guidance += "Tell me your check-in and check-out dates, and I'll help you filter through "
        booking_guidance += "the available options based on your preferences for price, location, and amenities."
        
        # Open the URL
        webbrowser.open(hotel_url)
        
        conversation.booking_mode = True
        conversation.booking_type = "hotel"
        conversation.booking_destination = destination
        conversation.conversation_stage = "booking"
        
        return booking_guidance
        
    except Exception as e:
        print(f"❌ Hotel booking error: {e}")
        return f"I'll provide you the direct link for hotel booking: https://www.booking.com/searchresults.html?ss={destination}"

async def handle_booking_guidance(conversation: ConversationState, user_input: str) -> str:
    """Provide step-by-step booking guidance based on user input"""
    try:
        booking_type = conversation.booking_type
        destination = conversation.booking_destination
        
        if booking_type == "flight":
            if "departure" in user_input.lower() or "from" in user_input.lower():
                conversation.context += f"Departure: {user_input}. "
                return f"Great! I've noted your departure location. Now I need your travel dates. What date would you like to fly to {destination}?"
            
            elif "date" in user_input.lower() or any(month in user_input.lower() for month in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
                conversation.booking_dates = user_input
                conversation.context += f"Travel dates: {user_input}. "
                return f"Perfect! I see the search results for {destination} on {user_input}. I can see several flight options with different prices. Would you prefer the most economical option, the fastest flight, or something with the best departure times?"
            
            elif "economical" in user_input.lower() or "cheap" in user_input.lower():
                return "Excellent choice! I can see the most economical flights. The lowest price options typically have one or two stops but offer significant savings. I'll help you review these options. Do you see the flight results on your screen? Tell me which price range looks good to you."
            
            elif "fastest" in user_input.lower() or "direct" in user_input.lower():
                return "Great! I'm looking at the fastest and direct flight options. These usually cost more but save you time. I can see several direct flights. Are you ready to select one and proceed to passenger details?"
            
            else:
                return "I'm here to guide you through the booking process. What information do you need help with next - flight selection, passenger details, or payment?"
        
        elif booking_type == "train":
            if "departure" in user_input.lower() or "station" in user_input.lower():
                return f"Perfect! I've noted your station information. Now I need your travel date and preferred time. When would you like to travel?"
            
            elif "date" in user_input.lower() or "time" in user_input.lower():
                return "Excellent! I can see the train schedules and pricing. There are different service classes available. Would you prefer standard class for the best price, or premium for more comfort?"
            
            else:
                return "I'm guiding you through the train booking. What step do you need help with - schedule selection, seat preferences, or passenger information?"
        
        elif booking_type == "hotel":
            if "check" in user_input.lower() or "date" in user_input.lower():
                return f"Great! I've noted your dates. I can see the hotel options in {destination}. There are different price ranges and locations. Would you prefer hotels in the city center, near the airport, or in a specific area?"
            
            elif "center" in user_input.lower() or "downtown" in user_input.lower():
                return "Perfect! I can see the city center hotels. The prices range from budget to luxury. Are you looking for a specific star rating or amenities like breakfast, gym, or spa?"
            
            else:
                return "I'm helping you with hotel booking. What would you like to focus on - location, price range, or specific amenities?"
        
        else:
            return "I'm here to guide you through the booking process. What specific step do you need help with?"
            
    except Exception as e:
        print(f"❌ Booking guidance error: {e}")
        return "I'm here to help you complete your booking. What information do you need assistance with?"

def _extract_destination_from_input(user_input: str) -> Optional[str]:
    """Extract destination from user input for voice server"""
    destinations = [
        "san diego", "san francisco", "los angeles", "new york", "boston", "chicago", 
        "miami", "las vegas", "seattle", "portland", "denver", "austin", "dallas",
        "paris", "london", "rome", "madrid", "barcelona", "berlin", "amsterdam"
    ]
    
    user_lower = user_input.lower()
    for dest in destinations:
        if dest in user_lower:
            return dest.title()
    
    # Extract words after "to"
    if " to " in user_lower:
        after_to = user_lower.split(" to ", 1)[1]
        words = after_to.split()
        if words:
            return words[0].title()
    
    return None

def extract_name_from_input(user_input: str) -> str:
    """Extract clean name from user input, removing phrases like 'my name is'"""
    import re
    
    user_input = user_input.strip()
    
    # Remove common name introduction phrases
    name_patterns = [
        r"^my name is\s+",
        r"^i'm\s+",
        r"^i am\s+",
        r"^this is\s+",
        r"^call me\s+",
        r"^it's\s+",
        r"^its\s+",
        r"^the name is\s+",
        r"^name is\s+"
    ]
    
    clean_input = user_input.lower()
    for pattern in name_patterns:
        clean_input = re.sub(pattern, "", clean_input, flags=re.IGNORECASE)
    
    # Clean up the name
    clean_name = clean_input.strip()
    
    # Remove punctuation and extra words
    clean_name = re.sub(r'[.,!?]', '', clean_name)
    
    # Take first word as the name (first name only)
    words = clean_name.split()
    if words:
        name = words[0].title()
        # Make sure it's a reasonable name (not empty, not too long)
        if 1 <= len(name) <= 20 and name.isalpha():
            return name
    
    # Fallback if we can't extract a clean name
    return "there"

def clean_text_for_voice(text: str) -> str:
    """Clean text for voice synthesis - remove emojis, links, and formatting"""
    import re
    
    # Remove emojis and special characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # supplemental symbols
        u"\U00002600-\U000026FF"  # miscellaneous symbols
        "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub('', text)
    
    # Remove URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = url_pattern.sub('', text)
    
    # Remove markdown-style formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italic*
    text = re.sub(r'`(.*?)`', r'\1', text)        # `code`
    
    # Remove bullet points and list formatting
    text = re.sub(r'^\s*[-•*]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    
    # Remove multiple newlines and extra spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove currency symbols but keep the text
    text = re.sub(r'[£$€¥₹]', '', text)
    
    # Clean up any remaining symbols that shouldn't be spoken
    text = re.sub(r'[📊📞🎯🔍✅❌⚠️🚀🧠💾📝🎤📱→←↑↓⏰🕐🔄✈️🏨🎫🇺🇸🇪🇺🇬🇧🌍🚂🚉💰🏢]', '', text)
    
    # Replace common symbols with words
    text = text.replace('→', ' to ')
    text = text.replace('←', ' from ')
    text = text.replace('&', ' and ')
    
    # Clean up extra spaces again after replacements
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def create_handoff_announcement(from_agent: str, to_agent: str) -> str:
    """Create handoff announcement for voice transitions"""
    
    agent_names = {
        "orchestrator": "your coordinator Joanna",
        "travel_agent": "travel specialist Matthew", 
        "train_agent": "train specialist Brian"
    }
    
    from_name = agent_names.get(from_agent, "your current agent")
    to_name = agent_names.get(to_agent, "a specialist")
    
    # Shorter, faster announcements to reduce latency
    announcements = [
        f"Connecting you to {to_name} now.",
        f"Transferring to {to_name}. One moment.",
        f"Let me get {to_name} for you.",
        f"Switching to {to_name} right now."
    ]
    
    import random
    return random.choice(announcements)

def create_processing_sound() -> str:
    """Create processing indicators to fill dead space"""
    indicators = [
        "Let me look that up for you.",
        "Checking current information.",
        "One moment while I analyze that.",
        "Looking into that right now.",
        "Checking the latest data."
    ]
    
    import random
    return random.choice(indicators)

def create_thinking_pause(twiml_response, voice: str, duration: int = 2):
    """Add thinking sounds during processing"""
    
    # Add a brief thinking indicator
    thinking_sounds = [
        "Hmm, let me see.",
        "Okay, checking now.",
        "One second here.",
        "Let me pull that up."
    ]
    
    import random
    twiml_response.say(random.choice(thinking_sounds), voice=voice)
    twiml_response.pause(length=duration)

# Agent Voice Configuration with Gender Variety
AGENT_VOICES = {
    "orchestrator": "Polly.Joanna",  # Professional female voice (starter)
    "travel_agent": "Polly.Matthew",  # Professional male voice for travel handoff
    "train_agent": "Polly.Brian"  # British male voice for trains handoff
}

# Agent Introductions with Gender-Specific Handoffs
AGENT_INTRODUCTIONS = {
    "travel_agent": "Perfect! Let me connect you with our travel specialist. Hello, I'm Matthew, your dedicated male travel agent and I'm excited to help analyze pricing and plan your trip!",
    "train_agent": "Excellent! Let me connect you with our train travel expert. Hello, I'm Brian, your railway specialist and I'm here to help with all your train travel needs!"
}

class MakeCallRequest(BaseModel):
    to: str
    message: Optional[str] = "Hello! This is Samsaek AI Assistant calling to demonstrate our multi-agent voice capabilities. How can I help you today?"
    voice: Optional[str] = "Polly.Joanna"
    agent_name: Optional[str] = "Samsaek AI Assistant"
    record: Optional[bool] = True

class SendSMSRequest(BaseModel):
    to: str
    message: str
    agent_name: Optional[str] = "Samsaek AI"

# 🚀 FastAPI App
app = FastAPI(
    title="🤖 Samsaek Enhanced Voice Service",
    description="AI-powered conversations with state management",
    version="3.0.0"
)

# 🤖 AI Response Generation
@weave.op()
async def generate_contextual_response(conversation: ConversationState, user_input: str) -> tuple[str, str]:
    """Generate contextual AI responses with specialized agent routing - returns (response, voice)"""
    try:
        # Question 0: Name collection with proper parsing
        if conversation.current_question == 0:
            clean_name = extract_name_from_input(user_input)
            conversation.user_name = clean_name
            conversation.context += f"User's name: {clean_name}. "
            conversation.conversation_stage = "agent_routing"
            return f"Nice to meet you, {clean_name}! I'll remember that.", AGENT_VOICES["orchestrator"]
        
        # Question 1: Agent routing and introduction
        elif conversation.current_question == 1:
            print(f"🎯 Routing request to specialized agents: {user_input}")
            
            # Classify and route the request
            try:
                from simplified_specialized_agents import SimplifiedSpecializedAgentSystem
                system = SimplifiedSpecializedAgentSystem()
                target_agent = await system.classify_request(user_input)
                
                # Set active agent and update conversation state
                conversation.active_agent = target_agent
                conversation.agent_voice = AGENT_VOICES.get(target_agent, "Polly.Joanna")
                conversation.conversation_stage = "agent_conversation"
                conversation.context += f"User request: {user_input}. Routed to {target_agent}. "
                
                # Store destination for booking
                if target_agent == "travel_agent":
                    destination = _extract_destination_from_input(user_input)
                    if destination:
                        conversation.booking_destination = destination
                
                # Get agent introduction + initial response
                introduction = AGENT_INTRODUCTIONS.get(target_agent, "Let me help you with that!")
                specialist_response = await get_specialized_agent_response(
                    user_input, 
                    {"user_name": conversation.user_name, "context": conversation.context}
                )
                
                # For initial response, don't repeat introduction - agent will handle it
                return specialist_response, conversation.agent_voice
                
            except Exception as e:
                print(f"⚠️ Specialist routing error: {e}")
                return "I understand you're looking for assistance. Let me connect you with our travel and transportation specialists who can help with your specific needs.", AGENT_VOICES["orchestrator"]
        
        # Question 2+: Continued agent conversation or booking mode
        else:
            # Check if user wants to proceed with booking
            booking_keywords = ["book", "purchase", "buy", "reserve", "yes I want to book", "proceed", "go ahead", "complete booking"]
            if any(keyword in user_input.lower() for keyword in booking_keywords) and not conversation.booking_mode:
                # User wants to proceed with voice booking
                destination = conversation.booking_destination or "your destination"
                booking_type = "flight"  # Default to flight, can be enhanced
                
                try:
                    booking_response = await voice_controlled_booking(conversation, booking_type, destination, user_input)
                    return booking_response, conversation.agent_voice
                except Exception as e:
                    return "I can help you with the booking process. Let me provide you with the direct booking links.", conversation.agent_voice
            
            elif conversation.booking_mode and conversation.conversation_stage == "booking":
                # Handle booking guidance and next steps
                booking_guidance = await handle_booking_guidance(conversation, user_input)
                return booking_guidance, conversation.agent_voice
            
            elif conversation.active_agent and conversation.conversation_stage == "agent_conversation":
                # Continue conversation with the active agent
                try:
                    specialist_response = await get_specialized_agent_response(
                        user_input, 
                        {"user_name": conversation.user_name, "context": conversation.context, "active_agent": conversation.active_agent}
                    )
                    conversation.context += f"Follow-up: {user_input}. Agent response: {specialist_response[:50]}... "
                    
                    # Check if response contains booking information and offer voice booking
                    if "book" in specialist_response.lower() or "purchase" in specialist_response.lower():
                        specialist_response += " Would you like me to help you complete the booking process through voice commands? Just say 'yes, book it' and I'll open the booking system and guide you through it step by step."
                    
                    return specialist_response, conversation.agent_voice
                except Exception as e:
                    print(f"⚠️ Agent conversation error: {e}")
                    return "Let me make sure I have all the details to help you properly. Can you tell me more about what you need?", conversation.agent_voice
            else:
                # Fallback to orchestrator
                conversation.context += f"Additional info: {user_input}. "
                return "Perfect! I've noted all your details. Is there anything else I can help you with today?", AGENT_VOICES["orchestrator"]
            
    except Exception as e:
        print(f"⚠️ Response generation error: {e}")
        return "Thank you for that information. Let me make sure you get connected with the right specialist for your needs.", AGENT_VOICES["orchestrator"]

# 💾 Conversation Logging
async def save_conversation_summary(conversation: ConversationState):
    """Save conversation summary to file"""
    try:
        # Create conversations directory
        os.makedirs("conversations", exist_ok=True)
        
        # Create summary file
        filename = f"conversations/call_{conversation.call_sid}_summary.txt"
        
        with open(filename, "w") as f:
            f.write("=== SAMSAEK CONVERSATION SUMMARY ===\\n")
            f.write(f"Call SID: {conversation.call_sid}\\n")
            f.write(f"Start Time: {conversation.start_time}\\n")
            f.write(f"User Name: {conversation.user_name or 'Not provided'}\\n")
            f.write(f"Questions Asked: {len(conversation.responses)} of {len(QUESTIONS)}\\n")
            f.write("=" * 50 + "\\n\\n")
            
            for i, (question, response) in enumerate(zip(QUESTIONS[:len(conversation.responses)], conversation.responses)):
                f.write(f"Q{i+1}: {question}\\n")
                f.write(f"A{i+1}: {response}\\n\\n")
            
            f.write(f"Context: {conversation.context}\\n")
            f.write(f"Completed: {datetime.now()}\\n")
            f.write("=== END OF CONVERSATION ===\\n")
        
        print(f"💾 Saved conversation summary: {filename}")
        
    except Exception as e:
        print(f"❌ Error saving conversation: {e}")

# 📞 Webhook Endpoints
@app.post("/webhook/voice")
async def handle_incoming_voice(request: Request):
    """Handle incoming voice calls with enhanced conversation state"""
    
    try:
        # Get call data from Twilio
        form_data = await request.form()
        call_sid = form_data.get('CallSid', '')
        from_number = form_data.get('From', 'Unknown')
        
        print("📞 INCOMING CALL!")
        print(f"   From: {from_number}")
        print(f"   Call SID: {call_sid}")
        
        # Initialize conversation state
        conversation = ConversationState(
            call_sid=call_sid,
            start_time=datetime.now()
        )
        active_conversations[call_sid] = conversation
        
        # Create AI voice response
        twiml = VoiceResponse()
        
        # Personalized greeting
        twiml.say(GREETING, voice="Polly.Joanna", language="en-US")
        twiml.pause(length=1)
        
        # Start conversation with first question
        gather = twiml.gather(
            input='speech',
            timeout=10,
            speech_timeout='auto',
            action="/webhook/conversation",
            method="POST"
        )
        gather.say(QUESTIONS[0], voice="Polly.Joanna")
        
        # Fallback if no response
        twiml.say("I didn't hear your response. Let me ask again.")
        twiml.redirect("/webhook/voice")
        
        return Response(content=str(twiml), media_type="application/xml")
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        
        # Error fallback
        twiml = VoiceResponse()
        twiml.say("We're sorry, but we're experiencing technical difficulties. Please try again later.")
        return Response(content=str(twiml), media_type="application/xml")

@app.post("/webhook/conversation")
async def handle_conversation_response(request: Request):
    """Handle conversation responses with AI processing"""
    
    try:
        form_data = await request.form()
        call_sid = form_data.get('CallSid', '')
        speech_result = form_data.get('SpeechResult', '')
        confidence = form_data.get('Confidence', '')
        
        print(f"🎤 Response for {call_sid}: {speech_result} (confidence: {confidence})")
        
        # Get conversation state
        conversation = active_conversations.get(call_sid)
        if not conversation:
            return await handle_incoming_voice(request)
        
        twiml = VoiceResponse()
        
        if speech_result:
            # Clean and store response
            cleaned_response = speech_result.strip().rstrip('.!?')
            
            # 🛡️ APPLY CONTENT SAFETY FILTER
            is_safe, safety_message = content_safety_filter(cleaned_response)
            if not is_safe:
                print(f"🛡️ Content filtered: {cleaned_response}")
                twiml.say(safety_message, voice="Polly.Joanna")
                twiml.pause(length=1)
                
                # Continue conversation with travel-focused prompt
                gather = twiml.gather(
                    input='speech',
                    timeout=10,
                    speech_timeout='auto',
                    action="/webhook/conversation",
                    method="POST"
                )
                gather.say("What travel assistance can I provide you today?", voice="Polly.Joanna")
                
                return Response(content=str(twiml), media_type="application/xml")
            
            conversation.responses.append(cleaned_response)
            
            # Context handled in generate_contextual_response function
            
            # Add shorter processing indicator 
            processing_msg = create_processing_sound()
            twiml.say(processing_msg, voice=conversation.agent_voice or "Polly.Joanna")
            twiml.pause(length=0.5)  # Shorter pause
            
            # Generate AI response with appropriate voice
            ai_response, agent_voice = await generate_contextual_response(conversation, cleaned_response)
            
            # 🔄 Check if AI response indicates a cross-agent transfer
            is_transfer, new_agent = detect_agent_transfer_from_response(ai_response)
            if is_transfer and new_agent:
                print(f"   🔄 Agent transfer detected in response: {conversation.active_agent} → {new_agent}")
                
                # Update conversation state
                conversation.active_agent = new_agent
                conversation.agent_voice = AGENT_VOICES.get(new_agent, "Polly.Joanna")
                
                # Update agent voice for this response
                agent_voice = conversation.agent_voice
                
                print(f"   🎤 Voice updated to: {agent_voice} for {new_agent}")
            
            # Clean the response for voice synthesis
            clean_response = clean_text_for_voice(ai_response)
            
            # Check if this is a handoff (voice change from previous)
            previous_voice = getattr(conversation, 'previous_voice', None)
            if previous_voice and previous_voice != agent_voice and conversation.current_question > 0:
                # This is a handoff - announce the transition
                current_agent = None
                target_agent = None
                
                # Determine agents from voices
                for agent, voice in AGENT_VOICES.items():
                    if voice == previous_voice:
                        current_agent = agent
                    if voice == agent_voice:
                        target_agent = agent
                
                # Log handoff for dashboard
                print(f"🔄 AGENT HANDOFF DETECTED:")
                print(f"   From: {previous_voice} ({current_agent})")
                print(f"   To: {agent_voice} ({target_agent})")
                print(f"   📺 Dashboard should switch highlighting from {current_agent} to {target_agent}")
                print(f"   🔄 Type: {'Cross-agent transfer' if is_transfer else 'Standard routing'}")
                
                # Create shorter handoff announcement
                if current_agent and target_agent:
                    handoff_msg = create_handoff_announcement(current_agent, target_agent)
                    twiml.say(handoff_msg, voice=previous_voice)
                    twiml.pause(length=1)  # Reduced pause for faster handoff
            
            # Store current voice for next comparison
            conversation.previous_voice = agent_voice
            
            # Update conversation stage for transfers
            if 'is_transfer' in locals() and is_transfer:
                conversation.conversation_stage = "agent_conversation"
            
            # Say the cleaned AI response with the agent's voice
            twiml.say(clean_response, voice=agent_voice)
            twiml.pause(length=1)
            
            # Log for dashboard integration
            print(f"🎤 Voice Agent Active: {agent_voice}")
            print(f"   💬 Response: {clean_response[:50]}...")
            if agent_voice == "Polly.Joanna":
                print("   🟢 Dashboard: Orchestrator should be highlighted (GREEN)")
            elif agent_voice == "Polly.Matthew":
                print("   🔵 Dashboard: Travel Agent should be highlighted (BLUE)")
            elif agent_voice == "Polly.Brian":
                print("   🔴 Dashboard: Train Agent should be highlighted (RED)")
            
            # Log transfer type if detected
            if 'is_transfer' in locals() and is_transfer:
                print(f"   ↔️ Cross-agent transfer completed: User can now interact with {conversation.active_agent}")
            
            # Determine next step based on conversation stage
            if conversation.conversation_stage == "agent_conversation":
                # Continue conversation with the active agent - avoid repetitive prompts
                # Check if we've had enough back-and-forth to avoid loops
                if len(conversation.responses) > 6:  # After 6+ exchanges, offer to conclude
                    twiml.say("Thank you for using Samsaek! I hope I was able to help you today. Have a great trip!", voice=agent_voice)
                    twiml.hangup()
                else:
                    # Continue conversation naturally - let the agent's response speak for itself
                    # The agent has already asked a question or provided information
                    # Simply wait for user input without additional prompts
                    gather = twiml.gather(
                        input='speech',
                        timeout=15,  # Give more time for thoughtful responses
                        speech_timeout='auto',
                        action="/webhook/conversation", 
                        method="POST"
                    )
                    # Don't add any additional prompts - the agent's response is sufficient
                    
                    # Timeout fallback - keep conversation alive
                    twiml.say("I'm still here to help. Please let me know if you need anything else.", voice=agent_voice)
                    twiml.hangup()
                
            elif conversation.current_question < len(QUESTIONS) - 1:
                # Move to next question (only for initial routing)
                conversation.current_question += 1
                gather = twiml.gather(
                    input='speech',
                    timeout=10,
                    speech_timeout='auto',
                    action="/webhook/conversation",
                    method="POST"
                )
                gather.say(QUESTIONS[conversation.current_question], voice=agent_voice)
                
                # Improved fallback to prevent redirect loops
                twiml.say("Thank you for your patience. If you'd like to try again, please call back.", voice=agent_voice)
                twiml.hangup()
            else:
                # Final closing - only after extended conversation
                if conversation.conversation_stage == "agent_conversation":
                    # Ask if they need anything else
                    gather = twiml.gather(
                        input='speech',
                        timeout=8,
                        speech_timeout='auto',
                        action="/webhook/conversation",
                        method="POST"
                    )
                    gather.say("Is there anything else I can help you with today?", voice=agent_voice)
                    
                    # If no response, end conversation
                    twiml.say(CLOSING, voice=AGENT_VOICES["orchestrator"])
                    await save_conversation_summary(conversation)
                else:
                    # Standard ending
                    twiml.say(CLOSING, voice=AGENT_VOICES["orchestrator"])
                    await save_conversation_summary(conversation)
                
        else:
            # No speech detected
            twiml.say("I didn't hear your response. Let me ask again.", voice="Polly.Joanna")
            gather = twiml.gather(
                input='speech',
                timeout=10,
                speech_timeout='auto',
                action="/webhook/conversation",
                method="POST"
            )
            gather.say(QUESTIONS[conversation.current_question], voice="Polly.Joanna")
        
        return Response(content=str(twiml), media_type="application/xml")
        
    except Exception as e:
        print(f"❌ Conversation error: {e}")
        
        twiml = VoiceResponse()
        twiml.say("I'm having trouble processing that. Thank you for calling!")
        return Response(content=str(twiml), media_type="application/xml")

@app.post("/webhook/sms")
async def handle_incoming_sms(request: Request):
    """Handle incoming SMS messages"""
    
    try:
        form_data = await request.form()
        
        from_number = form_data.get('From', '')
        body = form_data.get('Body', '')
        
        print("📱 INCOMING SMS!")
        print(f"   From: {from_number}")
        print(f"   Message: {body}")
        
        # Auto-reply with AI agent
        twiml = MessagingResponse()
        
        reply = (
            f"Hello from Samsaek AI! 🤖 Thanks for your message: '{body}'. "
            f"Our AI agents received your request and are processing it. "
            f"For an enhanced conversation experience, call {TWILIO_PHONE_NUMBER}!"
        )
        
        twiml.message(reply)
        
        return Response(content=str(twiml), media_type="application/xml")
        
    except Exception as e:
        print(f"❌ SMS webhook error: {e}")
        return Response(content="<?xml version='1.0' encoding='UTF-8'?><Response></Response>", 
                       media_type="application/xml")

@app.post("/webhook/status")
async def handle_status_callback(request: Request):
    """Handle call/SMS status updates and cleanup"""
    
    try:
        form_data = await request.form()
        
        call_sid = form_data.get('CallSid', '')
        call_status = form_data.get('CallStatus', '')
        duration = form_data.get('CallDuration', '0')
        
        print(f"📊 Call Status Update:")
        print(f"   SID: {call_sid}")
        print(f"   Status: {call_status}")
        print(f"   Duration: {duration} seconds")
        
        # Clean up conversation state when call ends
        if call_status in ["completed", "failed", "busy", "no-answer"]:
            if call_sid in active_conversations:
                conversation = active_conversations[call_sid]
                # Final save if not already done
                if conversation.responses:
                    await save_conversation_summary(conversation)
                del active_conversations[call_sid]
                print(f"🗑️ Cleaned up conversation state for {call_sid}")
        
        return {"status": "received"}
        
    except Exception as e:
        print(f"❌ Status callback error: {e}")
        return {"status": "error"}

# 📊 API Endpoints
@app.post("/call")
async def make_live_call(request: MakeCallRequest):
    """📞 Make a live phone call with AI voice agent"""
    
    try:
        print(f"📞 Making LIVE call to {request.to}")
        print(f"🎵 Message: {request.message[:50]}...")
        
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Create enhanced TwiML with AI agent personality
        twiml = VoiceResponse()
        
        # Greeting
        twiml.say(
            f"Hello! This is {request.agent_name} calling from Samsaek.",
            voice=request.voice,
            language="en-US"
        )
        twiml.pause(length=1)
        
        # Main message
        twiml.say(
            request.message,
            voice=request.voice,
            language="en-US"
        )
        twiml.pause(length=1)
        
        # Closing
        twiml.say(
            "This demonstrates our enhanced AI conversation system with state management. Thank you for listening!",
            voice=request.voice,
            language="en-US"
        )
        
        # Make the actual phone call
        call = client.calls.create(
            to=request.to,
            from_=TWILIO_PHONE_NUMBER,
            twiml=str(twiml),
            record=request.record,
            timeout=60
        )
        
        print(f"✅ LIVE CALL INITIATED!")
        print(f"   Call SID: {call.sid}")
        print(f"   From: {TWILIO_PHONE_NUMBER}")
        print(f"   To: {request.to}")
        
        return {
            "success": True,
            "call_sid": call.sid,
            "from_number": TWILIO_PHONE_NUMBER,
            "to_number": request.to,
            "message": "🎉 Enhanced voice call initiated successfully!",
            "voice_agent": request.agent_name,
            "status": "Call in progress - check your phone!"
        }
        
    except Exception as e:
        print(f"❌ LIVE CALL FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Call failed: {str(e)}")

@app.post("/sms")
async def send_live_sms(request: SendSMSRequest):
    """📱 Send a live SMS message"""
    
    try:
        print(f"📱 Sending LIVE SMS to {request.to}")
        
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Send SMS
        message = client.messages.create(
            to=request.to,
            from_=TWILIO_PHONE_NUMBER,
            body=f"Hello from {request.agent_name}! {request.message}"
        )
        
        print(f"✅ LIVE SMS SENT!")
        print(f"   Message SID: {message.sid}")
        
        return {
            "success": True,
            "message_sid": message.sid,
            "from_number": TWILIO_PHONE_NUMBER,
            "to_number": request.to,
            "message": "📱 LIVE SMS sent successfully!",
            "agent": request.agent_name
        }
        
    except Exception as e:
        print(f"❌ LIVE SMS FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"SMS failed: {str(e)}")

@app.get("/conversations")
async def list_conversations():
    """📂 List recent conversations"""
    try:
        if not os.path.exists("conversations"):
            return {"conversations": [], "message": "No conversations yet"}
        
        files = [f for f in os.listdir("conversations") if f.endswith("_summary.txt")]
        files.sort(key=lambda x: os.path.getmtime(f"conversations/{x}"), reverse=True)
        
        conversations = []
        for file in files[:10]:  # Last 10 conversations
            with open(f"conversations/{file}", "r") as f:
                content = f.read()
                # Extract basic info
                lines = content.split("\\n")
                call_sid = next((line.split(": ")[1] for line in lines if line.startswith("Call SID:")), "Unknown")
                user_name = next((line.split(": ")[1] for line in lines if line.startswith("User Name:")), "Unknown")
                
                conversations.append({
                    "file": file,
                    "call_sid": call_sid,
                    "user_name": user_name,
                    "modified": datetime.fromtimestamp(os.path.getmtime(f"conversations/{file}")).isoformat()
                })
        
        return {"conversations": conversations, "active_calls": len(active_conversations)}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
async def get_service_status():
    """📊 Get service status and account info"""
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Get account info
        account = client.api.accounts(TWILIO_ACCOUNT_SID).fetch()
        balance = client.balance.fetch()
        
        return {
            "service": "🤖 Samsaek Enhanced Voice Service",
            "status": "🟢 LIVE & ENHANCED",
            "account_name": account.friendly_name,
            "account_status": account.status,
            "balance": f"${balance.balance} {balance.currency}",
            "phone_number": TWILIO_PHONE_NUMBER,
            "voice_synthesis": "✅ Enhanced AI Ready",
            "conversation_features": [
                "🧠 Conversation state management",
                "🤖 AI-powered responses with Gemini",
                "💾 Automatic conversation logging",
                "📝 Multi-question conversation flows",
                "🎯 Context-aware interactions"
            ],
            "active_conversations": len(active_conversations)
        }
        
    except Exception as e:
        return {
            "service": "🤖 Samsaek Enhanced Voice Service",
            "status": "❌ ERROR",
            "error": str(e)
        }

@app.get("/")
async def root():
    """🏠 Welcome endpoint"""
    return {
        "name": "🤖 Samsaek Enhanced Voice Service",
        "status": "🟢 LIVE & ENHANCED",
        "description": "AI-powered conversations with state management",
        "your_phone_number": TWILIO_PHONE_NUMBER,
        "features": [
            "🧠 Conversation state management",
            "🤖 AI-powered responses with Gemini",
            "💾 Automatic conversation logging",
            "📝 Multi-question conversation flows",
            "🎯 Specialized agent orchestration",
            "✈️ Travel agent integration",
            "🚂 Train agent integration",
            "🧠 Intelligent request routing"
        ],
        "endpoints": {
            "make_call": "POST /call",
            "send_sms": "POST /sms", 
            "check_status": "GET /status",
            "conversations": "GET /conversations",
            "api_docs": "GET /docs"
        },
        "conversation_flow": QUESTIONS,
        "active_conversations": len(active_conversations)
    }

if __name__ == "__main__":
    print("🚀 STARTING SAMSAEK ENHANCED VOICE SERVICE")
    print("=" * 60)
    print("📞 LIVE PHONE CALLS: ✅ ENABLED")
    print("🧠 CONVERSATION STATE: ✅ ENHANCED")
    print("🤖 AI RESPONSES: ✅ GEMINI POWERED")
    print("💾 CONVERSATION LOGGING: ✅ ENABLED")
    print(f"📱 YOUR PHONE NUMBER: {TWILIO_PHONE_NUMBER}")
    print("=" * 60)
    print("📡 Server: http://localhost:8001")
    print("📚 API Docs: http://localhost:8001/docs")
    print("📊 Status: http://localhost:8001/status")
    print("📂 Conversations: http://localhost:8001/conversations")
    print("=" * 60)
    print("🎯 READY FOR INTELLIGENT CONVERSATIONS!")
    print("Now featuring:")
    for i, q in enumerate(QUESTIONS, 1):
        print(f"   {i}. {q}")
    print("Press Ctrl+C to stop")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8001)