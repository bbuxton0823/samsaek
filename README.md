# 🤖 Samsaek Multi-Agent AI System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![Twilio](https://img.shields.io/badge/Twilio-Voice%20%26%20SMS-red.svg)](https://www.twilio.com/)

> **Advanced Multi-Agent Travel Assistant with Real-time Voice Integration**

Samsaek is a sophisticated multi-agent AI system featuring enhanced travel capabilities with real-time transit data, live voice interaction, and comprehensive travel planning services. The system integrates powerful APIs to provide intelligent travel assistance through phone calls, web interface, and direct API access.

## 🌟 Key Features

### 🎯 **Enhanced Travel Agent Integration**
- **Real-time Transit Data**: Live routes, schedules, and departures via TransitLand API
- **Traffic Monitoring**: Current traffic incidents and road conditions via 511 API
- **Intelligent Search**: Advanced travel information retrieval using Exa AI
- **Geolocation Services**: Location-based recommendations via Google Maps
- **Multi-modal Transportation**: Flight, train, bus, and local transit support

### 📞 **Live Voice Interaction**
- **Phone Call Support**: Direct phone integration via Twilio
- **Natural Voice Synthesis**: High-quality speech generation via Eleven Labs
- **Real-time Conversations**: Immediate response processing
- **Agent Handoffs**: Seamless transfer between travel and train specialists

### 🧠 **Multi-Agent Architecture**
- **Travel Agent (Matthew)**: Flight bookings, hotels, vacation planning
- **Train Agent (Brian)**: Rail travel, local transportation, route planning
- **Orchestrator**: Intelligent request routing and agent coordination
- **Cross-Agent Transfers**: Automatic handoffs based on request type

### 🔐 **Production-Ready Security**
- **API Key Management**: Secure configuration handling
- **Safety Filtering**: Content validation and dangerous request blocking
- **Error Handling**: Comprehensive fallback mechanisms
- **Rate Limiting**: API usage protection

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Valid API keys for required services
- Twilio account for phone integration
- ngrok for webhook tunneling (development)

### 1. Clone and Setup
```bash
git clone https://github.com/bbuxton0823/samsaek.git
cd samsaek
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file with your API keys:
```bash
# Core API Keys
GOOGLE_MAPS_API_KEY=your_google_maps_key
TRANSITLAND_API_KEY=your_transitland_key
511APIKEY=your_511_traffic_key
EXA_SEARCH_API_KEY=your_exa_search_key

# Voice & Phone Services
ELEVENLABS_API_KEY=your_elevenlabs_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=your_twilio_number

# AI Services
GEMINI_API_KEY=your_gemini_key
WEAVE_API_KEY=your_weave_key (optional)
```

### 3. Start the Enhanced Voice Server
```bash
python enhanced_voice_server.py
```

The server will start on `http://localhost:8001` with the following endpoints:
- **📞 Phone Calls**: Your Twilio number (e.g., +15716267576)
- **🌐 Web Interface**: http://localhost:8001
- **📚 API Documentation**: http://localhost:8001/docs
- **📊 Status Check**: http://localhost:8001/status

## 📱 Using the System

### Phone Integration
1. **Call the Twilio Number**: Dial your configured Twilio phone number
2. **Introduce Yourself**: "Hi, my name is [Your Name]"
3. **Request Travel Help**: Ask about flights, hotels, transit, or traffic
4. **Agent Handoffs**: The system automatically transfers between specialists

### Example Conversations
```
User: "Hi, my name is John. I need help with travel."
System: "Hello John! I'm Matthew, your travel specialist..."

User: "I need a flight to San Francisco"
Matthew: "I can help you find flights to San Francisco. Current market analysis shows..."

User: "Actually, can you help me find train routes instead?"
Matthew: "Let me transfer you to Brian, our train specialist..."
Brian: "Hi John! I can help you with train routes to San Francisco..."
```

### Web API Usage
```bash
# Test the status endpoint
curl http://localhost:8001/status

# Make a travel request
curl -X POST "http://localhost:8001/voice-webhook" \
  -H "Content-Type: application/json" \
  -d '{"Body": "I need a flight to Paris", "From": "+1234567890"}'
```

## 🏗️ Architecture Overview

### Core Components

#### Enhanced Travel Agent (`enhanced_travel_agent.py`)
```python
class EnhancedTravelAgent:
    - search_transit_routes()     # TransitLand API integration
    - get_nearby_transit()        # Location-based transit search
    - get_transit_departures()    # Real-time departure data
    - get_traffic_data()          # Live traffic incidents
    - search_travel_info()        # Exa AI search integration
    - handle_travel_request()     # Main request processor
```

#### Specialized Agent System (`simplified_specialized_agents.py`)
```python
class SimplifiedSpecializedAgentSystem:
    - classify_request()          # Intelligent request routing
    - detect_cross_agent_transfer() # Agent handoff detection
    - apply_safety_filter()       # Content validation
    - handle_travel_request()     # Travel agent delegation
    - handle_train_request()      # Train agent delegation
```

#### Enhanced Voice Server (`enhanced_voice_server.py`)
```python
class EnhancedVoiceServer:
    - /voice-webhook             # Twilio webhook endpoint
    - /status                    # Health check endpoint
    - /conversations             # Conversation history
    - /docs                      # API documentation
```

### API Integrations

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **TransitLand** | Real-time transit data | `TRANSITLAND_API_KEY` |
| **511 Traffic** | Live traffic incidents | `511APIKEY` |
| **Exa AI** | Advanced search | `EXA_SEARCH_API_KEY` |
| **Google Maps** | Geolocation services | `GOOGLE_MAPS_API_KEY` |
| **Eleven Labs** | Voice synthesis | `ELEVENLABS_API_KEY` |
| **Twilio** | Phone integration | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` |
| **Gemini** | AI responses | `GEMINI_API_KEY` |

## 🔧 Configuration

### Environment Variables
The system supports comprehensive configuration through environment variables:

```python
# Core Settings
DEBUG=true
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8001

# Database (optional)
DATABASE_URL=sqlite:///./samsaek.db

# Voice Settings
VOICE_MODEL=eleven_monolingual_v1
VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Rachel voice
VOICE_STABILITY=0.75
VOICE_CLARITY=0.85

# Agent Settings
AGENT_TIMEOUT=30
MAX_CONVERSATION_LENGTH=10
ENABLE_AGENT_TRANSFERS=true
```

### Twilio Webhook Configuration
Set your Twilio webhook URL to:
```
https://your-domain.com/voice-webhook
```

For development with ngrok:
```bash
ngrok http 8001
# Then use: https://your-ngrok-url.ngrok.io/voice-webhook
```

## 🧪 Testing

### Run System Tests
```bash
# Test API integrations
python test_api_integration.py

# Test voice synthesis
python test_voice_synthesis.py

# Test enhanced travel agent
python -c "from enhanced_travel_agent import test_enhanced_travel_agent; import asyncio; asyncio.run(test_enhanced_travel_agent())"

# Test specialized agents
python -c "from simplified_specialized_agents import test_simplified_agents; import asyncio; asyncio.run(test_simplified_agents())"
```

### Test Phone Integration
1. Start the server: `python enhanced_voice_server.py`
2. Expose with ngrok: `ngrok http 8001`
3. Configure Twilio webhook URL
4. Call your Twilio number and test conversations

### Example Test Scenarios
- **Flight Booking**: "I need a flight to Tokyo next month"
- **Transit Routes**: "Show me train routes from New York to Boston"
- **Traffic Updates**: "What's the traffic like on Highway 101?"
- **Hotel Search**: "Find me hotels in Paris for next week"
- **Agent Transfer**: "Can you help me with train schedules?" (while talking to travel agent)

## 📊 Monitoring & Logging

### Real-time Status
Access system status at: `http://localhost:8001/status`

### Conversation Logs
View conversation history: `http://localhost:8001/conversations`

### Weave Integration (Optional)
Enable advanced monitoring with Weave:
```bash
export WEAVE_API_KEY=your_weave_key
```

## 🔒 Security Features

### Content Filtering
The system includes comprehensive safety filtering:
- **Dangerous Keywords**: Blocks harmful content requests
- **Topic Validation**: Ensures travel-focused conversations
- **Input Sanitization**: Prevents injection attacks
- **Rate Limiting**: Protects against abuse

### API Security
- **Environment Variables**: Secure key storage
- **Request Validation**: Input validation on all endpoints
- **Error Handling**: Graceful failure without exposure
- **CORS Configuration**: Controlled cross-origin access

## 🚀 Deployment

### Production Deployment
1. **Set Production Environment**:
   ```bash
   export NODE_ENV=production
   export DEBUG=false
   ```

2. **Configure Production Webhook**:
   Update Twilio webhook to your production URL

3. **Scale with Docker** (Optional):
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["python", "enhanced_voice_server.py"]
   ```

4. **Monitor with Health Checks**:
   Regular checks on `/status` endpoint

### Cloud Deployment Options
- **Heroku**: Easy deployment with Procfile
- **AWS Lambda**: Serverless voice processing
- **Google Cloud Run**: Container-based scaling
- **DigitalOcean Apps**: Simple container deployment

## 🐛 Troubleshooting

### Common Issues

#### Voice Server Won't Start
```bash
# Check if port is in use
lsof -ti:8001
# Kill existing process
lsof -ti:8001 | xargs kill -9
# Restart server
python enhanced_voice_server.py
```

#### API Key Issues
```bash
# Verify environment variables
echo $TRANSITLAND_API_KEY
echo $ELEVENLABS_API_KEY
# Test API connectivity
python test_api_integration.py
```

#### Twilio Webhook Issues
1. Check ngrok is running and accessible
2. Verify webhook URL in Twilio console
3. Test with Twilio webhook debugger
4. Check server logs for incoming requests

#### Agent Not Responding
- Verify Gemini API key is set and valid
- Check conversation logs for errors
- Ensure safety filters aren't blocking requests
- Test with simple travel queries

### Debug Mode
Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
python enhanced_voice_server.py
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Run tests before committing
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Include docstrings for public methods
- Add tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SamSaekTransit**: Original transit integration framework
- **TransitLand**: Real-time transit data API
- **511.org**: Traffic incident data
- **Exa AI**: Advanced search capabilities
- **Eleven Labs**: Natural voice synthesis
- **Twilio**: Voice and SMS communication
- **Google**: Maps and Gemini AI services

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/bbuxton0823/samsaek/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bbuxton0823/samsaek/discussions)
- **Email**: Contact repository maintainers

---

**🎯 Ready for Production**: Your enhanced travel assistant is live and ready to help users with comprehensive travel planning through natural voice interactions!

---

*Last Updated: January 13, 2025*
*Generated with Claude Code*