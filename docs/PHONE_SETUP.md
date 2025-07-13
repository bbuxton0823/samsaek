# Samsaek Phone Service Setup Guide

This guide explains how to set up and use the Twilio phone service integration in Samsaek.

## Prerequisites

1. **Twilio Account**: Sign up at [https://www.twilio.com/](https://www.twilio.com/)
2. **Account SID and Auth Token**: Available in your Twilio Console
3. **Phone Number**: Purchase or configure a Twilio phone number

## Configuration

### 1. Environment Variables

Add the following to your `.env.production` file:

```bash
# Twilio Configuration
TWILIO_ACCOUNT_SID=your-twilio-account-sid-here
TWILIO_AUTH_TOKEN=your-twilio-auth-token-here
SAMSAEK_TWILIO_PHONE_NUMBER=+1234567890
SAMSAEK_TWILIO_WEBHOOK_URL=https://yourdomain.com/api/phone/webhook
```

### 2. Webhook Configuration

Set up webhooks in your Twilio Console:

1. Go to **Phone Numbers > Manage > Active numbers**
2. Click on your phone number
3. Configure webhooks:
   - **Voice**: `https://yourdomain.com/api/phone/webhook/call`
   - **SMS**: `https://yourdomain.com/api/phone/webhook/sms`
   - **Status Callback**: `https://yourdomain.com/api/phone/webhook/status`

## Phone Number Provisioning

### Search for Available Numbers

Use the search endpoint to find available phone numbers:

```bash
curl -X GET "http://localhost:8000/api/phone/numbers/search?area_code=415&country=US" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Purchase a Phone Number

Once you find a suitable number, purchase it:

```bash
curl -X POST "http://localhost:8000/api/phone/numbers/purchase" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "phone_number": "+14155550123",
    "friendly_name": "Samsaek Main Line",
    "webhook_url": "https://yourdomain.com/api/phone/webhook"
  }'
```

## Making Phone Calls

### Basic Call

```bash
curl -X POST "http://localhost:8000/api/phone/call" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "to": "+1987654321",
    "message": "Hello! This is Samsaek AI Assistant calling.",
    "agent_name": "Samsaek Assistant",
    "record": true
  }'
```

### Advanced Call with Custom TwiML

```bash
curl -X POST "http://localhost:8000/api/phone/call" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "to": "+1987654321",
    "message": "Welcome to Samsaek! Please hold for an agent.",
    "voice": "alice",
    "language": "en",
    "call_purpose": "customer_support",
    "workflow_id": "support_001"
  }'
```

## Sending SMS Messages

```bash
curl -X POST "http://localhost:8000/api/phone/sms" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "to": "+1987654321",
    "message": "Hello! This is a message from Samsaek AI Assistant.",
    "agent_name": "Samsaek Bot"
  }'
```

## Monitoring and Management

### Check Account Balance

```bash
curl -X GET "http://localhost:8000/api/phone/balance" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### View Call History

```bash
curl -X GET "http://localhost:8000/api/phone/calls/history?limit=10" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Get Call Status

```bash
curl -X GET "http://localhost:8000/api/phone/calls/CA1234567890abcdef" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Hangup Active Call

```bash
curl -X POST "http://localhost:8000/api/phone/calls/CA1234567890abcdef/hangup" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Phone Service Health Check

```bash
curl -X GET "http://localhost:8000/api/phone/health" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Security and Permissions

The phone service uses role-based permissions:

- `phone:view` - View phone numbers, calls, and balance
- `phone:call` - Make phone calls and hangup calls
- `phone:sms` - Send SMS messages
- `phone:admin` - Purchase phone numbers and manage configuration

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify your Twilio Account SID and Auth Token
   - Check that credentials are properly set in environment variables

2. **Invalid Phone Numbers**
   - Ensure phone numbers are in E.164 format (e.g., +1234567890)
   - Verify the destination number can receive calls/SMS

3. **Insufficient Funds**
   - Check your Twilio account balance
   - Add funds to your Twilio account

4. **Webhook Issues**
   - Ensure your webhook URLs are publicly accessible
   - Check that webhook endpoints are properly configured

### Error Codes

- `400` - Invalid request (check phone number format)
- `401` - Authentication required
- `403` - Insufficient permissions
- `404` - Call/SMS not found
- `500` - Service error (check logs)

## Cost Management

### Pricing Information

- **Voice Calls**: ~$0.0085/minute (US)
- **SMS Outbound**: ~$0.0075/message (US)
- **SMS Inbound**: ~$0.0075/message (US)
- **Phone Numbers**: ~$1.00/month (US local)

### Usage Monitoring

Monitor usage through the API:

```bash
curl -X GET "http://localhost:8000/api/phone/stats" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Integration with Voice Service

Combine phone calls with voice synthesis:

1. Generate speech using Eleven Labs voice service
2. Host the audio file publicly (S3, CDN, etc.)
3. Use the audio URL in TwiML for phone calls

Example TwiML with custom audio:

```xml
<Response>
    <Play>https://yourdomain.com/audio/welcome.mp3</Play>
    <Pause length="2"/>
    <Say voice="alice">How can I help you today?</Say>
</Response>
```

## Support

For additional support:
- Twilio Documentation: [https://www.twilio.com/docs](https://www.twilio.com/docs)
- Samsaek GitHub Issues: [Project Repository Issues]
- Twilio Support: [https://support.twilio.com/](https://support.twilio.com/)