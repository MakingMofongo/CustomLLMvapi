# Custom LLM Integration with Vapi

This project provides a high-performance proxy server that enables seamless integration between Vapi and OpenAI's LLM models. The server is built with Quart (async Flask) and optimized for low latency, efficient streaming, and reliability, especially for long responses.

## Overview

- Acts as a middleware between Vapi and OpenAI API
- Handles both streaming and non-streaming completions
- Features a Streamlit dashboard for monitoring requests and performance
- Optimized for minimal time-to-first-token and robust streaming
- Provides detailed latency metrics and error handling

## Performance Optimizations

### Connection Optimizations
- **HTTP/2 Support**: Enabled for better connection multiplexing
- **Connection Pooling**: Maintains persistent connections to reduce setup time
- **Connection Warm-up**: Pre-establishes connections to avoid cold start delays
- **Automatic Retries**: Handles transient connection failures automatically
- **Dynamic Timeouts**: Scales timeout durations based on expected response length

### Streaming Optimizations
- **Fast Path for Streaming**: Prioritizes streaming setup for quicker first token delivery
- **Parallel Processing**: Preprocesses messages while establishing connections
- **Chunk-Based Processing**: Uses efficient byte-level streaming with proper buffer management
- **Partial UTF-8 Handling**: Gracefully handles partial characters at chunk boundaries
- **Windows-Specific Fixes**: Special handling for Windows connection errors with large responses

### Memory and Cache Optimizations
- **Response Caching**: Implements TTL caching for identical non-streaming requests
- **Message Preprocessing**: Reduces payload size by stripping unnecessary content
- **Optimized Headers**: Minimizes header size for faster transmission
- **Dynamic Buffer Sizing**: Adjusts buffer sizes based on expected response length

## Monitoring and Metrics

### Streamlit Dashboard
- Real-time request monitoring
- Response time visualization
- Token rate and performance metrics
- Error tracking and diagnostics

### Latency Metrics
- Time to first byte/token
- Total request processing time
- Tokens per second
- Connection establishment time
- Server processing overhead

## Features

- endpoint (`/chat/completions`) for regular chat completions AS WELL AS streaming completions
- Proper formatting of OpenAI responses for Vapi compatibility
- Environment variable configuration
- Detailed error handling and reporting
- SQLite database for request tracking

## Prerequisites

- Python 3.7 or higher
- OpenAI API key
- Vapi account
- Ngrok (for local testing)

## Setup Instructions

### 1. Clone the repository

```bash
git clone [repository-url]
cd custom-llm-vapi
```

### 2. Set up the environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
# Create .env file from example
cp .env.example .env

# Edit .env file with your OpenAI API key
# OPENAI_API_KEY=your_openai_api_key_here
# Optional settings:
# PORT=5000
# OPENAI_API_URL=https://api.openai.com/v1/chat/completions
# ENABLE_LATENCY_METRICS=true
# ENABLE_LATENCY_HEADERS=true
# WORKERS=4
```

### 4. Run the server

```bash
python app.py
```

The server will start on port 5000 by default. The Streamlit dashboard will be available at http://localhost:8501.

### 5. Expose the server with Ngrok

In a new terminal window:

```bash
ngrok http 5000
```

Ngrok will provide a public URL (e.g., https://your-unique-id.ngrok.io) that you'll use in the Vapi dashboard.

## Configuring Vapi

1. Log in to your Vapi account and navigate to the "Model" section
2. Select "Custom LLM" option
3. Enter your Ngrok URL (e.g., https://your-unique-id.ngrok.io/chat/completions) into the endpoint field
4. For streaming, use the streaming endpoint (e.g., https://your-unique-id.ngrok.io/chat/completions/stream)
5. Test the connection by sending a message through the Vapi interface

## API Endpoints

### POST /chat/completions

Non-streaming endpoint for chat completions.

**Request Body:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "model": "gpt-3.5-turbo",
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### POST /chat/completions/stream

Streaming endpoint for real-time chat completions.

**Request Body:** Same as non-streaming endpoint.

### GET /metrics/latency

Returns latency metrics when enabled.

## Deployment

For production, deploy the Quart application to a cloud provider of your choice (AWS, GCP, Azure, Heroku, etc.). Make sure to set the environment variables accordingly.

## Handling Long Responses

The server is optimized to handle long responses with:
- Dynamic buffer sizing for efficient memory use
- Proper error recovery for connection interruptions
- Partial response delivery when possible
- Chunked transfer encoding

## Troubleshooting

- **Connection issues**: Ensure your Ngrok tunnel is running and the URL in Vapi dashboard is correct
- **OpenAI errors**: Verify your API key and check OpenAI's status page for any outages
- **Format errors**: Ensure the response format matches what Vapi expects
- **Windows errors**: If encountering connection resets on Windows, try adjusting the `max_tokens` value
- **Dashboard errors**: If the Streamlit dashboard fails to load, check port 8501 is available 