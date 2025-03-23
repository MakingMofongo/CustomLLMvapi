# Custom LLM Integration with Vapi

This project allows you to connect Vapi to OpenAI's gpt-3.5-turbo-instruct model using a custom LLM configuration. It creates a Flask server that acts as a proxy between Vapi and OpenAI.

## Features

- Non-streaming endpoint (`/chat/completions`) for regular chat completions
- Streaming endpoint (`/chat/completions/stream`) for real-time responses
- Proper formatting of OpenAI responses for Vapi compatibility
- Environment variable configuration

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
```

### 4. Run the server

```bash
python app.py
```

The server will start on port 5000 by default (can be changed in the .env file).

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
  "model": "gpt-3.5-turbo-instruct",
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### POST /chat/completions/stream

Streaming endpoint for real-time chat completions.

**Request Body:** Same as non-streaming endpoint.

## Deployment

For production, deploy the Flask application to a cloud provider of your choice (AWS, GCP, Azure, Heroku, etc.). Make sure to set the environment variables accordingly.

## Troubleshooting

- **Connection issues**: Ensure your Ngrok tunnel is running and the URL in Vapi dashboard is correct
- **OpenAI errors**: Verify your API key and check OpenAI's status page for any outages
- **Format errors**: Ensure the response format matches what Vapi expects 