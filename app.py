from quart import Quart, request, Response, jsonify, make_response
import os
import json
import openai
from dotenv import load_dotenv
import traceback
import asyncio
import httpx
import time
import gzip
import functools
import cachetools.func
import logging
import sqlite3
import subprocess
import sys
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vapi-latency")

# Load environment variables
load_dotenv()

# Initialize Quart app (async version of Flask)
app = Quart(__name__)

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get closest OpenAI API endpoint (can be overridden in .env)
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

# Enable latency logging
ENABLE_LATENCY_METRICS = os.getenv("ENABLE_LATENCY_METRICS", "true").lower() == "true"
ENABLE_LATENCY_HEADERS = os.getenv("ENABLE_LATENCY_HEADERS", "true").lower() == "true"

# Create a global httpx client for connection pooling with optimized settings
client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=3.0, read=60.0, write=60.0, pool=5.0),  # Separate timeouts for connection phases
    limits=httpx.Limits(max_keepalive_connections=50, max_connections=100, keepalive_expiry=30.0),
    http2=True,  # Enable HTTP/2 for better performance if supported
    transport=httpx.AsyncHTTPTransport(retries=1)  # Add automatic retry for transient failures
)

# Simple cache for repeated identical requests
@cachetools.func.ttl_cache(maxsize=100, ttl=60)  # Cache for 60 seconds
async def cached_completion(model, messages_hash, temperature, max_tokens):
    """Cache identical non-streaming requests"""
    # Implementation is just a placeholder for the cache decorator
    pass

# Message preprocessing to reduce payload size
def preprocess_messages(messages):
    """Strip unnecessary whitespace and optimize message format"""
    processed = []
    for msg in messages:
        if isinstance(msg, dict):
            # Keep only essential fields
            processed_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", "").strip()
            }
            processed.append(processed_msg)
    return processed

# Initialize database for dashboard
def init_db():
    try:
        conn = sqlite3.connect("requests.db")
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            input TEXT,
            output TEXT,
            metrics TEXT,
            completed BOOLEAN
        )
        ''')
        conn.commit()
        conn.close()
        logger.info("Dashboard database initialized")
    except Exception as e:
        logger.error(f"Error initializing dashboard database: {str(e)}")

# Function to store request data
async def store_request(request_id, metrics, messages=None, output=None, completed=False):
    try:
        # Extract input from messages if available - get the LAST user message
        input_text = ""
        if messages and isinstance(messages, list):
            # Reverse the messages list to find the most recent user message
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    input_text = msg.get("content", "")
                    break
        
        # Store in SQLite for dashboard
        conn = sqlite3.connect("requests.db")
        cursor = conn.cursor()
        
        # Check if request exists
        cursor.execute("SELECT id FROM requests WHERE id = ?", (request_id,))
        exists = cursor.fetchone()
        
        if exists:
            # Update existing record
            cursor.execute(
                "UPDATE requests SET metrics = ?, output = ?, completed = ? WHERE id = ?", 
                (json.dumps(metrics), output or "", completed, request_id)
            )
        else:
            # Create new record
            cursor.execute(
                "INSERT INTO requests (id, timestamp, input, output, metrics, completed) VALUES (?, ?, ?, ?, ?, ?)",
                (request_id, time.time(), input_text, output or "", json.dumps(metrics), completed)
            )
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error storing request data: {str(e)}")

# Function to launch Streamlit dashboard
def launch_dashboard():
    try:
        logger.info("Launching Streamlit dashboard...")
        # Use a more resilient approach to launching the dashboard
        dashboard_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "dashboard.py", "--server.port=8501", 
            "--server.headless=true",
            "--server.maxUploadSize=10",  # Reduce max upload size to prevent memory issues
            "--server.maxMessageSize=100"  # Limit WebSocket message size
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Register cleanup handler
        import atexit
        def cleanup_dashboard():
            try:
                if dashboard_process.poll() is None:
                    dashboard_process.terminate()
                    logger.info("Dashboard process terminated")
            except Exception as e:
                logger.error(f"Error terminating dashboard: {str(e)}")
        atexit.register(cleanup_dashboard)
        
        logger.info("Dashboard started with PID: %s", dashboard_process.pid)
    except Exception as e:
        logger.error(f"Error launching dashboard: {str(e)}")

# Connection pool warm-up with better error handling
async def warmup_connection_pool():
    """Establish initial connections to OpenAI API to avoid cold start delays"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Simple ping to warm up the connection pool
            base_url = OPENAI_API_URL.split('/v1/')[0]
            async with client.stream("GET", f"{base_url}/v1/models", 
                                headers={"Authorization": f"Bearer {openai.api_key}"}, 
                                timeout=httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=5.0)) as response:
                await response.aread()
                logger.info("Connection pool warmed up successfully")
                return
        except Exception as e:
            retry_count += 1
            logger.warning(f"Connection pool warmup attempt {retry_count} failed: {str(e)}")
            if retry_count < max_retries:
                await asyncio.sleep(1)  # Wait before retrying
    
    logger.warning("Connection pool warmup failed after maximum retries")

# Signal handler to gracefully shut down
def signal_handler():
    logger.info("Shutting down server...")
    # Close the httpx client
    asyncio.create_task(client.aclose())

@app.route("/chat/completions", methods=["POST"])
async def chat_completions():
    start_time = time.time()
    request_id = f"{int(start_time)}-{id(request)}"
    metrics = {
        "request_id": request_id,
        "total_tokens": 0,
        "request_start": start_time,
        "time_to_first_token": None,
        "time_to_last_token": None,
        "tokens_per_second": None,
        "total_request_time": None,
        "connection_time": None,
        "llm_processing_time": None,
        "server_processing_time": None,
        "from_cache": False
    }
    
    # Track active connections to handle timeouts better
    active_connections = getattr(app, 'active_connections', set())
    setattr(app, 'active_connections', active_connections)
    active_connections.add(request_id)
    
    try:
        # Parse data with minimal preprocessing initially
        data = await request.get_json()
        metrics["parse_time"] = time.time() - start_time
        
        # Extract core parameters needed for request
        model = data.get("model", "gpt-3.5-turbo")
        stream = data.get("stream", True)
        
        # Fast path for streaming - defer full preprocessing
        if stream:
            raw_messages = data.get("messages", [])
            temperature = data.get("temperature", 0.7)
            max_tokens = data.get("max_tokens", 1000)
            
            # Determine dynamic buffer size based on expected response size
            # If max_tokens is high, we expect a longer response
            buffer_size = 16384 if max_tokens > 500 else 4096
            
            # Start request processing in parallel with message preprocessing
            async def generate():
                nonlocal raw_messages
                
                # Optimize headers - only include what's necessary
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai.api_key}",
                    "Accept": "text/event-stream",
                }
                
                # Parallel message preprocessing
                preprocessing_start = time.time()
                messages = preprocess_messages(raw_messages)
                preprocessing_time = time.time() - preprocessing_start
                metrics["preprocessing_time"] = preprocessing_time
                
                # Optimize payload - only include necessary fields
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True
                }
                
                # Store initial request - do this in parallel
                preprocessing_task = asyncio.create_task(store_request(request_id, metrics, raw_messages))
                
                request_start = time.time()
                metrics["llm_request_start"] = request_start
                first_token_received = False
                token_count = 0
                
                full_output = ""
                
                try:
                    # Calculate dynamic timeouts based on expected response size
                    dynamic_read_timeout = max(60.0, min(300.0, max_tokens / 10))
                    
                    # Use shorter connection timeout for streaming to prioritize first token
                    connection_start = time.time()
                    
                    # Create a custom timeout for this specific request
                    custom_timeout = httpx.Timeout(
                        connect=2.0,
                        read=dynamic_read_timeout,
                        write=60.0,
                        pool=5.0
                    )
                    
                    async with client.stream(
                        "POST", 
                        OPENAI_API_URL, 
                        json=payload, 
                        headers=headers, 
                        timeout=custom_timeout
                    ) as response:
                        # Make sure preprocessing task completes 
                        if not preprocessing_task.done():
                            await preprocessing_task

                        connection_time = time.time() - connection_start
                        metrics["connection_time"] = connection_time
                        
                        first_byte_time = time.time()
                        metrics["time_to_first_byte"] = first_byte_time - request_start
                        
                        token_start_time = None
                        last_token_time = None
                        
                        # Add better error handling for stream reading
                        try:
                            # Use a more efficient line reader with improved buffer handling
                            buffer = ""
                            chunk_size = buffer_size
                            
                            async for raw_chunk in response.aiter_bytes(chunk_size=chunk_size):
                                if not raw_chunk:
                                    continue
                                    
                                # Decode chunks safely, handling partial UTF-8 characters
                                try:
                                    text_chunk = raw_chunk.decode('utf-8')
                                except UnicodeDecodeError:
                                    # Handle partial UTF-8 characters at chunk boundaries
                                    # by trying to decode with the buffer content
                                    buffer += raw_chunk.decode('utf-8', errors='ignore')
                                    continue
                                
                                # Process the chunk
                                buffer += text_chunk
                                lines = buffer.split('\n')
                                
                                # Process all complete lines
                                buffer = lines.pop()  # Keep the last incomplete line in the buffer
                                
                                for line in lines:
                                    line = line.strip()
                                    if not line:
                                        continue
                                        
                                    if line.startswith("data: "):
                                        if line.strip() == "data: [DONE]":
                                            # Calculate final metrics
                                            metrics["time_to_last_token"] = last_token_time - request_start if last_token_time else None
                                            total_streaming_time = last_token_time - token_start_time if token_start_time and last_token_time else 0
                                            if total_streaming_time > 0 and token_count > 0:
                                                metrics["tokens_per_second"] = token_count / total_streaming_time
                                            
                                            metrics["total_tokens"] = token_count
                                            metrics["total_request_time"] = time.time() - start_time
                                            
                                            # Log metrics if enabled
                                            if ENABLE_LATENCY_METRICS:
                                                logger.info(f"Streaming Latency Metrics: {json.dumps(metrics)}")
                                            
                                            # Store completed request data
                                            await store_request(request_id, metrics, raw_messages, full_output, True)
                                            
                                            # Remove from active connections
                                            active_connections.discard(request_id)
                                            
                                            yield "data: [DONE]\n\n"
                                            continue
                                        
                                        line = line[6:]  # Remove "data: " prefix
                                        try:
                                            chunk = json.loads(line)
                                            
                                            # Capture time of first token
                                            if not first_token_received:
                                                token_start_time = time.time()
                                                metrics["time_to_first_token"] = token_start_time - request_start
                                                first_token_received = True
                                            
                                            last_token_time = time.time()
                                            
                                            # Optimized chunk format - only include needed fields
                                            formatted_chunk = {
                                                "id": chunk.get("id", ""),
                                                "object": "chat.completion.chunk",
                                                "created": chunk.get("created", 0),
                                                "model": model,
                                                "choices": [
                                                    {
                                                        "index": choice.get("index", 0),
                                                        "delta": {
                                                            "content": choice.get("delta", {}).get("content", "")
                                                        },
                                                        "finish_reason": choice.get("finish_reason")
                                                    }
                                                    for choice in chunk.get("choices", [])
                                                    if choice.get("delta", {}).get("content", "")  # Skip empty deltas
                                                ]
                                            }
                                            
                                            # Count tokens (approximation by counting deltas)
                                            for choice in formatted_chunk["choices"]:
                                                content = choice["delta"]["content"]
                                                if content:
                                                    # Simple approximation: each chunk is roughly one token
                                                    token_count += 1
                                            
                                            # Only yield if there's actual content
                                            if any(choice["delta"]["content"] for choice in formatted_chunk["choices"]):
                                                # Add content to full output
                                                for choice in formatted_chunk["choices"]:
                                                    content = choice["delta"]["content"]
                                                    if content:
                                                        full_output += content
                                                
                                                # Update request data periodically - less frequently for long outputs
                                                update_interval = 20 if max_tokens > 500 else 5
                                                if token_count % update_interval == 0:
                                                    await store_request(request_id, metrics, raw_messages, full_output)
                                                
                                                # Yield the chunk with proper formatting
                                                yield f"data: {json.dumps(formatted_chunk)}\n\n"
                                        except json.JSONDecodeError:
                                            # Skip invalid JSON
                                            continue
                                            
                            # Process any remaining content in the buffer
                            if buffer and buffer.startswith("data: "):
                                try:
                                    line = buffer[6:]  # Remove "data: " prefix
                                    chunk = json.loads(line)
                                    
                                    # Process similar to the loop above
                                    # (Code would be similar to above, but for brevity not duplicating)
                                    pass
                                except Exception:
                                    pass
                                    
                        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                            logger.warning(f"Timeout during streaming: {str(e)}")
                            error_msg = {
                                "error": {
                                    "message": f"Streaming timeout: {str(e)}",
                                    "type": "timeout",
                                }
                            }
                            metrics["error"] = f"Timeout: {str(e)}"
                            if ENABLE_LATENCY_METRICS:
                                logger.error(f"Streaming Timeout: {json.dumps(metrics)}")
                            
                            # Store error in request data
                            await store_request(request_id, metrics, raw_messages, full_output, True)
                            
                            # Remove from active connections
                            active_connections.discard(request_id)
                            
                            yield f"data: {json.dumps(error_msg)}\n\n"
                            yield "data: [DONE]\n\n"
                            
                        except ConnectionError as e:
                            # Handle Windows-specific connection errors
                            logger.warning(f"Connection error during streaming: {str(e)}")
                            # If we already got content, we can return what we have
                            if full_output:
                                metrics["error"] = f"Partial content due to connection error: {str(e)}"
                                metrics["time_to_last_token"] = last_token_time - request_start if last_token_time else None
                                metrics["total_tokens"] = token_count
                                metrics["total_request_time"] = time.time() - start_time
                                
                                if ENABLE_LATENCY_METRICS:
                                    logger.info(f"Partial Streaming Metrics: {json.dumps(metrics)}")
                                
                                # Store partial request data as completed
                                await store_request(request_id, metrics, raw_messages, full_output, True)
                                
                                # Notify client of completion despite error
                                yield "data: [DONE]\n\n"
                            else:
                                error_msg = {
                                    "error": {
                                        "message": f"Connection error: {str(e)}",
                                        "type": "connection_error",
                                    }
                                }
                                metrics["error"] = f"Connection error: {str(e)}"
                                if ENABLE_LATENCY_METRICS:
                                    logger.error(f"Streaming Connection Error: {json.dumps(metrics)}")
                                
                                # Store error in request data
                                await store_request(request_id, metrics, raw_messages, full_output, True)
                                
                                yield f"data: {json.dumps(error_msg)}\n\n"
                                yield "data: [DONE]\n\n"
                            
                            # Remove from active connections
                            active_connections.discard(request_id)
                            
                        except Exception as e:
                            error_msg = {
                                "error": {
                                    "message": str(e),
                                    "type": "server_error",
                                    "traceback": traceback.format_exc()
                                }
                            }
                            metrics["error"] = str(e)
                            if ENABLE_LATENCY_METRICS:
                                logger.error(f"Streaming Error Metrics: {json.dumps(metrics)}")
                            
                            # Store error in request data
                            await store_request(request_id, metrics, raw_messages, full_output, True)
                            
                            # Remove from active connections
                            active_connections.discard(request_id)
                            
                            yield f"data: {json.dumps(error_msg)}\n\n"
                except Exception as e:
                    error_time = time.time() - start_time
                    metrics["error"] = str(e)
                    metrics["error_time"] = error_time
                    
                    if ENABLE_LATENCY_METRICS:
                        logger.error(f"Error Metrics: {json.dumps(metrics)}")
                    
                    # Store error in request data
                    await store_request(request_id, metrics, raw_messages, full_output, True)
                    
                    # Remove from active connections
                    active_connections.discard(request_id)
                    
                    error_msg = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                            "traceback": traceback.format_exc()
                        }
                    }
                    yield f"data: {json.dumps(error_msg)}\n\n"
                    yield "data: [DONE]\n\n"
            
            # Set up a more resilient streaming response
            response = Response(generate(), mimetype='text/event-stream')
            response.headers['X-Accel-Buffering'] = 'no'  # Disable proxy buffering
            response.headers['Cache-Control'] = 'no-cache'  # Prevent caching
            response.headers['Transfer-Encoding'] = 'chunked'  # Use chunked transfer for large responses
            
            # Add latency headers if enabled
            if ENABLE_LATENCY_HEADERS:
                response.headers['X-Server-Processing-Time'] = str(time.time() - start_time)
                response.headers['X-Request-ID'] = request_id
            
            return response
        else:
            # For non-streaming, use caching for identical requests
            messages_hash = hash(json.dumps(messages))
            llm_request_start = time.time()
            
            try:
                # Try to get result from cache for identical requests
                cache_start = time.time()
                result = await cached_completion(model, messages_hash, temperature, max_tokens)
                cache_time = time.time() - cache_start
                metrics["cache_lookup_time"] = cache_time
                metrics["from_cache"] = True
            except Exception:
                # If not in cache or error, make direct request
                metrics["from_cache"] = False
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai.api_key}",
                    "Accept-Encoding": "gzip, deflate"
                }
                
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
                
                connection_start = time.time()
                response = await client.post(
                    OPENAI_API_URL,
                    json=payload,
                    headers=headers
                )
                connection_time = time.time() - connection_start
                metrics["connection_time"] = connection_time
                
                result = response.json()
                
                # Store in cache
                if 'choices' in result:
                    cached_completion.cache_set(
                        (model, messages_hash, temperature, max_tokens), 
                        result
                    )
            
            llm_processing_time = time.time() - llm_request_start
            metrics["llm_processing_time"] = llm_processing_time
            
            # Estimate token count from non-streaming response
            if 'choices' in result:
                for choice in result.get('choices', []):
                    content = choice.get('message', {}).get('content', '')
                    if content:
                        # Rough approximation: 4 chars ~= 1 token
                        metrics["total_tokens"] = len(content) // 4
            
            # Format the response for Vapi
            formatting_start = time.time()
            formatted_response = {
                "id": result.get("id", ""),
                "object": "chat.completion",
                "created": result.get("created", 0),
                "model": result.get("model", model),
                "choices": [
                    {
                        "index": choice.get("index", 0),
                        "message": {
                            "role": choice.get("message", {}).get("role", "assistant"),
                            "content": choice.get("message", {}).get("content", "")
                        },
                        "finish_reason": choice.get("finish_reason", "stop")
                    }
                    for choice in result.get("choices", [])
                ]
            }
            formatting_time = time.time() - formatting_start
            metrics["formatting_time"] = formatting_time
            
            # Calculate final metrics
            metrics["total_request_time"] = time.time() - start_time
            metrics["server_processing_time"] = time.time() - start_time
            
            # Log metrics if enabled
            if ENABLE_LATENCY_METRICS:
                logger.info(f"Non-Streaming Latency Metrics: {json.dumps(metrics)}")
            
            # Apply gzip compression for responses
            response = await make_response(jsonify(formatted_response))
            response.headers['Content-Encoding'] = 'gzip'
            
            # Add latency headers if enabled
            if ENABLE_LATENCY_HEADERS:
                response.headers['X-Server-Processing-Time'] = str(metrics["server_processing_time"])
                response.headers['X-LLM-Processing-Time'] = str(metrics["llm_processing_time"])
                response.headers['X-From-Cache'] = str(metrics["from_cache"]).lower()
                response.headers['X-Total-Tokens'] = str(metrics["total_tokens"])
                response.headers['X-Request-ID'] = request_id
            
            response.set_data(gzip.compress(response.data))
            
            # Get response content for storage
            output_text = ""
            if "choices" in result and result["choices"]:
                output_text = result["choices"][0].get("message", {}).get("content", "")
            
            # Store completed request data
            await store_request(request_id, metrics, raw_messages, output_text, True)
            
            return response
        
    except Exception as e:
        error_time = time.time() - start_time
        metrics["error"] = str(e)
        metrics["error_time"] = error_time
        
        if ENABLE_LATENCY_METRICS:
            logger.error(f"Error Metrics: {json.dumps(metrics)}")
        
        # Store error in request data
        await store_request(request_id, metrics, raw_messages)
        
        error = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "traceback": traceback.format_exc()
            }
        }
        return jsonify(error), 500

# Optional legacy endpoint for explicit streaming
@app.route("/chat/completions/stream", methods=["POST"])
async def chat_completions_stream():
    # Redirect to the main endpoint with stream=True
    return await chat_completions()

# Endpoint to get latest latency metrics
@app.route("/metrics/latency", methods=["GET"])
async def get_latency_metrics():
    if not ENABLE_LATENCY_METRICS:
        return jsonify({"error": "Latency metrics are disabled"}), 404
    
    # This would be expanded to return actual metrics from a store
    # For now, we'll just return a message
    return jsonify({
        "message": "Latency metrics endpoint. In a real implementation, this would return aggregated metrics."
    })

if __name__ == "__main__":
    # Initialize database for dashboard
    init_db()
    
    # Register signal handlers for graceful shutdown
    import signal
    for sig in (signal.SIGINT, signal.SIGTERM):
        if hasattr(signal, 'SIGBREAK'):  # Windows-specific signal
            signal.signal(signal.SIGBREAK, lambda s, f: signal_handler())
        signal.signal(sig, lambda s, f: signal_handler())
    
    # Launch Streamlit dashboard with proper error handling
    threading.Thread(target=launch_dashboard, daemon=True).start()
    
    port = int(os.getenv("PORT", 5000))
    import hypercorn.asyncio
    import hypercorn.config
    
    # Optimize Hypercorn configuration for performance and stability
    config = hypercorn.config.Config()
    config.bind = [f"0.0.0.0:{port}"]
    config.use_reloader = True
    config.workers = int(os.getenv("WORKERS", 4))
    config.keep_alive_timeout = 120
    config.h2_max_concurrent_streams = 250
    config.graceful_timeout = 10.0  # Allow 10 seconds for graceful shutdown
    
    # Define async main function to properly handle async initialization
    async def main():
        # Warm up connection pool with retries
        await warmup_connection_pool()
        # Start the server
        await hypercorn.asyncio.serve(app, config)
    
    # Run the async main function
    asyncio.run(main()) 