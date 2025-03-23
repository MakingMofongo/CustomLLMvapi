import streamlit as st
import time
import json
import pandas as pd
from collections import deque
import threading
import sqlite3
import os

# Set up SQLite for storing request data
DB_PATH = "requests.db"

def setup_db():
    conn = sqlite3.connect(DB_PATH)
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

# Initialize Streamlit page
st.set_page_config(
    page_title="Vapi LLM Chat Dashboard",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for chat bubbles
st.markdown("""
<style>
    /* User chat bubble (left) */
    .chat-bubble-user {
        background-color: #f0f2f6;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px 0px 5px 0px;
        display: inline-block;
        width: 100%;
        color: #000000;
    }
    
    /* Assistant chat bubble (right) */
    .chat-bubble-assistant {
        background-color: #175ece;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0px 5px 0px;
        display: inline-block;
        width: 100%;
    }
    
    /* Metrics display */
    .metrics-display {
        font-size: 11px;
        color: #666;
        text-align: right;
        margin: 5px 0;
    }
    
    /* Chat divider */
    .chat-divider {
        margin: 5px 0;
        opacity: 0.2;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversations' not in st.session_state:
    st.session_state.conversations = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# Function to fetch latest conversations
def fetch_latest_conversations():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, input, output, metrics, completed FROM requests ORDER BY timestamp DESC LIMIT 15")
        rows = cursor.fetchall()
        conn.close()
        
        conversations = []
        for row in rows:
            req_id, timestamp, input_text, output_text, metrics_json, completed = row
            try:
                metrics = json.loads(metrics_json) if metrics_json else {}
            except:
                metrics = {}
            
            conversations.append({
                "id": req_id,
                "timestamp": timestamp,
                "input": input_text,
                "output": output_text,
                "metrics": metrics,
                "completed": completed
            })
        
        return conversations
    except Exception as e:
        st.error(f"Error fetching conversations: {str(e)}")
        return []

# Format important metrics to display below chat
def format_metrics_for_display(metrics):
    key_metrics = []
    
    if "time_to_first_token" in metrics:
        ttft_value = metrics.get("time_to_first_token")
        if ttft_value is not None:
            ttft = round(float(ttft_value), 2)
            key_metrics.append(f"⏱️ First token: {ttft}s")
    
    if "tokens_per_second" in metrics:
        tps_value = metrics.get("tokens_per_second")
        if tps_value is not None:
            tps = round(float(tps_value), 1)
            key_metrics.append(f"⚡ {tps} tok/s")
    
    if "total_tokens" in metrics:
        tokens = metrics.get("total_tokens", 0)
        key_metrics.append(f"📊 {tokens} tokens")
    
    if "total_request_time" in metrics:
        time_value = metrics.get("total_request_time")
        if time_value is not None:
            total_time = round(float(time_value), 2)
            key_metrics.append(f"⏳ Total: {total_time}s")
        
    return " • ".join(key_metrics)

# Page title
st.title("Vapi LLM Chat Dashboard")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    refresh_btn = st.button("Refresh", help="Manually refresh the conversation list")
    auto_refresh = st.checkbox("Auto-refresh (3s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    if st.button("Clear History", help="Clear all conversation history"):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM requests")
            conn.commit()
            conn.close()
            st.success("Conversation history cleared!")
        except Exception as e:
            st.error(f"Error clearing history: {str(e)}")
    
    st.markdown("---")
    st.subheader("Metrics Legend")
    st.markdown("⏱️ **TTFT**: Time to first token")
    st.markdown("⚡ **tok/s**: Tokens per second")
    st.markdown("📊 **tokens**: Total tokens generated")
    st.markdown("⏳ **Total**: Complete request time")

# Create main layout
conversations = fetch_latest_conversations()
st.session_state.conversations = conversations

# Display conversations
st.subheader("Chat History")

# If we have conversations, display them
if conversations:
    for i, convo in enumerate(conversations):
        convo_id = convo["id"]
        input_text = convo["input"]
        output_text = convo["output"]
        metrics = convo["metrics"]
        timestamp = time.strftime('%H:%M:%S', time.localtime(convo.get("timestamp", 0)))
        completed = convo.get("completed", False)
        
        # Simple container for each conversation
        with st.container():
            # Create a popover for this conversation
            with st.popover(f"Conversation at {timestamp}", use_container_width=True):
                # Detailed metrics
                st.subheader(f"Conversation Details")
                metrics = convo.get("metrics", {})
                
                # Create multiple columns for key metrics
                cols = st.columns(4)
                if "time_to_first_token" in metrics and metrics["time_to_first_token"] is not None:
                    cols[0].metric("Time to First Token (s)", round(float(metrics["time_to_first_token"]), 3))
                if "tokens_per_second" in metrics and metrics["tokens_per_second"] is not None:
                    cols[1].metric("Tokens per Second", round(float(metrics["tokens_per_second"]), 1))
                if "total_tokens" in metrics:
                    cols[2].metric("Total Tokens", metrics.get("total_tokens", 0))
                if "total_request_time" in metrics and metrics["total_request_time"] is not None:
                    cols[3].metric("Total Request Time (s)", round(float(metrics["total_request_time"]), 3))
                
                # Show all metrics in a dataframe
                st.markdown("### All Metrics")
                df = pd.DataFrame([{"Metric": k, "Value": str(v)} for k, v in metrics.items()])
                st.dataframe(df, use_container_width=True)
                
                # Add the conversation for context
                st.markdown("### Conversation")
                st.markdown('<p><strong>User:</strong></p>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-bubble-user">{input_text}</div>', unsafe_allow_html=True)
                st.markdown('<p style="text-align:right;"><strong>Assistant:</strong></p>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-bubble-assistant">{output_text}</div>', unsafe_allow_html=True)
            
            # User message
            st.markdown('<p><strong>User</strong></p>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-user">{input_text}</div>', unsafe_allow_html=True)
            
            # Assistant message
            assistant_label = "Assistant" if completed else "Assistant (typing...)"
            st.markdown(f'<p style="text-align:right;"><strong>{assistant_label}</strong></p>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-assistant">{output_text}</div>', unsafe_allow_html=True)
            
            # Metrics display
            metrics_text = format_metrics_for_display(metrics)
            st.markdown(f'<div class="metrics-display">{metrics_text}</div>', unsafe_allow_html=True)
            
            # Add divider between conversations
            st.markdown('<hr class="chat-divider">', unsafe_allow_html=True)
else:
    st.info("No conversations yet. Start chatting with the LLM to see them here.")

# Auto-refresh
if st.session_state.auto_refresh:
    time.sleep(0.1)  # Small delay to prevent too frequent refreshes
    st.rerun()

# Setup the database when first run
if not os.path.exists(DB_PATH):
    setup_db() 