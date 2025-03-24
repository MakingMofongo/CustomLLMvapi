import streamlit as st
import time
import json
import pandas as pd
from collections import deque
import threading
import sqlite3
import os

# Add new imports for assistant management
from typing import Optional, Dict, List

# Add import for VAPIAssistant
from vapi_assistant import VAPIAssistant

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

def setup_assistant_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    DROP TABLE IF EXISTS assistants
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS assistants (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        model TEXT NOT NULL,
        transcriber_provider TEXT,
        language TEXT,
        messages TEXT,
        first_message TEXT,
        first_message_mode TEXT,
        voice_provider TEXT,
        voice_id TEXT,
        created_at REAL,
        updated_at REAL
    )
    ''')
    conn.commit()
    conn.close()

# Add Assistant Management Functions
def save_assistant(assistant_data: Dict) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        current_time = time.time()
        
        if 'id' in assistant_data:  # Update
            cursor.execute('''
                UPDATE assistants 
                SET name=?, description=?, model=?, transcriber_provider=?, 
                    language=?, messages=?, first_message=?, first_message_mode=?,
                    voice_provider=?, voice_id=?, updated_at=?
                WHERE id=?
            ''', (
                assistant_data['name'],
                assistant_data.get('description', ''),
                assistant_data['model'],
                assistant_data.get('transcriber_provider', ''),
                assistant_data.get('language', ''),
                json.dumps(assistant_data.get('messages', [])),
                assistant_data.get('first_message', ''),
                assistant_data.get('first_message_mode', ''),
                assistant_data.get('voice_provider', ''),
                assistant_data.get('voice_id', ''),
                current_time,
                assistant_data['id']
            ))
        else:  # Create new
            assistant_data['id'] = f"asst_{int(current_time)}"
            cursor.execute('''
                INSERT INTO assistants (
                    id, name, description, model, transcriber_provider,
                    language, messages, first_message, first_message_mode,
                    voice_provider, voice_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                assistant_data['id'],
                assistant_data['name'],
                assistant_data.get('description', ''),
                assistant_data['model'],
                assistant_data.get('transcriber_provider', ''),
                assistant_data.get('language', ''),
                json.dumps(assistant_data.get('messages', [])),
                assistant_data.get('first_message', ''),
                assistant_data.get('first_message_mode', ''),
                assistant_data.get('voice_provider', ''),
                assistant_data.get('voice_id', ''),
                current_time,
                current_time
            ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving assistant: {str(e)}")
        return False

def get_assistants() -> List[Dict]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, description, model, transcriber_provider,
                   language, messages, first_message, first_message_mode,
                   voice_provider, voice_id, created_at, updated_at 
            FROM assistants 
            ORDER BY created_at DESC
        ''')
        assistants = []
        for row in cursor.fetchall():
            assistants.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "model": row[3],
                "transcriber_provider": row[4],
                "language": row[5],
                "messages": json.loads(row[6]) if row[6] else [],
                "first_message": row[7],
                "first_message_mode": row[8],
                "voice_provider": row[9],
                "voice_id": row[10],
                "created_at": row[11],
                "updated_at": row[12]
            })
        conn.close()
        return assistants
    except Exception as e:
        st.error(f"Error fetching assistants: {str(e)}")
        return []

def delete_assistant(assistant_id: str) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM assistants WHERE id = ?", (assistant_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting assistant: {str(e)}")
        return False

# Add function to manage message roles
def render_message_form(key_prefix: str, existing_messages: List[Dict] = None):
    messages = existing_messages or []
    
    # Container for all messages
    st.write("System Messages")
    messages_container = st.container()
    
    # Add new message section
    col1, col2 = st.columns([6, 1])
    with col1:
        new_role = st.selectbox(
            "New Message Role",
            ["assistant", "function", "system", "tool", "user"],
            key=f"{key_prefix}_new_role"
        )
    with col2:
        # This will be handled in the form submission
        st.markdown("&nbsp;")  # Spacer for alignment
        
    new_content = st.text_area(
        "New Message Content",
        key=f"{key_prefix}_new_content",
        height=100
    )
    
    # Store current messages in session state
    if f"{key_prefix}_messages" not in st.session_state:
        st.session_state[f"{key_prefix}_messages"] = messages
    
    # Display existing messages
    for i, msg in enumerate(st.session_state[f"{key_prefix}_messages"]):
        with messages_container:
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 6, 1])
            with col1:
                st.write(f"**Role:** {msg['role']}")
            with col2:
                st.write(f"**Content:** {msg['content']}")
            with col3:
                # Mark for deletion (will be handled in form submission)
                st.checkbox("Delete", key=f"{key_prefix}_delete_{i}")
    
    # Return current messages and new message data
    return {
        "current_messages": st.session_state[f"{key_prefix}_messages"],
        "new_message": {
            "role": new_role,
            "content": new_content
        }
    }

# Update the render_assistant_management function
def render_assistant_management():
    st.title("Assistant Management")
    vapi = VAPIAssistant()
    
    # Add tab selection in the sidebar with new Show Assistants option
    operation = st.sidebar.radio(
        "Operation",
        ["Show Assistants", "Create Assistant", "Update Assistant", "Delete Assistant", "Update Model URL"],
        key="assistant_operation"
    )
    
    # Add refresh button in the header
    col1, col2 = st.columns([6, 1])
    with col1:
        st.subheader("Assistants")
    with col2:
        if st.button("🔄 Refresh", help="Refresh the list of assistants"):
            st.rerun()

    # Handle different operations
    if operation == "Update Model URL":
        st.subheader("Update Model URL for All Assistants")
        st.write("This will update the model URL for all assistants with the latest ngrok URL.")
        
        if st.button("Update Model URL for All Assistants", type="primary"):
            try:
                with st.spinner("Updating model URLs..."):
                    results = vapi.update_model_url_for_all()
                
                # Display results
                st.success(f"Successfully updated {len(results['success'])} out of {results['total']} assistants")
                st.write(f"New URL: `{results['new_url']}`")
                
                if results['failed']:
                    st.error("Some updates failed:")
                    for failure in results['failed']:
                        st.write(f"- Assistant ID {failure['id']}: {failure['error']}")
                        
                # Show successful updates
                if results['success']:
                    with st.expander("Successfully updated assistants"):
                        for assistant_id in results['success']:
                            st.write(f"- {assistant_id}")
            except Exception as e:
                st.error(f"Error updating model URLs: {str(e)}")
    
    elif operation == "Show Assistants":
        # Get all assistants
        assistants = get_assistants()
        if not assistants:
            st.info("No assistants created yet.")
            return
            
        # Display assistants with detailed configuration
        for assistant in assistants:
            with st.expander(f"{assistant['name']} ({assistant['model']})", expanded=True):
                # Basic Information
                st.markdown("### Basic Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID:** {assistant['id']}")
                    st.write(f"**Name:** {assistant['name']}")
                    st.write(f"**Model:** {assistant['model']}")
                with col2:
                    # Handle timestamps more safely
                    try:
                        created_at = float(assistant['created_at']) if isinstance(assistant['created_at'], (int, float, str)) else 0
                        updated_at = float(assistant['updated_at']) if isinstance(assistant['updated_at'], (int, float, str)) else 0
                        
                        st.write(f"**Created:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at))}")
                        st.write(f"**Last Updated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(updated_at))}")
                    except (ValueError, TypeError):
                        st.write("**Created:** Not available")
                        st.write("**Last Updated:** Not available")
                
                if assistant.get('description'):
                    st.write(f"**Description:** {assistant['description']}")
                
                # Transcriber Settings
                st.markdown("### Transcriber Settings")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Provider:** {assistant.get('transcriber_provider', 'Not set')}")
                with col2:
                    st.write(f"**Language:** {assistant.get('language', 'Not set')}")
                
                # Voice Settings
                st.markdown("### Voice Settings")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Provider:** {assistant.get('voice_provider', 'Not set')}")
                with col2:
                    st.write(f"**Voice ID:** {assistant.get('voice_id', 'Not set')}")
                
                # First Message Settings
                st.markdown("### First Message Settings")
                st.write(f"**Mode:** {assistant.get('first_message_mode', 'Not set')}")
                if assistant.get('first_message'):
                    st.write("**First Message:**")
                    st.text(assistant.get('first_message'))
                
                # System Messages
                if assistant.get('messages'):
                    st.markdown("### System Messages")
                    try:
                        messages = json.loads(assistant['messages']) if isinstance(assistant['messages'], str) else assistant['messages']
                        for msg in messages:
                            with st.expander(f"Message ({msg.get('role', 'unknown')})"):
                                st.text(msg.get('content', ''))
                    except Exception as e:
                        st.error(f"Error displaying messages: {str(e)}")

    elif operation == "Create Assistant":
        # Create new assistant section
        with st.expander("Create New Assistant", expanded=True):
            # Check for success message in session state and display it
            if 'create_success' in st.session_state:
                st.success(st.session_state.create_success)
                # Clear the success message after displaying
                del st.session_state.create_success
                # Clear all form fields with empty values
                st.session_state.update({
                    'new_name': '',
                    'new_description': '',
                    'new_transcriber': 'assembly-ai',  # Default value
                    'new_language': 'en',  # Default value
                    'new_model': 'gpt-4',  # Default value
                    'new_voice_provider': 'vapi',  # Default value
                    'new_voice_id': '',
                    'new_first_message': '',
                    'new_first_message_mode': 'assistant-waits-for-user',  # Default value
                    'new_messages': []
                })
                # Also clear any message-related keys
                for key in list(st.session_state.keys()):
                    if key.startswith('new_delete_'):
                        del st.session_state[key]
                st.rerun()

            with st.form("new_assistant_form"):
                name = st.text_input("Name*", value=st.session_state.get('new_name', ''), key="new_name")
                description = st.text_area("Description", value=st.session_state.get('new_description', ''), key="new_description")
                
                # Create a two-column layout for settings
                col1, col2 = st.columns(2)
                
                with col1:
                    # Transcriber settings
                    st.subheader("Transcriber Settings")
                    transcriber = st.selectbox(
                        "Transcriber Provider*",
                        ["assembly-ai", "deepgram", "azure", "gladia", "talkscriber"],
                        index=["assembly-ai", "deepgram", "azure", "gladia", "talkscriber"].index(
                            st.session_state.get('new_transcriber', 'assembly-ai')
                        ),
                        key="new_transcriber"
                    )
                    language = st.text_input("Language Code", 
                                           value=st.session_state.get('new_language', 'en'), 
                                           key="new_language")
                    
                    # Model settings
                    st.subheader("Model Settings")
                    model = st.selectbox(
                        "Model*",
                        ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
                        index=["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"].index(
                            st.session_state.get('new_model', 'gpt-4')
                        ),
                        key="new_model"
                    )
                
                with col2:
                    # Voice settings
                    st.subheader("Voice Settings")
                    voice_provider = st.selectbox(
                        "Voice Provider*",
                        ["vapi", "11labs", "azure", "cartesia", "custom-voice", "deepgram", 
                         "hume", "lmnt", "neets", "neuphonic", "openai", "playht", 
                         "rime-ai", "smallest-ai", "tavus"],
                        key="new_voice_provider"
                    )
                    voice_id = st.text_input(
                        "Voice ID",
                        help="Common options: andrea, burt, drew, joseph, mark",
                        key="new_voice_id"
                    )
                    
                    # First message settings
                    st.subheader("First Message Settings")
                    first_message = st.text_area("First Message", key="new_first_message")
                    first_message_mode = st.selectbox(
                        "First Message Mode*",
                        ["assistant-speaks-first",
                         "assistant-speaks-first-with-model-generated-message",
                         "assistant-waits-for-user"],
                        key="new_first_message_mode"
                    )
                
                # System messages moved to bottom
                st.markdown("---")
                st.subheader("System Messages")
                message_data = render_message_form("new")
                
                # Add Message button right after system messages
                add_message = st.form_submit_button("Add Message", use_container_width=True)
                
                if add_message:
                    new_msg = message_data["new_message"]
                    if new_msg["content"]:  # Only add if content is not empty
                        current_messages = message_data["current_messages"]
                        current_messages.append(new_msg)
                        st.session_state[f"new_messages"] = current_messages
                        st.rerun()
                
                # Create Assistant button at the bottom with proper spacing
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                with col2:
                    submit = st.form_submit_button("Create Assistant", type="primary", use_container_width=True)
                
                # Handle form submission
                if submit:
                    if not name:
                        st.error("Name is required")
                    else:
                        # Process deletions
                        current_messages = message_data["current_messages"]
                        messages_to_keep = []
                        for i, msg in enumerate(current_messages):
                            if not st.session_state.get(f"new_delete_{i}", False):
                                messages_to_keep.append(msg)
                        
                        try:
                            response = vapi.create_assistant(
                                name=name,
                                transcriber_provider=transcriber,
                                language=language,
                                model=model,
                                messages=messages_to_keep if messages_to_keep else None,
                                first_message=first_message,
                                first_message_mode=first_message_mode,
                                voice_provider=voice_provider,
                                voice_id=voice_id
                            )
                            # Store success message in session state
                            st.session_state.create_success = f"Assistant created successfully! ID: {response.get('id')}"
                            # Clear the messages from session state
                            if "new_messages" in st.session_state:
                                del st.session_state["new_messages"]
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error creating assistant: {str(e)}")

    elif operation == "Update Assistant":
        # Get existing assistants
        assistants = get_assistants()
        if not assistants:
            st.info("No assistants available to update.")
            return
            
        # Select assistant to update
        selected_assistant = st.selectbox(
            "Select Assistant to Update",
            options=assistants,
            format_func=lambda x: f"{x['name']} ({x['model']})"
        )
        
        if selected_assistant:
            with st.form(f"edit_assistant_{selected_assistant['id']}"):
                name = st.text_input("Name*", 
                                   value=selected_assistant['name'],
                                   key=f"edit_name_{selected_assistant['id']}")
                description = st.text_area("Description",
                                         value=selected_assistant.get('description', ''),
                                         key=f"edit_desc_{selected_assistant['id']}")
                
                # Transcriber settings
                st.subheader("Transcriber Settings")
                transcriber = st.selectbox(
                    "Transcriber Provider*",
                    ["assembly-ai", "deepgram", "azure", "gladia", "talkscriber"],
                    index=["assembly-ai", "deepgram", "azure", "gladia", "talkscriber"].index(
                        selected_assistant.get('transcriber_provider', 'assembly-ai')
                    ) if selected_assistant.get('transcriber_provider') in ["assembly-ai", "deepgram", "azure", "gladia", "talkscriber"] else 0,
                    key=f"edit_transcriber_{selected_assistant['id']}"
                )
                language = st.text_input("Language Code",
                                       value=selected_assistant.get('language', 'en'),
                                       key=f"edit_language_{selected_assistant['id']}")
                
                # Model settings
                st.subheader("Model Settings")
                model = st.selectbox(
                    "Model*",
                    ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
                    index=["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"].index(selected_assistant['model']),
                    key=f"edit_model_{selected_assistant['id']}"
                )
                
                # System messages
                st.subheader("System Messages")
                if f"edit_{selected_assistant['id']}_messages" not in st.session_state:
                    st.session_state[f"edit_{selected_assistant['id']}_messages"] = selected_assistant.get('messages', [])
                message_data = render_message_form(f"edit_{selected_assistant['id']}")
                
                # Voice settings
                st.subheader("Voice Settings")
                voice_providers = ["vapi", "11labs", "azure", "cartesia", "custom-voice", "deepgram",
                                 "hume", "lmnt", "neets", "neuphonic", "openai", "playht",
                                 "rime-ai", "smallest-ai", "tavus"]
                voice_provider = st.selectbox(
                    "Voice Provider*",
                    voice_providers,
                    index=voice_providers.index(selected_assistant.get('voice_provider', 'vapi'))
                    if selected_assistant.get('voice_provider') in voice_providers else 0,
                    key=f"edit_voice_provider_{selected_assistant['id']}"
                )
                voice_id = st.text_input(
                    "Voice ID",
                    value=selected_assistant.get('voice_id', ''),
                    help="Common options: andrea, burt, drew, joseph, mark",
                    key=f"edit_voice_id_{selected_assistant['id']}"
                )
                
                # First message settings
                st.subheader("First Message Settings")
                first_message = st.text_area("First Message",
                                           value=selected_assistant.get('first_message', ''),
                                           key=f"edit_first_message_{selected_assistant['id']}")
                first_message_modes = ["assistant-speaks-first",
                                     "assistant-speaks-first-with-model-generated-message",
                                     "assistant-waits-for-user"]
                first_message_mode = st.selectbox(
                    "First Message Mode*",
                    first_message_modes,
                    index=first_message_modes.index(selected_assistant.get('first_message_mode', 'assistant-waits-for-user'))
                    if selected_assistant.get('first_message_mode') in first_message_modes else 2,
                    key=f"edit_first_message_mode_{selected_assistant['id']}"
                )
                
                # Update button
                if st.form_submit_button("Update"):
                    if not name:
                        st.error("Name is required")
                    else:
                        try:
                            vapi.update_assistant(
                                assistant_id=selected_assistant['id'],
                                name=name,
                                description=description,
                                transcriber_provider=transcriber,
                                language=language,
                                model=model,
                                messages=message_data["current_messages"] if message_data["current_messages"] else None,
                                first_message=first_message,
                                first_message_mode=first_message_mode,
                                voice_provider=voice_provider,
                                voice_id=voice_id
                            )
                            st.success("Assistant updated successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error updating assistant: {str(e)}")
                
    else:  # Delete Assistant
        # Get existing assistants
        assistants = get_assistants()
        if not assistants:
            st.info("No assistants available to delete.")
            return
            
        # Select assistant to delete
        selected_assistant = st.selectbox(
            "Select Assistant to Delete",
            options=assistants,
            format_func=lambda x: f"{x['name']} ({x['model']})"
        )
        
        if selected_assistant:
            # Show assistant details
            st.write(f"**Name:** {selected_assistant['name']}")
            st.write(f"**Model:** {selected_assistant['model']}")
            if selected_assistant.get('description'):
                st.write(f"**Description:** {selected_assistant['description']}")
            
            # Confirm deletion
            if st.button("Delete Assistant", type="primary"):
                try:
                    vapi.delete_assistant(selected_assistant['id'])
                    st.success("Assistant deleted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting assistant: {str(e)}")

    # Display list of existing assistants only for Update and Delete operations
    if operation not in ["Create Assistant", "Show Assistants"]:
        st.subheader("Existing Assistants")
        assistants = get_assistants()
        
        if not assistants:
            st.info("No assistants created yet.")
            return

        # Display assistants in a more compact way
        for assistant in assistants:
            with st.expander(f"{assistant['name']} ({assistant['model']})"):
                st.write(f"**ID:** {assistant['id']}")
                if assistant.get('description'):
                    st.write(f"**Description:** {assistant['description']}")
                st.write(f"**Model:** {assistant['model']}")
                
                # Handle timestamps more safely
                try:
                    created_at = float(assistant['created_at']) if isinstance(assistant['created_at'], (int, float, str)) else 0
                    updated_at = float(assistant['updated_at']) if isinstance(assistant['updated_at'], (int, float, str)) else 0
                    
                    st.write(f"**Created:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at))}")
                    st.write(f"**Last Updated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(updated_at))}")
                except (ValueError, TypeError):
                    st.write("**Created:** Not available")
                    st.write("**Last Updated:** Not available")

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
    
    # Add tab selection
    tab = st.radio("View", ["Chat History", "Assistant Management"])
    
    if tab == "Chat History":
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

# Main content area
if tab == "Chat History":
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
else:
    render_assistant_management()

# Setup the database tables when first run
if not os.path.exists(DB_PATH):
    setup_db()
    setup_assistant_table()