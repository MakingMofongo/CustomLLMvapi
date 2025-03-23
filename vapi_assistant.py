import requests
import os
from dotenv import load_dotenv
from ngrok_utils import get_active_ngrok_url
from typing import List, Dict
from app import store_assistant, delete_assistant
import asyncio

class VAPIAssistant:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('VAPI_API_KEY')
        self.base_url = "https://api.vapi.ai/assistant"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def create_assistant(self, name: str, transcriber_provider: str = "deepgram", 
                        language: str = "en", model: str = "", 
                        messages: List[Dict[str, str]] = None,
                        first_message: str = "", first_message_mode: str = "",
                        voice_provider: str = "11labs", voice_id: str = ""):
        """
        Create a new VAPI assistant
        
        Args:
            name (str): Name of the assistant
            transcriber_provider (str): Provider for transcription (default: "deepgram")
            language (str): Language code (default: "en")
            model (str): Model name
            messages (List[Dict[str, str]]): List of message objects with roles and content
            first_message (str): First message from assistant
            first_message_mode (str): Mode for first message (must be one of: assistant-speaks-first, 
                                    assistant-speaks-first-with-model-generated-message, assistant-waits-for-user)
            voice_provider (str): Provider for voice (default: "11labs")
            voice_id (str): ID of the voice to use
        
        Returns:
            dict: Response from the API
        """
        try:
            # Get ngrok URL for the model endpoint
            ngrok_url = get_active_ngrok_url()
            if not ngrok_url:
                raise Exception("Failed to get ngrok URL")

            payload = {
                "name": name,
                "transcriber": {
                    "provider": transcriber_provider,
                    "language": language
                },
                "model": {
                    "provider": "custom-llm",
                    "model": model,
                    "url": ngrok_url,
                },
                "firstMessage": first_message,
                "firstMessageMode": first_message_mode,
                "voice": {
                    "provider": voice_provider,
                    "voiceId": voice_id
                }
            }

            # Only add messages if provided
            if messages is not None:
                payload["model"]["messages"] = messages

            # Print the payload for debugging
            print("Sending payload:", payload)
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Store complete assistant data from API response
            assistant_data = {
                "id": response_data["id"],
                "name": response_data.get("name"),
                "description": response_data.get("description"),
                "model": response_data.get("model", {}).get("model"),
                "transcriber_provider": response_data.get("transcriber", {}).get("provider"),
                "language": response_data.get("transcriber", {}).get("language"),
                "messages": response_data.get("model", {}).get("messages"),
                "first_message": response_data.get("firstMessage"),
                "first_message_mode": response_data.get("firstMessageMode"),
                "voice_provider": response_data.get("voice", {}).get("provider"),
                "voice_id": response_data.get("voice", {}).get("voiceId")
            }
            
            # Store in database
            asyncio.run(store_assistant(assistant_data))
            
            return response_data
        except requests.exceptions.RequestException as e:
            print(f"Failed to create assistant: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response content: {e.response.content}")
            raise

    def update_assistant(self, assistant_id: str, name: str = None, 
                        transcriber_provider: str = None, language: str = None,
                        model: str = None, messages: List[Dict[str, str]] = None,
                        first_message: str = None, first_message_mode: str = None,
                        voice_provider: str = None, voice_id: str = None,
                        description: str = None):
        """
        Update an existing VAPI assistant
        
        Args:
            assistant_id (str): ID of the assistant to update
            name (str): Name of the assistant
            transcriber_provider (str): Provider for transcription
            language (str): Language code
            model (str): Model name
            messages (List[Dict[str, str]]): List of message objects with roles and content
            first_message (str): First message from assistant
            first_message_mode (str): Mode for first message (must be one of: assistant-speaks-first, 
                                    assistant-speaks-first-with-model-generated-message, assistant-waits-for-user)
            voice_provider (str): Provider for voice
            voice_id (str): ID of the voice to use
            description (str): Description of the assistant
        
        Returns:
            dict: Response from the API
        """
        try:
            # First, get the current assistant data
            current_assistant = requests.get(
                f"{self.base_url}/{assistant_id}",
                headers=self.headers
            )
            current_assistant.raise_for_status()
            current_data = current_assistant.json()

            # Get ngrok URL for the model endpoint
            ngrok_url = get_active_ngrok_url()
            if not ngrok_url:
                raise Exception("Failed to get ngrok URL")

            # Build payload starting with current data
            payload = {}

            # Update name if provided
            if name is not None:
                payload["name"] = name
            
            # Remove description from payload since VAPI API doesn't support it
            
            # Handle transcriber settings
            current_transcriber = current_data.get("transcriber", {})
            if transcriber_provider is not None or language is not None:
                transcriber = current_transcriber.copy()
                if transcriber_provider is not None:
                    transcriber["provider"] = transcriber_provider
                if language is not None:
                    transcriber["language"] = language
                payload["transcriber"] = transcriber

            # Handle model settings
            current_model = current_data.get("model", {})
            if model is not None or messages is not None:
                model_config = current_model.copy()
                model_config["provider"] = "custom-llm"
                model_config["url"] = ngrok_url
                if model is not None:
                    model_config["model"] = model
                if messages is not None:
                    model_config["messages"] = messages
                payload["model"] = model_config

            # Handle first message settings
            if first_message is not None:
                payload["firstMessage"] = first_message
            if first_message_mode is not None:
                payload["firstMessageMode"] = first_message_mode
            
            # Handle voice settings
            current_voice = current_data.get("voice", {})
            if voice_provider is not None or voice_id is not None:
                voice = current_voice.copy()
                if voice_provider is not None:
                    voice["provider"] = voice_provider
                if voice_id is not None:
                    voice["voiceId"] = voice_id
                payload["voice"] = voice

            # Only proceed if we have fields to update
            if not payload:
                raise Exception("No valid fields provided for update")

            print("Update payload:", payload)
            response = requests.patch(
                f"{self.base_url}/{assistant_id}",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Update assistant data in database
            # Only include fields that were actually updated
            assistant_data = {
                "id": assistant_id
            }
            if name is not None:
                assistant_data["name"] = name
            if description is not None:
                assistant_data["description"] = description
            if model is not None:
                assistant_data["model"] = model
            if transcriber_provider is not None:
                assistant_data["transcriber_provider"] = transcriber_provider
            if language is not None:
                assistant_data["language"] = language
            if messages is not None:
                assistant_data["messages"] = messages
            if first_message is not None:
                assistant_data["first_message"] = first_message
            if first_message_mode is not None:
                assistant_data["first_message_mode"] = first_message_mode
            if voice_provider is not None:
                assistant_data["voice_provider"] = voice_provider
            if voice_id is not None:
                assistant_data["voice_id"] = voice_id
            
            asyncio.run(store_assistant(assistant_data))
            
            return response_data

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"{error_msg} - {error_data.get('message', '')}"
                except:
                    pass
            print(f"Failed to update assistant: {error_msg}")
            if hasattr(e, 'response'):
                print(f"Response content: {e.response.content}")
            raise Exception(f"Failed to update assistant: {error_msg}")

    def delete_assistant(self, assistant_id: str):
        """
        Delete a VAPI assistant
        
        Args:
            assistant_id (str): ID of the assistant to delete
        
        Returns:
            dict: Response from the API
        """
        try:
            response = requests.delete(
                f"{self.base_url}/{assistant_id}",
                headers=self.headers
            )
            response.raise_for_status()
            
            # Delete assistant from database
            asyncio.run(delete_assistant(assistant_id))
            
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to delete assistant: {str(e)}")

    def update_model_url_for_all(self):
        """
        Update the model URL for all assistants with the latest ngrok URL
        
        Returns:
            dict: Summary of the update operation
        """
        try:
            # Get new ngrok URL
            ngrok_url = get_active_ngrok_url()
            if not ngrok_url:
                raise Exception("Failed to get ngrok URL")

            # Get all assistants
            response = requests.get(
                self.base_url,
                headers=self.headers
            )
            response.raise_for_status()
            assistants = response.json()

            results = {
                "success": [],
                "failed": [],
                "total": len(assistants),
                "new_url": ngrok_url
            }

            # Update each assistant
            for assistant in assistants:
                try:
                    assistant_id = assistant["id"]
                    current_model = assistant.get("model", {})
                    
                    # Prepare model payload
                    model_config = current_model.copy()
                    model_config["provider"] = "custom-llm"
                    model_config["url"] = ngrok_url

                    # Update assistant
                    update_response = requests.patch(
                        f"{self.base_url}/{assistant_id}",
                        headers=self.headers,
                        json={"model": model_config}
                    )
                    update_response.raise_for_status()
                    results["success"].append(assistant_id)
                except Exception as e:
                    results["failed"].append({
                        "id": assistant_id,
                        "error": str(e)
                    })

            return results
        except Exception as e:
            raise Exception(f"Failed to update model URLs: {str(e)}")
