from pyngrok import ngrok
import requests
import os
import time
import threading
from dotenv import load_dotenv

# Global variable to track the tunnel thread
_tunnel_thread = None
_tunnel_url = None

def start_ngrok(port=5000):
    """
    Start an ngrok tunnel on the specified port.
    Returns the public URL.
    """
    try:
        # Load ngrok token from environment if available
        load_dotenv()
        ngrok_auth_token = os.getenv('NGROK_AUTH_TOKEN')
        if ngrok_auth_token:
            ngrok.set_auth_token(ngrok_auth_token)

        # Start ngrok if it's not already running
        tunnels = ngrok.get_tunnels()
        if not tunnels:
            # Start HTTP tunnel
            public_url = ngrok.connect(port).public_url
            print(f"Started ngrok tunnel at: {public_url}")
            return public_url
        
        return tunnels[0].public_url

    except Exception as e:
        print(f"Error starting ngrok: {str(e)}")
        return None

def _keep_tunnel_alive():
    """
    Internal function to keep the tunnel running in a separate thread
    """
    while True:
        time.sleep(1)

def get_active_ngrok_url(port=5000):
    """
    Get or create an active ngrok URL.
    If a tunnel is already running, returns its URL.
    If no tunnel exists, creates one and returns its URL.
    
    Args:
        port (int): The port to tunnel to (default: 5000)
    
    Returns:
        str: The ngrok URL or None if failed
    """
    global _tunnel_thread, _tunnel_url

    try:
        # Check if we already have an active tunnel
        if _tunnel_thread and _tunnel_thread.is_alive():
            return _tunnel_url

        # Start a new tunnel
        url = start_ngrok(port)
        if url:
            _tunnel_url = url
            # Start a daemon thread to keep the tunnel alive
            _tunnel_thread = threading.Thread(target=_keep_tunnel_alive, daemon=True)
            _tunnel_thread.start()
            return url
        
        return None

    except Exception as e:
        print(f"Error in get_active_ngrok_url: {str(e)}")
        return None

def stop_tunnel():
    """
    Stop the ngrok tunnel and cleanup
    """
    try:
        ngrok.kill()
        global _tunnel_thread, _tunnel_url
        _tunnel_thread = None
        _tunnel_url = None
        print("Ngrok tunnel closed")
    except Exception as e:
        print(f"Error stopping tunnel: {str(e)}")

def get_ngrok_url():
    """
    Get the public URL of the active ngrok tunnel.
    Returns the URL as a string, or None if no tunnel is found.
    """
    try:
        # Get all tunnels
        tunnels = ngrok.get_tunnels()
        
        # Return the public URL of the first tunnel if available
        if tunnels:
            return tunnels[0].public_url
        
        return None
    
    except Exception as e:
        print(f"Error getting ngrok URL: {str(e)}")
        return None

def get_ngrok_url_api():
    """
    Alternative method to get ngrok URL using the ngrok API.
    Requires NGROK_AUTH_TOKEN in environment variables.
    """
    try:
        load_dotenv()
        ngrok_auth_token = os.getenv('NGROK_AUTH_TOKEN')
        
        if not ngrok_auth_token:
            print("NGROK_AUTH_TOKEN not found in environment variables")
            return None

        headers = {
            'Authorization': f'Bearer {ngrok_auth_token}',
            'Ngrok-Version': '2'
        }
        
        response = requests.get('https://api.ngrok.com/tunnels', headers=headers)
        
        if response.status_code == 200:
            tunnels = response.json().get('tunnels', [])
            if tunnels:
                return tunnels[0]['public_url']
        
        return None

    except Exception as e:
        print(f"Error getting ngrok URL via API: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    url = get_active_ngrok_url()
    if url:
        print(f"Ngrok tunnel URL: {url}")
        print("Tunnel is active. Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Closing ngrok tunnel...")
            stop_tunnel()
