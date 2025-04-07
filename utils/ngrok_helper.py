import subprocess
import time
import requests

def start_ngrok(port=5000):
    """
    Start een ngrok tunnel op de gegeven poort en retourneer de publieke URL.
    """
    # Start ngrok tunnel
    ngrok_process = subprocess.Popen(["ngrok", "http", str(port)],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
    
    # Wacht even zodat ngrok opstart
    time.sleep(2)
    
    # Probeer maximaal 10 keer de tunnel-URL op te halen
    max_retries = 10
    for _ in range(max_retries):
        try:
            tunnels = requests.get("http://127.0.0.1:4040/api/tunnels").json()
            public_url = None
            for tunnel in tunnels.get("tunnels", []):
                if tunnel.get("proto") == "https":
                    public_url = tunnel.get("public_url")
                    break
            if public_url:
                return public_url
        except Exception:
            time.sleep(1)
    
    raise RuntimeError("❌ Failed to start ngrok or retrieve public URL")
