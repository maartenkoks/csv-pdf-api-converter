import os
import subprocess
import time
import requests
import signal
from pathlib import Path

NGROK_BIN = "ngrok" if os.name != "nt" else "ngrok.exe"
NGROK_CONFIG_DIR = Path.home() / ".config/ngrok"
NGROK_CONFIG_FILE = NGROK_CONFIG_DIR / "ngrok.yml"


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    """Helper om synchronously een shell-commando te runnen."""
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def configure_authtoken(token: str) -> None:
    """
    Sla de Ngrok-authtoken op, maar alleen als die nog niet bestaat.
    """
    if NGROK_CONFIG_FILE.exists() and "authtoken:" in NGROK_CONFIG_FILE.read_text():
        # Token staat al in config; niets doen
        return
    _run_cmd([NGROK_BIN, "config", "add-authtoken", token])


def _kill_existing_ngrok() -> None:
    """
    Stop eventueel al draaiende ngrok-processen om fouten op poort 4040 te voorkomen.
    """
    try:
        resp = requests.get("http://127.0.0.1:4040/api/tunnels")
        if resp.ok:
            # Als 4040 luistert: kill alle lokale ngrok-processen
            subprocess.call(["pkill", "-f", NGROK_BIN])
            time.sleep(1)
    except requests.exceptions.ConnectionError:
        pass  # 4040 luistert niet → geen ngrok actief


def start_ngrok(port: int = 5000) -> str:
    """
    Start een ngrok-tunnel en retourneer de publieke https-URL.
    """
    _kill_existing_ngrok()

    # Start ngrok als background-proces
    ngrok_process = subprocess.Popen(
        [NGROK_BIN, "http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Poll maximaal 10 s op /api/tunnels
    for _ in range(20):
        try:
            tunnels = requests.get("http://127.0.0.1:4040/api/tunnels").json()
            for t in tunnels.get("tunnels", []):
                if t.get("proto") == "https":
                    return t["public_url"]
        except Exception:
            pass
        time.sleep(0.5)

    # Mislukt → proces killen en fout gooien
    ngrok_process.send_signal(signal.SIGINT)
    raise RuntimeError("❌ Ngrok is niet binnen 10 s gestart.")
