#!/bin/bash
# start_with_ngrok.sh: Start Flask op poort 8000 en Ngrok, toon publieke URL, stop alles bij Ctrl+C

set -e

export FLASK_APP=app.py
export FLASK_ENV=production
export FLASK_RUN_PORT=8000

mkdir -p data uploads temp_uploads

# Start Flask op de achtergrond
flask run --port=8000 &
FLASK_PID=$!

# Start Ngrok op poort 8000
./ngrok http 8000 > /dev/null &
NGROK_PID=$!

# Wacht tot ngrok tunnel actief is
sleep 4

# Haal publieke URL op
NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | grep -o 'https://[^"]*')

if [ -n "$NGROK_URL" ]; then
  echo "Ngrok public URL: $NGROK_URL"
else
  echo "Ngrok tunnel kon niet worden opgehaald."
fi

echo "Druk Ctrl+C om beide processen te stoppen."

trap 'echo "\nStoppen..."; kill $FLASK_PID $NGROK_PID 2>/dev/null; exit 0' SIGINT

wait $FLASK_PID $NGROK_PID
