#!/usr/bin/env bash
#
# Start the review app: nginx (port 7866) + Flask (port 5000).
# Run from the app directory with the 'gaa' conda env active.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGINX_PREFIX="$HOME/nginx"
NGINX_CONF="$SCRIPT_DIR/nginx.conf"

mkdir -p "$NGINX_PREFIX/logs" "$NGINX_PREFIX/temp"

# Stop any existing nginx using this prefix
if [ -f "$NGINX_PREFIX/logs/nginx.pid" ]; then
    echo "Stopping existing nginx..."
    nginx -p "$NGINX_PREFIX" -s stop 2>/dev/null || true
    sleep 1
fi

echo "Starting nginx (port 7866)..."
nginx -c "$NGINX_CONF" -p "$NGINX_PREFIX"
echo "  nginx PID: $(cat "$NGINX_PREFIX/logs/nginx.pid")"

echo "Starting Flask (port 5000)..."
cd "$SCRIPT_DIR"
python app.py
