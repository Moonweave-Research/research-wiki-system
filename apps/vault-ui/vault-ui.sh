#!/usr/bin/env bash
# vault-ui control: dev (two servers) or production (single always-on uvicorn that
# serves the built SPA + API), plus launchd install for auto-start at login.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="$ROOT/apps/vault-ui"
DIST="$APP/frontend/dist"
RUN_DIR="$ROOT/.run"
LOG_DIR="$HOME/Library/Logs"
LABEL="com.researchwiki.vault-ui"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
mkdir -p "$RUN_DIR"

BACKEND_PORT="${VAULT_UI_BACKEND_PORT:-8050}"
FRONTEND_PORT="${VAULT_UI_FRONTEND_PORT:-4180}"
PORT="${VAULT_UI_PORT:-4180}"  # unified production port (SPA + API)

# --- dev: two servers (Vite dev + backend), proxied ---
start_backend() {
  local log="$RUN_DIR/vault-ui-backend.log"
  echo "starting backend on :$BACKEND_PORT"
  (cd "$ROOT" && PYTHONPATH="$APP" nohup uv run uvicorn backend.main:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" > "$log" 2>&1 < /dev/null &
   echo $! > "$RUN_DIR/vault-ui-backend.pid"; disown "$!" 2>/dev/null || true)
}

start_frontend() {
  local log="$RUN_DIR/vault-ui-frontend.log"
  echo "starting frontend on :$FRONTEND_PORT"
  (cd "$APP/frontend" && VITE_API_PROXY_TARGET="http://127.0.0.1:$BACKEND_PORT" \
    nohup npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" > "$log" 2>&1 < /dev/null &
   echo $! > "$RUN_DIR/vault-ui-frontend.pid"; disown "$!" 2>/dev/null || true)
}

build_frontend() {
  echo "building frontend bundle"
  (cd "$APP/frontend" && npm run build)
}

# --- production: one uvicorn serving the built SPA + API (foreground, for launchd) ---
serve() {
  [ "${1:-}" = "--build" ] && build_frontend
  if [ ! -f "$DIST/index.html" ]; then
    echo "no build at $DIST — run: $0 serve --build" >&2
    exit 1
  fi
  echo "serving vault-ui (SPA + API) on http://127.0.0.1:$PORT"
  cd "$ROOT"
  exec env PYTHONPATH="$APP" VAULT_UI_STATIC_DIR="$DIST" PYTHONUNBUFFERED=1 \
    "$UV" run uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"
}

write_plist() {
  mkdir -p "$(dirname "$PLIST")" "$LOG_DIR"
  cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>$LABEL</string>
	<key>ProgramArguments</key>
	<array>
		<string>/bin/bash</string>
		<string>$APP/vault-ui.sh</string>
		<string>serve</string>
	</array>
	<key>EnvironmentVariables</key>
	<dict>
		<key>VAULT_UI_PORT</key>
		<string>$PORT</string>
		<key>PYTHONUNBUFFERED</key>
		<string>1</string>
		<key>PATH</key>
		<string>$(dirname "$UV"):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
	</dict>
	<key>WorkingDirectory</key>
	<string>$ROOT</string>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<true/>
	<key>StandardOutPath</key>
	<string>$LOG_DIR/vault-ui.log</string>
	<key>StandardErrorPath</key>
	<string>$LOG_DIR/vault-ui.log</string>
</dict>
</plist>
PLISTEOF
  echo "wrote $PLIST"
}

case "${1:-help}" in
  start)
    start_backend
    start_frontend
    echo "open: http://localhost:$FRONTEND_PORT"
    ;;
  stop)
    for name in backend frontend; do
      pid_file="$RUN_DIR/vault-ui-$name.pid"
      [ -f "$pid_file" ] && kill "$(cat "$pid_file")" 2>/dev/null || true
      rm -f "$pid_file"
    done
    pkill -f "uvicorn backend.main:app" 2>/dev/null || true
    pkill -f "vault-ui/frontend.*vite" 2>/dev/null || true
    ;;
  serve)
    serve "${2:-}"
    ;;
  install)
    build_frontend
    write_plist
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "loaded $LABEL — open http://localhost:$PORT"
    ;;
  uninstall)
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "unloaded + removed $LABEL"
    ;;
  restart)
    # Pick up new backend code (uvicorn does not hot-reload in production serve).
    build_frontend
    launchctl kickstart -k "gui/$(id -u)/$LABEL" && echo "restarted $LABEL" \
      || echo "restart failed — is it installed? ($0 install)"
    ;;
  status)
    launchctl list 2>/dev/null | grep "$LABEL" || echo "$LABEL not loaded"
    curl -s "http://127.0.0.1:$PORT/api/health" && echo || echo "(no response on :$PORT)"
    ;;
  *)
    echo "usage: $0 {start|stop|serve [--build]|install|uninstall|restart|status}"
    exit 1
    ;;
esac
