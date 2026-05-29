#!/usr/bin/env bash
# Uzak server yönetimi — Tailscale üzerinden
# Kullanım: ./scripts/server.sh [up|down|ps|logs|deploy|restart]

SERVER="100.98.163.58"
REPO="~/CognitiveWeave"
REMOTE="ssh $SERVER"

_server() {
  ssh "$SERVER" "
    export DOCKER_HOST=unix:///Users/suderemote/.colima/default/docker.sock
    export PATH=/opt/homebrew/bin:\$PATH
    $1
  "
}

case "${1:-ps}" in
  up)
    echo "▶ Container'lar başlatılıyor..."
    _server "cd $REPO && docker-compose up -d"
    ;;
  down)
    echo "⏹ Container'lar durduruluyor..."
    _server "cd $REPO && docker-compose down"
    ;;
  restart)
    echo "🔄 Deploy + restart..."
    ssh "$SERVER" "cd $REPO && git pull"
    _server "cd $REPO && docker-compose up -d"
    ;;
  deploy)
    echo "📦 Server güncelleniyor..."
    ssh "$SERVER" "cd $REPO && git pull"
    ;;
  ps)
    _server "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
    ;;
  logs)
    _server "cd $REPO && docker-compose logs --tail=80"
    ;;
  *)
    echo "Kullanım: $0 [up|down|restart|deploy|ps|logs]"
    exit 1
    ;;
esac
