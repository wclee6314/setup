#!/bin/bash

set -e

echo "🚀 n8n Docker Installation Script"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker."
    exit 1
fi

echo "✅ Docker is installed and running"

# Create n8n data directory
DATA_DIR="$HOME/.n8n"
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
    echo "📁 Created n8n data directory: $DATA_DIR"
fi

# Stop existing n8n container if running
if docker ps -a --filter "name=n8n" --format "table {{.Names}}" | grep -q "^n8n$"; then
    echo "🛑 Stopping existing n8n container..."
    docker stop n8n || true
    docker rm n8n || true
fi

# Pull latest n8n image
echo "⬇️  Pulling latest n8n Docker image..."
docker pull n8nio/n8n

# Run n8n container
echo "🐳 Starting n8n container..."
docker run -d \
    --name n8n \
    --restart unless-stopped \
    -p 5678:5678 \
    -v ~/.n8n:/home/node/.n8n \
    -e N8N_BASIC_AUTH_ACTIVE=true \
    -e N8N_BASIC_AUTH_USER=admin \
    -e N8N_BASIC_AUTH_PASSWORD=changeme123! \
    -e GENERIC_TIMEZONE=Asia/Seoul \
    n8nio/n8n

# Wait for container to start
echo "⏳ Waiting for n8n to start..."
sleep 10

# Check if container is running
if docker ps --filter "name=n8n" --filter "status=running" | grep -q n8n; then
    echo "✅ n8n is now running!"
    echo ""
    echo "📋 Connection Information:"
    echo "   URL: http://localhost:5678"
    echo "   Username: admin"
    echo "   Password: changeme123!"
    echo ""
    echo "⚠️  Security Notice:"
    echo "   Please change the default password after first login!"
    echo "   You can update credentials by stopping the container and"
    echo "   running with different N8N_BASIC_AUTH_USER and N8N_BASIC_AUTH_PASSWORD values."
    echo ""
    echo "🔧 Useful Commands:"
    echo "   View logs: docker logs n8n"
    echo "   Stop n8n: docker stop n8n"
    echo "   Start n8n: docker start n8n"
    echo "   Remove n8n: docker stop n8n && docker rm n8n"
    echo ""
    echo "📚 Documentation: https://docs.n8n.io/"
else
    echo "❌ Failed to start n8n container"
    echo "Check logs with: docker logs n8n"
    exit 1
fi