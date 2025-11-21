#!/bin/bash

echo "🌸 Starting Iris ML Model Platform..."
echo ""

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt --break-system-packages
fi

echo ""
echo "🚀 Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

python main.py
