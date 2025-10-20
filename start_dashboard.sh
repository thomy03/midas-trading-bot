#!/bin/bash
# Linux/Mac shell script to start the Market Screener Dashboard

echo "========================================"
echo "Market Screener Dashboard"
echo "========================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
fi

echo ""
echo "Starting dashboard..."
echo ""
echo "The dashboard will open in your browser at:"
echo "http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run dashboard.py
