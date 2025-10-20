#!/bin/bash
# Linux/Mac shell script to run the market screener

echo "========================================"
echo "Market Screener"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "========================================"
echo "Choose an option:"
echo "========================================"
echo "1. Run screening once"
echo "2. Start scheduler (auto mode)"
echo "3. Test notifications"
echo "4. Screen a specific symbol"
echo "5. View recent alerts"
echo "6. Run tests"
echo "7. Launch Dashboard (Web UI)"
echo "========================================"
echo ""

read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo "Running screening..."
        python main.py run
        ;;
    2)
        echo "Starting scheduler..."
        echo "Press Ctrl+C to stop"
        python main.py schedule
        ;;
    3)
        echo "Testing notifications..."
        python main.py test
        ;;
    4)
        read -p "Enter stock symbol (e.g., AAPL): " symbol
        echo "Screening $symbol..."
        python main.py screen --symbol "$symbol"
        ;;
    5)
        read -p "Number of days to look back (default 7): " days
        days=${days:-7}
        python main.py alerts --days "$days"
        ;;
    6)
        echo "Running tests..."
        python tests/test_basic.py
        ;;
    7)
        echo "Starting Dashboard..."
        echo "Opening browser at http://localhost:8501"
        streamlit run dashboard.py
        ;;
    *)
        echo "Invalid choice!"
        ;;
esac

echo ""
