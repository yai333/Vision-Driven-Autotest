#!/bin/bash

# Start a local web server to serve the index.html file
echo "Starting local web server on port 8000..."
python -m http.server 8000 &
SERVER_PID=$!

# Wait for the server to start
sleep 2

echo "Server started with PID: $SERVER_PID"

# Set environment variables
export OPENAI_API_KEY="your_openai_api_key_here"  # Replace with your actual API key
export VISION_MODEL="gpt-4o"  # Specify the OpenAI vision model to use

# Optional: Set non-headless mode to see the browser
export PW_HEADLESS="false"

# Run the test using the test_scenario.txt file
echo "Running test scenario..."
python main.py --scenario-file test_scenario.txt --html-report test_report.html

# Capture the exit code
EXIT_CODE=$?

# Stop the local web server
echo "Stopping local web server..."
kill $SERVER_PID

# Show the test result
if [ $EXIT_CODE -eq 0 ]; then
    echo "Test completed successfully!"
    echo "HTML report generated at: test_report.html"
else
    echo "Test failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE