# Vision-Driven Autotest with LangGraph

A powerful vision-based automated testing framework using LangGraph for workflow management, Playwright for browser automation, and state-of-the-art vision LLMs (Gemini, GPT-4o) for visual element recognition.

## Architecture

This project reimplements the original vision-driven autotest concept using LangGraph, providing:

- Structured workflow management with a directed state graph
- Clear separation of concerns with modular components
- Improved error handling and recovery
- Built-in retry mechanisms
- Comprehensive test reporting
- Enhanced observability through state tracking

## Features

- **Natural Language Test Scenarios**: Write tests in plain English that get translated into test steps
- **Vision-First Interaction**: Uses vision LLMs to locate elements on the page based on natural language descriptions
- **Selector Fallback**: Automatically falls back to DOM selectors for more robust element location
- **Comprehensive Reporting**: Detailed reports with screenshots and status for each step
- **LangGraph Workflow**: Structured directed graph for test execution flow
- **Multi-cloud Support**: Works with either Gemini or OpenAI (GPT-4o) vision models

## Components

- `vision_service.py`: Encapsulates vision LLM interactions with support for multiple providers
- `browser_interface.py`: Provides Playwright-based browser automation with vision-enhanced tools
- `test_state.py`: Defines the state models for test execution and action types
- `test_parser.py`: Converts natural language test scenarios into structured test steps
- `test_graph.py`: Implements the LangGraph workflow for test execution
- `main.py`: Command-line entry point for running tests

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/username/vision-autotest-langgraph.git
cd vision-autotest-langgraph

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Playwright browsers
python -m playwright install --with-deps
```

## Configuration

Create a `.env` file with your API keys and configuration:

```
# Required for vision operations
GOOGLE_API_KEY=your_google_api_key  # For Gemini
# OR
OPENAI_API_KEY=your_openai_api_key  # For GPT-4o

# Optional configuration
VISION_MODEL=gemini-2.0-pro-vision  # Default vision model
PW_HEADLESS=false                   # Run browser in visible mode (default is true)
```

## Running Tests

```bash
# Run with a scenario from the command line
python main.py --scenario "Open http://example.com. Click the Sign In button. Fill the username field with 'testuser'. Fill the password field with 'password123'. Click the Login button."

# Run with a scenario from a file
python main.py --scenario-file my_test.txt

# Generate reports
python main.py --scenario-file my_test.txt --report reports/test_report.json --html-report reports/test_report.html

# Run in visible browser mode
python main.py --scenario-file my_test.txt --headless=false
```

## Test Scenario Format

Write test scenarios in natural language, describing the steps to take:

```
Open https://example.com.
Click the login button.
Fill the username field with "testuser".
Fill the password field with "testuser123".
Click the Sign In button.
Verify that the welcome message contains "Hello, Test User".
Scroll to the Recent Orders section.
Verify that the order list is visible.
```

## Example Test Report

```json
{
  "test_id": "test_123456",
  "name": "Login Test",
  "description": "Open https://example.com...",
  "status": "passed",
  "progress": "7/7",
  "passed_steps": 7,
  "executed_steps": 7,
  "total_steps": 7,
  "steps": [
    {
      "index": 0,
      "type": "visit",
      "description": "Visit https://example.com",
      "url": "https://example.com",
      "result": {
        "success": true,
        "message": "Successfully visited https://example.com. Page title: Example",
        "screenshot": "./screenshots/visit_https_example_com_1234567890.png",
        "error": null
      }
    },
    // Additional steps...
  ]
}
```

## Advantages Over Original Implementation

1. **Structured Workflow**: LangGraph provides a clear, directed graph for test flow
2. **Improved State Management**: Pydantic models for robust state tracking
3. **Better Error Handling**: Comprehensive error handling with recovery options
4. **Enhanced Observability**: Complete visibility into test execution state
5. **Retry Mechanisms**: Built-in retry logic for more resilient tests
6. **Stronger Typing**: Type annotations throughout for better code quality
7. **Proper Async Handling**: Async context managers for resource management
8. **Comprehensive Reporting**: Detailed JSON and HTML reports for each test run

## Extensibility

- Add new action types by creating new Action subclasses in `test_state.py` and handler nodes in `test_graph.py`
- Support additional LLM providers by extending the `VisionService` class
- Implement custom parsing logic in `test_parser.py` for domain-specific test scenarios
- Add new browser tools in `browser_interface.py` for specialized interactions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.