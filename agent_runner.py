"""
Vision-first test runner using LangGraph.
Run with `python agent_runner.py` to execute test scenarios.
"""
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any, Optional

from test_parser import TestParser
from test_graph import TestExecutor
from test_state import TestState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_test(scenario: str, headless: bool = True, report_path: Optional[str] = None) -> None:
    """
    Run a test from a natural language scenario.

    Args:
        scenario: Natural language test scenario
        headless: Whether to run browser in headless mode
        report_path: Path to save test report
    """
    # Parse test
    parser = TestParser()
    test_state = parser.parse_test(scenario)

    # Print parsed actions
    logger.info(f"Parsed {len(test_state.actions)} actions from scenario:")
    for i, action in enumerate(test_state.actions):
        logger.info(f"  {i+1}. {action.type}: {action.description}")

    # Execute test
    vision_model = os.environ.get("VISION_MODEL", "gpt-4o")
    executor = TestExecutor(vision_model=vision_model, headless=headless)

    # Run test and track progress
    final_state = None
    async for state in executor.execute_test(test_state):
        final_state = state
        progress = f"{len(state.results)}/{len(state.actions)}"
        logger.info(
            f"Progress: {progress} - Current action: {state.current_action.description if state.current_action else 'N/A'}")

    # Show results
    if final_state:
        status = "✅ PASSED" if final_state.status == "passed" else f"❌ FAILED: {final_state.error}"
        print(status)

        # Save report if requested
        if report_path and final_state:
            report = final_state.as_report()
            os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Test report saved to {report_path}")
    else:
        print("❌ FAILED: Test execution error")


if __name__ == "__main__":
    # Default test scenario
    DEFAULT_SCENARIO = """
    Open http://localhost:8000.
    Log in with username user and password pass.
    Click the Advanced Search button.
    Search for party id iag00001.
    Verify the results table contains party id iag00001, first name test user1, last name test, dob 1991-10-10.
    """

    # Get scenario from command line if provided
    if len(sys.argv) > 1:
        scenario = " ".join(sys.argv[1:])
    else:
        scenario = DEFAULT_SCENARIO
        logger.info("No scenario provided, using default test scenario")

    # Set headless mode from environment
    headless = os.environ.get("PW_HEADLESS", "true").lower() != "false"

    # Run test
    asyncio.run(run_test(scenario, headless=headless,
                report_path="test_report.json"))
