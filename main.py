"""
Main entry point for running vision autotests.
"""
import os
import sys
import asyncio
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from test_parser import TestParser
from test_graph import TestExecutor
from test_state import TestState


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("autotest.log")
    ]
)
logger = logging.getLogger(__name__)


async def run_test_from_scenario(
    scenario: str,
    vision_model: Optional[str] = None,
    headless: bool = True,
    report_path: Optional[str] = None
) -> TestState:
    """
    Run a test from a natural language scenario.
    
    Args:
        scenario: Natural language test scenario
        vision_model: Name of vision model to use
        headless: Whether to run browser in headless mode
        report_path: Path to save test report
        
    Returns:
        Final test state
    """
    # Parse the scenario into a test state
    parser = TestParser()
    test_state = parser.parse_test(scenario)
    
    # Print the parsed actions
    logger.info(f"Parsed {len(test_state.actions)} actions from scenario:")
    for i, action in enumerate(test_state.actions):
        logger.info(f"  {i+1}. {action.type}: {action.description}")
    
    # Execute the test
    executor = TestExecutor(vision_model=vision_model, headless=headless)
    final_state = await executor.run_test(test_state)
    
    # Generate and save report if requested
    if report_path:
        report = final_state.as_report()
        
        # Ensure the directory exists
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        
        # Save the report
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to {report_path}")
    
    return final_state


async def generate_report_html(test_state: TestState, output_path: str) -> None:
    """
    Generate an HTML report from a test state.
    
    Args:
        test_state: Test state to generate report from
        output_path: Path to save HTML report
    """
    # Get the test report data
    report = test_state.as_report()
    
    # Convert screenshot paths to public URLs or data URIs if needed
    # Here we just keep the paths, assuming they're accessible from where the HTML will be served
    
    # Generate the HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Test Report: {report["name"]}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .status-passed {{ color: green; }}
            .status-failed {{ color: red; }}
            .step {{ margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
            .step-header {{ display: flex; justify-content: space-between; align-items: center; }}
            .step-success {{ color: green; }}
            .step-failure {{ color: red; }}
            .step-details {{ margin-top: 10px; }}
            .screenshot {{ max-width: 100%; border: 1px solid #ccc; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>Test Report: {report["name"]}</h1>
        <p><strong>Status:</strong> <span class="status-{report['status']}">{report['status'].upper()}</span></p>
        <p><strong>Progress:</strong> {report['progress']}</p>
        <p><strong>Description:</strong> {report['description']}</p>
        
        {"<p class='status-failed'><strong>Error:</strong> " + report.get('error', '') + "</p>" if report.get('error') else ""}
        
        <h2>Steps</h2>
        <div class="steps">
    """
    
    # Add steps to the HTML
    for step in report["steps"]:
        result = step.get("result", "pending")
        result_class = ""
        result_text = ""
        
        if result == "pending":
            result_class = "step-pending"
            result_text = "Pending"
        elif isinstance(result, dict):
            if result.get("success", False):
                result_class = "step-success"
                result_text = "Success"
            else:
                result_class = "step-failure"
                result_text = "Failure"
        
        html += f"""
        <div class="step">
            <div class="step-header">
                <h3>Step {step['index'] + 1}: {step['type']}</h3>
                <span class="{result_class}">{result_text}</span>
            </div>
            <div class="step-details">
                <p><strong>Description:</strong> {step['description']}</p>
        """
        
        # Add type-specific details
        if step["type"] == "visit":
            html += f"<p><strong>URL:</strong> {step.get('url', '')}</p>"
        elif step["type"] in ["click", "fill", "scroll", "assert_visible", "assert_text"]:
            html += f"<p><strong>Element:</strong> {step.get('element_description', '')}</p>"
            if step["type"] == "fill":
                html += f"<p><strong>Text:</strong> {step.get('text', '')}</p>"
            elif step["type"] == "assert_text":
                html += f"<p><strong>Expected Text:</strong> {step.get('expected_text', '')}</p>"
        elif step["type"] == "assert_row":
            html += f"<p><strong>Expected Data:</strong> {json.dumps(step.get('expected_data', {}), indent=2)}</p>"
        
        # Add result details if available
        if isinstance(result, dict):
            html += f"<p><strong>Message:</strong> {result.get('message', '')}</p>"
            if result.get('error'):
                html += f"<p><strong>Error:</strong> <span class='step-failure'>{result.get('error', '')}</span></p>"
            if result.get('screenshot'):
                html += f"<img class='screenshot' src='{result.get('screenshot')}' alt='Screenshot' />"
        
        html += """
            </div>
        </div>
        """
    
    # Close the HTML
    html += """
        </div>
    </body>
    </html>
    """
    
    # Save the HTML
    with open(output_path, "w") as f:
        f.write(html)
    
    logger.info(f"HTML report saved to {output_path}")


async def main() -> None:
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description="Run vision-based autotest")
    parser.add_argument("--scenario", "-s", type=str, help="Test scenario as natural language")
    parser.add_argument("--scenario-file", "-f", type=str, help="File containing test scenario")
    parser.add_argument("--vision-model", "-m", type=str, help="Vision model to use")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--report", "-r", type=str, help="Path to save test report (JSON)")
    parser.add_argument("--html-report", type=str, help="Path to save HTML test report")
    
    args = parser.parse_args()
    
    # Get the scenario
    scenario = None
    if args.scenario:
        scenario = args.scenario
    elif args.scenario_file:
        with open(args.scenario_file, "r") as f:
            scenario = f.read()
    else:
        scenario = """
        Open http://localhost:80.
        Log in with username user and password pass.
        Click the Advanced Search button.
        Search for party id iag00001.
        Verify the results table contains party id iag00001, first name test user1, last name test, dob 1991-10-10.
        """
        logger.info("No scenario provided, using default test scenario")
    
    # Configure vision model
    vision_model = args.vision_model or os.environ.get("VISION_MODEL")
    
    # Run the test
    final_state = await run_test_from_scenario(
        scenario=scenario,
        vision_model=vision_model,
        headless=args.headless,
        report_path=args.report
    )
    
    # Generate HTML report if requested
    if args.html_report:
        await generate_report_html(final_state, args.html_report)
    
    # Log final status
    logger.info(f"Test completed with status: {final_state.status}")
    if final_state.has_errors:
        logger.error(f"Test failed: {final_state.error}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())