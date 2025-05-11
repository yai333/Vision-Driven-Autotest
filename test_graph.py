"""
Core LangGraph implementation for vision autotest.
"""
import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, TypeVar, AsyncGenerator

import langgraph.graph as lg
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from test_state import (
    TestState, Action, ActionResult,
    VisitAction, ClickAction, FillAction, ScrollAction,
    AssertVisibleAction, AssertTextAction, AssertRowAction,
)
from vision_service import VisionService, VisionServiceError
from browser_interface import BrowserInterface, BrowserInterfaceError

logger = logging.getLogger(__name__)

# Type variable for state
StateType = TypeVar("StateType", bound=Dict[str, Any])


class TestExecutor:
    """
    Test executor using LangGraph for workflow management.
    """

    def __init__(self, vision_model: Optional[str] = None, headless: bool = True):
        """
        Initialize the test executor.

        Args:
            vision_model: Name of vision model to use
            headless: Whether to run browser in headless mode
        """
        self.vision_model = vision_model
        self.headless = headless
        self.vision_service = None
        self.browser_interface = None
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for test execution.

        Returns:
            StateGraph: The constructed graph
        """
        # Define the graph
        builder = StateGraph(TestState)

        # Define the nodes
        builder.add_node("initialize", self._initialize)
        builder.add_node("execute_visit", self._execute_visit)
        builder.add_node("execute_click", self._execute_click)
        builder.add_node("execute_fill", self._execute_fill)
        builder.add_node("execute_scroll", self._execute_scroll)
        builder.add_node("execute_assert_visible",
                         self._execute_assert_visible)
        builder.add_node("execute_assert_text", self._execute_assert_text)
        builder.add_node("execute_assert_row", self._execute_assert_row)
        builder.add_node("route_next_action", self._route_next_action)
        builder.add_node("finalize", self._finalize)

        # Define the edges (workflow)
        builder.set_entry_point("initialize")

        # After initialization, route to the appropriate action
        builder.add_edge("initialize", "route_next_action")

        # From each action execution, route to the next action
        builder.add_edge("execute_visit", "route_next_action")
        builder.add_edge("execute_click", "route_next_action")
        builder.add_edge("execute_fill", "route_next_action")
        builder.add_edge("execute_scroll", "route_next_action")
        builder.add_edge("execute_assert_visible", "route_next_action")
        builder.add_edge("execute_assert_text", "route_next_action")
        builder.add_edge("execute_assert_row", "route_next_action")

        # Define conditional routing based on the next action type
        builder.add_conditional_edges(
            "route_next_action",
            self._route_condition,
            {
                "visit": "execute_visit",
                "click": "execute_click",
                "fill": "execute_fill",
                "scroll": "execute_scroll",
                "assert_visible": "execute_assert_visible",
                "assert_text": "execute_assert_text",
                "assert_row": "execute_assert_row",
                "complete": "finalize",
                "error": "finalize",
            }
        )

        # Finalize is an end state
        builder.add_edge("finalize", END)

        # Compile and return the graph
        return builder.compile()

    def _route_condition(self, state: TestState) -> str:
        """
        Determine the next action based on current state.

        Args:
            state: Current test state

        Returns:
            String indicating which node to route to next
        """
        # If there was an error in the last action, go to finalize
        if state.has_errors:
            return "error"

        # If we've completed all actions, go to finalize
        if state.is_complete:
            return "complete"

        # Otherwise, route based on the next action's type
        state.advance()  # Move to next action
        current_action = state.current_action

        if current_action is None:
            return "error"

        return current_action.type

    async def _initialize(self, state: TestState) -> TestState:
        """
        Initialize test resources.

        Args:
            state: Current test state

        Returns:
            Updated test state
        """
        logger.info(f"Initializing test: {state.name}")

        # Update test status
        state.status = "running"

        # Initialize vision service if not already done
        if self.vision_service is None:
            self.vision_service = VisionService(model_name=self.vision_model)

        # Initialize browser interface if not already done
        if self.browser_interface is None:
            self.browser_interface = BrowserInterface(
                vision_service=self.vision_service,
                headless=self.headless
            )

        return state

    async def _execute_visit(self, state: TestState) -> TestState:
        """
        Execute a visit action.

        Args:
            state: Current test state

        Returns:
            Updated test state
        """
        action = state.current_action
        if not isinstance(action, VisitAction):
            # This shouldn't happen due to routing
            result = ActionResult(
                success=False,
                message="Invalid action type for execute_visit",
                error="Type mismatch in action routing"
            )
            state.add_result(result)
            return state

        logger.info(f"Executing visit action: {action.url}")

        try:
            async with self.browser_interface.browser_context() as browser:
                title = await browser.visit(action.url)
                screenshot_path = await browser.save_screenshot(f"visit_{action.url.replace('://', '_')}")

                result = ActionResult(
                    success=True,
                    message=f"Successfully visited {action.url}. Page title: {title}",
                    screenshot_path=screenshot_path
                )
        except BrowserInterfaceError as e:
            logger.error(f"Browser error during visit: {e}")
            result = ActionResult(
                success=False,
                message=f"Failed to visit {action.url}",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during visit: {e}")
            result = ActionResult(
                success=False,
                message=f"Unexpected error visiting {action.url}",
                error=str(e)
            )

        state.add_result(result)
        return state

    async def _execute_click(self, state: TestState) -> TestState:
        """
        Execute a click action.

        Args:
            state: Current test state

        Returns:
            Updated test state
        """
        action = state.current_action
        if not isinstance(action, ClickAction):
            # This shouldn't happen due to routing
            result = ActionResult(
                success=False,
                message="Invalid action type for execute_click",
                error="Type mismatch in action routing"
            )
            state.add_result(result)
            return state

        logger.info(f"Executing click action: {action.element_description}")

        try:
            async with self.browser_interface.browser_context() as browser:
                click_result = await browser.vision_click(action.element_description, retry_count=2)
                screenshot_path = await browser.save_screenshot(f"click_{action.element_description.replace(' ', '_')}")

                result = ActionResult(
                    success=True,
                    message=f"Successfully clicked on {action.element_description}. Method: {click_result}",
                    screenshot_path=screenshot_path
                )
        except BrowserInterfaceError as e:
            logger.error(f"Browser error during click: {e}")
            result = ActionResult(
                success=False,
                message=f"Failed to click on {action.element_description}",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during click: {e}")
            result = ActionResult(
                success=False,
                message=f"Unexpected error clicking on {action.element_description}",
                error=str(e)
            )

        state.add_result(result)
        return state

    async def _execute_fill(self, state: TestState) -> TestState:
        """
        Execute a fill action.

        Args:
            state: Current test state

        Returns:
            Updated test state
        """
        action = state.current_action
        if not isinstance(action, FillAction):
            # This shouldn't happen due to routing
            result = ActionResult(
                success=False,
                message="Invalid action type for execute_fill",
                error="Type mismatch in action routing"
            )
            state.add_result(result)
            return state

        logger.info(
            f"Executing fill action: {action.element_description} with text '{action.text}'")

        try:
            async with self.browser_interface.browser_context() as browser:
                fill_result = await browser.vision_fill(action.element_description, action.text, retry_count=2)
                screenshot_path = await browser.save_screenshot(f"fill_{action.element_description.replace(' ', '_')}")

                result = ActionResult(
                    success=True,
                    message=f"Successfully filled {action.element_description} with '{action.text}'. Method: {fill_result}",
                    screenshot_path=screenshot_path
                )
        except BrowserInterfaceError as e:
            logger.error(f"Browser error during fill: {e}")
            result = ActionResult(
                success=False,
                message=f"Failed to fill {action.element_description} with '{action.text}'",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during fill: {e}")
            result = ActionResult(
                success=False,
                message=f"Unexpected error filling {action.element_description} with '{action.text}'",
                error=str(e)
            )

        state.add_result(result)
        return state

    async def _execute_scroll(self, state: TestState) -> TestState:
        """
        Execute a scroll action.

        Args:
            state: Current test state

        Returns:
            Updated test state
        """
        action = state.current_action
        if not isinstance(action, ScrollAction):
            # This shouldn't happen due to routing
            result = ActionResult(
                success=False,
                message="Invalid action type for execute_scroll",
                error="Type mismatch in action routing"
            )
            state.add_result(result)
            return state

        logger.info(f"Executing scroll action: {action.element_description}")

        try:
            async with self.browser_interface.browser_context() as browser:
                scroll_result = await browser.vision_scroll_into_view(action.element_description, retry_count=2)
                screenshot_path = await browser.save_screenshot(f"scroll_{action.element_description.replace(' ', '_')}")

                result = ActionResult(
                    success=True,
                    message=f"Successfully scrolled to {action.element_description}. Method: {scroll_result}",
                    screenshot_path=screenshot_path
                )
        except BrowserInterfaceError as e:
            logger.error(f"Browser error during scroll: {e}")
            result = ActionResult(
                success=False,
                message=f"Failed to scroll to {action.element_description}",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during scroll: {e}")
            result = ActionResult(
                success=False,
                message=f"Unexpected error scrolling to {action.element_description}",
                error=str(e)
            )

        state.add_result(result)
        return state

    async def _execute_assert_visible(self, state: TestState) -> TestState:
        """
        Execute an assert_visible action.

        Args:
            state: Current test state

        Returns:
            Updated test state
        """
        action = state.current_action
        if not isinstance(action, AssertVisibleAction):
            # This shouldn't happen due to routing
            result = ActionResult(
                success=False,
                message="Invalid action type for execute_assert_visible",
                error="Type mismatch in action routing"
            )
            state.add_result(result)
            return state

        logger.info(
            f"Executing assert_visible action: {action.element_description}")

        try:
            async with self.browser_interface.browser_context() as browser:
                assert_result = await browser.vision_assert_visible(action.element_description)
                screenshot_path = await browser.save_screenshot(f"assert_visible_{action.element_description.replace(' ', '_')}")

                result = ActionResult(
                    success=True,
                    message=f"Successfully verified {action.element_description} is visible",
                    screenshot_path=screenshot_path
                )
        except BrowserInterfaceError as e:
            logger.error(f"Browser error during assert_visible: {e}")
            result = ActionResult(
                success=False,
                message=f"Failed to verify {action.element_description} is visible",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during assert_visible: {e}")
            result = ActionResult(
                success=False,
                message=f"Unexpected error verifying {action.element_description} is visible",
                error=str(e)
            )

        state.add_result(result)
        return state

    async def _execute_assert_text(self, state: TestState) -> TestState:
        """
        Execute an assert_text action.

        Args:
            state: Current test state

        Returns:
            Updated test state
        """
        action = state.current_action
        if not isinstance(action, AssertTextAction):
            # This shouldn't happen due to routing
            result = ActionResult(
                success=False,
                message="Invalid action type for execute_assert_text",
                error="Type mismatch in action routing"
            )
            state.add_result(result)
            return state

        logger.info(
            f"Executing assert_text action: {action.element_description} contains '{action.expected_text}'")

        try:
            async with self.browser_interface.browser_context() as browser:
                assert_result = await browser.vision_assert_text(action.element_description, action.expected_text)
                screenshot_path = await browser.save_screenshot(f"assert_text_{action.element_description.replace(' ', '_')}")

                result = ActionResult(
                    success=True,
                    message=f"Successfully verified {action.element_description} contains '{action.expected_text}'",
                    screenshot_path=screenshot_path
                )
        except BrowserInterfaceError as e:
            logger.error(f"Browser error during assert_text: {e}")
            result = ActionResult(
                success=False,
                message=f"Failed to verify {action.element_description} contains '{action.expected_text}'",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during assert_text: {e}")
            result = ActionResult(
                success=False,
                message=f"Unexpected error verifying {action.element_description} contains '{action.expected_text}'",
                error=str(e)
            )

        state.add_result(result)
        return state

    async def _execute_assert_row(self, state: TestState) -> TestState:
        """
        Execute an assert_row action.

        Args:
            state: Current test state

        Returns:
            Updated test state
        """
        action = state.current_action
        if not isinstance(action, AssertRowAction):
            # This shouldn't happen due to routing
            result = ActionResult(
                success=False,
                message="Invalid action type for execute_assert_row",
                error="Type mismatch in action routing"
            )
            state.add_result(result)
            return state

        logger.info(f"Executing assert_row action: {action.expected_data}")

        try:
            async with self.browser_interface.browser_context() as browser:
                assert_result = await browser.vision_expect_row(action.expected_data)
                screenshot_path = await browser.save_screenshot("assert_row")

                result = ActionResult(
                    success=True,
                    message=f"Successfully verified table row contains expected data",
                    screenshot_path=screenshot_path
                )
        except BrowserInterfaceError as e:
            logger.error(f"Browser error during assert_row: {e}")
            result = ActionResult(
                success=False,
                message=f"Failed to verify table row contains expected data",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during assert_row: {e}")
            result = ActionResult(
                success=False,
                message=f"Unexpected error verifying table row contains expected data",
                error=str(e)
            )

        state.add_result(result)
        return state

    async def _route_next_action(self, state: TestState) -> TestState:
        """
        Route to the next action based on current state.

        Args:
            state: Current test state

        Returns:
            Updated test state
        """
        # This is just a routing node, no actual logic
        # The _route_condition function handles the routing decision
        return state

    async def _finalize(self, state: TestState) -> TestState:
        """
        Finalize the test and clean up resources.

        Args:
            state: Current test state

        Returns:
            Updated test state
        """
        logger.info(f"Finalizing test: {state.name}")

        # Update test status if not already set
        if not state.has_errors and state.status == "running":
            state.status = "passed"

        # Generate final report
        logger.info(f"Test {state.name} {state.status}")
        if state.error:
            logger.error(f"Test error: {state.error}")

        return state

    async def execute_test(self, test_state: TestState) -> AsyncGenerator[TestState, None]:
        """
        Execute a test and yield intermediate states.

        Args:
            test_state: Initial test state

        Yields:
            Updated test states during execution
        """
        saver = MemorySaver()

        async for state in self.graph.astream(
            test_state,
            checkpoint_saver=saver,
        ):
            yield state

    async def run_test(self, test_state: TestState) -> TestState:
        """
        Run a test and return the final state.

        Args:
            test_state: Initial test state

        Returns:
            Final test state after execution
        """
        final_state = None

        async for state in self.execute_test(test_state):
            final_state = state

        return final_state
