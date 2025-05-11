"""
State management for vision autotest LangGraph.
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field


class ActionResult(BaseModel):
    """Result of a single action in the test."""
    success: bool = True
    message: str = ""
    screenshot_path: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_error(self) -> bool:
        """Check if this result represents an error."""
        return not self.success or self.error is not None


class Action(BaseModel):
    """Base class for all test actions."""
    type: str
    description: str = ""


class VisitAction(Action):
    """Action to visit a URL."""
    type: str = "visit"
    url: str


class ClickAction(Action):
    """Action to click on an element."""
    type: str = "click"
    element_description: str


class FillAction(Action):
    """Action to fill a form field."""
    type: str = "fill"
    element_description: str
    text: str


class ScrollAction(Action):
    """Action to scroll to an element."""
    type: str = "scroll"
    element_description: str


class AssertVisibleAction(Action):
    """Action to assert an element is visible."""
    type: str = "assert_visible"
    element_description: str


class AssertTextAction(Action):
    """Action to assert an element contains text."""
    type: str = "assert_text"
    element_description: str
    expected_text: str


class AssertRowAction(Action):
    """Action to assert a table row exists with data."""
    type: str = "assert_row"
    expected_data: Dict[str, Any]


class TestState(BaseModel):
    """State of a running test."""
    test_id: str = Field(default_factory=lambda: f"test_{id(object())}")
    name: str = ""
    description: str = ""
    actions: List[Action] = Field(default_factory=list)
    current_action_index: int = -1
    results: List[ActionResult] = Field(default_factory=list)
    status: str = "pending"  # pending, running, passed, failed
    error: Optional[str] = None
    extra_data: Dict[str, Any] = Field(default_factory=dict)

    @property
    def current_action(self) -> Optional[Action]:
        """Get the current action to execute."""
        if 0 <= self.current_action_index < len(self.actions):
            return self.actions[self.current_action_index]
        return None

    @property
    def next_action(self) -> Optional[Action]:
        """Get the next action to execute."""
        next_index = self.current_action_index + 1
        if 0 <= next_index < len(self.actions):
            return self.actions[next_index]
        return None

    @property
    def is_complete(self) -> bool:
        """Check if all actions have been executed."""
        return self.current_action_index >= len(self.actions) - 1

    @property
    def has_errors(self) -> bool:
        """Check if any actions have failed."""
        return any(result.is_error for result in self.results)

    def advance(self) -> None:
        """Advance to the next action."""
        self.current_action_index += 1

    def add_result(self, result: ActionResult) -> None:
        """Add a result for the current action."""
        self.results.append(result)

        # Update test status based on result
        if result.is_error:
            self.status = "failed"
            self.error = result.error or result.message

    def as_report(self) -> Dict[str, Any]:
        """Generate a report of the test execution."""
        passed_steps = sum(1 for r in self.results if not r.is_error)
        total_steps = len(self.actions)
        executed_steps = len(self.results)

        report = {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "progress": f"{executed_steps}/{total_steps}",
            "passed_steps": passed_steps,
            "executed_steps": executed_steps,
            "total_steps": total_steps,
            "steps": []
        }

        for i, action in enumerate(self.actions):
            step = {
                "index": i,
                "type": action.type,
                "description": action.description,
            }

            # Add type-specific fields
            if isinstance(action, VisitAction):
                step["url"] = action.url
            elif isinstance(action, ClickAction):
                step["element_description"] = action.element_description
            elif isinstance(action, FillAction):
                step["element_description"] = action.element_description
                step["text"] = action.text
            elif isinstance(action, ScrollAction):
                step["element_description"] = action.element_description
            elif isinstance(action, AssertVisibleAction):
                step["element_description"] = action.element_description
            elif isinstance(action, AssertTextAction):
                step["element_description"] = action.element_description
                step["expected_text"] = action.expected_text
            elif isinstance(action, AssertRowAction):
                step["expected_data"] = action.expected_data

            # Add result if available
            if i < len(self.results):
                result = self.results[i]
                step["result"] = {
                    "success": result.success,
                    "message": result.message,
                    "screenshot": result.screenshot_path,
                    "error": result.error
                }
            else:
                step["result"] = "pending"

            report["steps"].append(step)

        if self.error:
            report["error"] = self.error

        return report
