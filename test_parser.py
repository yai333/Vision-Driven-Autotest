"""
Test parser module for converting natural language test scenarios to structured actions.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
import langchain_core.prompts as prompts
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from test_state import (
    Action, VisitAction, ClickAction, FillAction, ScrollAction,
    AssertVisibleAction, AssertTextAction, AssertRowAction, TestState
)

logger = logging.getLogger(__name__)


class TestParser:
    """
    Parser for converting natural language test scenarios to structured actions.
    Uses LangChain and LLM to parse test descriptions.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the test parser.

        Args:
            model_name: Name of the LLM model to use for parsing
        """
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.action_parser = PydanticOutputParser(pydantic_object=TestState)

    def parse_test(self, scenario: str) -> TestState:
        """
        Parse a natural language test scenario into a TestState object.

        Args:
            scenario: Natural language description of the test

        Returns:
            TestState object with parsed actions
        """
        # First try rule-based parsing for common patterns
        test_state = self._rule_based_parse(scenario)
        if test_state and test_state.actions:
            logger.info(
                f"Rule-based parsing extracted {len(test_state.actions)} actions.")
            return test_state

        # Fall back to LLM-based parsing
        logger.info("Rule-based parsing insufficient, using LLM for parsing.")
        return self._llm_parse(scenario)

    def _rule_based_parse(self, scenario: str) -> Optional[TestState]:
        """
        Attempt to parse a test scenario using rule-based patterns.

        Args:
            scenario: Natural language description of the test

        Returns:
            TestState object with parsed actions, or None if parsing failed
        """
        actions: List[Action] = []

        # Create a new TestState
        test_state = TestState(name="Parsed Test", description=scenario)

        # Split into sentences
        sentences = re.split(r'[.;](?=(?:[^"]*"[^"]*")*[^"]*$)', scenario)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            # Try to identify action type from the sentence
            action = None

            # Visit URL
            url_match = re.search(
                r'(?:open|visit|navigate to|go to)\s+(?:the\s+)?(?:url\s+)?([^\s.]+)', sentence, re.IGNORECASE)
            if url_match:
                url = url_match.group(1).strip('"\'')
                # Add http:// prefix if missing
                if not (url.startswith('http://') or url.startswith('https://')):
                    url = 'http://' + url
                action = VisitAction(url=url, description=f"Visit {url}")

            # Click action
            elif re.search(r'(?:click|press|select|choose)', sentence, re.IGNORECASE):
                match = re.search(
                    r'(?:click|press|select|choose)(?:\s+on)?(?:\s+the)?\s+(.+?)(?:button|link|icon|element|$)', sentence, re.IGNORECASE)
                if match:
                    element_desc = match.group(1).strip()
                    action = ClickAction(
                        element_description=element_desc, description=f"Click on {element_desc}")

            # Fill action
            elif re.search(r'(?:fill|enter|type|input)', sentence, re.IGNORECASE):
                # Try to match "fill X with Y" pattern
                fill_match = re.search(
                    r'(?:fill|enter|type|input)(?:\s+in)?(?:\s+the)?\s+(.+?)\s+(?:with|as)\s+(?:the\s+)?(?:value\s+)?[\'\"]?([^\'"]+)[\'\"]?', sentence, re.IGNORECASE)
                if fill_match:
                    field_desc = fill_match.group(1).strip()
                    input_value = fill_match.group(2).strip()
                    action = FillAction(
                        element_description=field_desc,
                        text=input_value,
                        description=f"Fill {field_desc} with '{input_value}'"
                    )

            # Scroll action
            elif re.search(r'(?:scroll)', sentence, re.IGNORECASE):
                match = re.search(
                    r'scroll(?:\s+to)?(?:\s+the)?\s+(.+)', sentence, re.IGNORECASE)
                if match:
                    element_desc = match.group(1).strip()
                    action = ScrollAction(
                        element_description=element_desc,
                        description=f"Scroll to {element_desc}"
                    )

            # Assert visible action
            elif re.search(r'(?:verify|check|ensure|assert).*(?:visible|appears|displays|shown)', sentence, re.IGNORECASE):
                match = re.search(
                    r'(?:verify|check|ensure|assert)(?:\s+that)?(?:\s+the)?\s+(.+?)\s+(?:is\s+)?(?:visible|appears|displays|shown)', sentence, re.IGNORECASE)
                if match:
                    element_desc = match.group(1).strip()
                    action = AssertVisibleAction(
                        element_description=element_desc,
                        description=f"Verify {element_desc} is visible"
                    )

            # Assert text action
            elif re.search(r'(?:verify|check|ensure|assert).*(?:contains|has|shows|displays).*(?:text|value|content)', sentence, re.IGNORECASE):
                match = re.search(
                    r'(?:verify|check|ensure|assert)(?:\s+that)?(?:\s+the)?\s+(.+?)\s+(?:contains|has|shows|displays)(?:\s+the)?(?:\s+text)?\s+[\'\"]?([^\'"]+)[\'\"]?', sentence, re.IGNORECASE)
                if match:
                    element_desc = match.group(1).strip()
                    expected_text = match.group(2).strip()
                    action = AssertTextAction(
                        element_description=element_desc,
                        expected_text=expected_text,
                        description=f"Verify {element_desc} contains text '{expected_text}'"
                    )

            # Process assertion for table rows if the rule-based approach is too simple
            # This is a complex case that might be better handled by the LLM

            if action:
                actions.append(action)

        # If no actions were identified, return None to fall back to LLM
        if not actions:
            return None

        test_state.actions = actions
        return test_state

    def _llm_parse(self, scenario: str) -> TestState:
        """
        Parse a test scenario using an LLM.

        Args:
            scenario: Natural language description of the test

        Returns:
            TestState object with parsed actions
        """
        # Create a prompt template for the LLM
        template = """
        You are an expert test automation engineer. Convert the following test scenario into a structured sequence of actions.
        The available action types are:
        
        1. visit - Navigate to a URL
        2. click - Click on an element
        3. fill - Enter text in a form field
        4. scroll - Scroll to make an element visible
        5. assert_visible - Verify an element is visible
        6. assert_text - Verify an element contains specific text
        7. assert_row - Verify a table row or list item contains specific data
        
        For each action, extract the necessary details like URLs, element descriptions, expected text, etc.
        
        TEST SCENARIO:
        {scenario}
        
        OUTPUT FORMAT:
        Return a JSON object with the following structure:
        {{
            "name": "Brief descriptive name for the test",
            "description": "The original test scenario",
            "actions": [
                {{
                    "type": "visit",
                    "url": "http://example.com"
                }},
                {{
                    "type": "click",
                    "element_description": "Sign in button"
                }},
                // More actions following the patterns above
            ]
        }}
        
        Ensure all the necessary details are extracted for each action. Use descriptive element_description values that will work well with a vision-based model that can see the page.
        """

        prompt = prompts.ChatPromptTemplate.from_template(template)

        # Generate the response
        chain = prompt | self.llm
        response = chain.invoke({"scenario": scenario})

        # Extract the JSON from the response
        import json
        import re

        # Look for JSON content in the response
        json_match = re.search(r'```json(.*?)```', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to extract JSON without code blocks
            json_str = response.content

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse LLM response as JSON: {response.content}")
            # Create a minimal TestState with the scenario as description
            return TestState(
                name="Failed to parse",
                description=scenario,
                error="Failed to parse test scenario"
            )

        # Create a TestState from the parsed data
        test_state = TestState(
            name=data.get("name", "Parsed Test"),
            description=data.get("description", scenario)
        )

        # Convert the actions to their specific types
        raw_actions = data.get("actions", [])
        typed_actions: List[Action] = []

        for raw_action in raw_actions:
            action_type = raw_action.get("type", "").lower()

            if action_type == "visit":
                typed_actions.append(VisitAction(
                    url=raw_action.get("url", ""),
                    description=f"Visit {raw_action.get('url', '')}"
                ))
            elif action_type == "click":
                typed_actions.append(ClickAction(
                    element_description=raw_action.get(
                        "element_description", ""),
                    description=f"Click on {raw_action.get('element_description', '')}"
                ))
            elif action_type == "fill":
                typed_actions.append(FillAction(
                    element_description=raw_action.get(
                        "element_description", ""),
                    text=raw_action.get("text", ""),
                    description=f"Fill {raw_action.get('element_description', '')} with '{raw_action.get('text', '')}'"
                ))
            elif action_type == "scroll":
                typed_actions.append(ScrollAction(
                    element_description=raw_action.get(
                        "element_description", ""),
                    description=f"Scroll to {raw_action.get('element_description', '')}"
                ))
            elif action_type == "assert_visible":
                typed_actions.append(AssertVisibleAction(
                    element_description=raw_action.get(
                        "element_description", ""),
                    description=f"Verify {raw_action.get('element_description', '')} is visible"
                ))
            elif action_type == "assert_text":
                typed_actions.append(AssertTextAction(
                    element_description=raw_action.get(
                        "element_description", ""),
                    expected_text=raw_action.get("expected_text", ""),
                    description=f"Verify {raw_action.get('element_description', '')} contains text '{raw_action.get('expected_text', '')}'"
                ))
            elif action_type == "assert_row":
                typed_actions.append(AssertRowAction(
                    expected_data=raw_action.get("expected_data", {}),
                    description=f"Verify table row contains {raw_action.get('expected_data', {})}"
                ))

        test_state.actions = typed_actions
        return test_state
