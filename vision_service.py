"""
Vision service module for handling visual recognition tasks.
"""
import os
import json
import logging
import base64
import re
from typing import Dict, Any, Optional, Union
from openai import OpenAI

logger = logging.getLogger(__name__)


class VisionServiceError(Exception):
    """Custom exception for Vision Service errors."""
    pass


class VisionService:
    """
    Service for vision-based operations using OpenAI's vision models.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the vision service.

        Args:
            model_name: Name of the vision model to use
        """
        self.model_name = model_name or os.environ.get(
            "VISION_MODEL", "gpt-4o")
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        logger.info(
            f"Vision service initialized with model: {self.model_name}")

    async def ask_vision_text(self, image_data: bytes, prompt: str) -> str:
        """
        Ask the vision model a question about an image and get a text response.

        Args:
            image_data: Image bytes
            prompt: Text prompt for the vision model

        Returns:
            Text response from the vision model
        """
        try:
            # Encode the image as base64
            base64_image = base64.b64encode(image_data).decode('utf-8')

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in analyzing web page screenshots. Provide precise, direct answers."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Vision model text query failed: {e}")
            raise VisionServiceError(f"Failed to query vision model: {e}")

    async def ask_vision_json(self, image_data: bytes, prompt: str) -> Dict[str, Any]:
        """
        Ask the vision model a question about an image and get a JSON response.

        Args:
            image_data: Image bytes
            prompt: Text prompt for the vision model

        Returns:
            JSON response from the vision model
        """
        try:
            text_response = await self.ask_vision_text(
                image_data,
                prompt + "\nImportant: Return ONLY valid JSON without any commentary, explanation or surrounding text."
            )

            # Try to find and extract JSON from the response
            json_pattern = r'```(?:json)?(.*?)```'
            json_matches = re.findall(json_pattern, text_response, re.DOTALL)

            if json_matches:
                # Use the first JSON code block
                json_str = json_matches[0].strip()
            else:
                # Assume the entire response is JSON
                json_str = text_response.strip()

            try:
                # Try to parse the JSON
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Clean the string and try again
                cleaned_json = ''.join(
                    line for line in json_str.splitlines() if not line.strip().startswith('//'))
                # Remove any text before { and after }
                cleaned_json = re.sub(r'^[^{]*', '', cleaned_json)
                cleaned_json = re.sub(r'[^}]*$', '', cleaned_json)
                return json.loads(cleaned_json)

        except Exception as e:
            logger.error(f"Vision model JSON query failed: {e}")
            raise VisionServiceError(
                f"Failed to parse JSON from vision model response: {e}")
