"""
Gemini Client - Direct Google AI API integration.

V4.9: Uses new google-genai SDK (replacing deprecated google-generativeai).
Benefits:
- Free tier: 15 requests/minute, 1M tokens/day
- Lower latency (no proxy)
- No markup costs
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Default model - Gemini 3 Flash Preview (1000 RPM on paid tier)
DEFAULT_MODEL = "gemini-3-flash-preview"


def _clean_json(json_str: str) -> str:
    """Clean LLM JSON output (fix trailing commas, extract from markdown, repair truncation)"""
    if not json_str:
        return "{}"

    original = json_str[:200]  # Keep for debug

    # V4.9.4: Replace Unicode quotes with ASCII quotes (Gemini 3 issue)
    # Use explicit Unicode code points for reliability across all encodings
    json_str = json_str.replace('\u201c', '"').replace('\u201d', '"')  # Curly double quotes
    json_str = json_str.replace('\u2018', "'").replace('\u2019', "'")  # Curly single quotes
    json_str = json_str.replace('\u201e', '"').replace('\u201f', '"')  # German low/high quotes
    json_str = json_str.replace('\u00ab', '"').replace('\u00bb', '"')  # French guillemets
    json_str = json_str.replace('\u300c', '"').replace('\u300d', '"')  # CJK brackets
    json_str = json_str.replace('\u300e', '"').replace('\u300f', '"')  # CJK white brackets

    # Remove any BOM or zero-width characters
    json_str = json_str.replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')

    # Remove markdown code blocks
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        parts = json_str.split("```")
        if len(parts) >= 2:
            json_str = parts[1]

    json_str = json_str.strip()

    # V4.9.2: Extract JSON from text (Gemini 3 sometimes adds explanation before JSON)
    # Find first { or [ that starts the JSON
    first_brace = json_str.find('{')
    first_bracket = json_str.find('[')

    if first_brace == -1 and first_bracket == -1:
        logger.debug(f"No JSON found in response: {original}")
        return "{}"

    # Use whichever comes first (and exists)
    if first_brace == -1:
        start = first_bracket
    elif first_bracket == -1:
        start = first_brace
    else:
        start = min(first_brace, first_bracket)

    if start > 0:
        # There's text before the JSON - remove it
        logger.debug(f"Removing prefix text before JSON: '{json_str[:start]}'")
        json_str = json_str[start:]

    # V4.9.1: More aggressive trailing comma fix (multiple passes)
    # Fix all variations of trailing commas
    for _ in range(3):  # Multiple passes for nested structures
        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
        json_str = re.sub(r',\s*$', '', json_str)  # Trailing comma at end

    # V4.9.1: Fix unterminated strings by finding and closing them
    # Check for unbalanced quotes (odd number = unterminated string)
    quote_count = json_str.count('"') - json_str.count('\\"')
    if quote_count % 2 != 0:
        # Find last complete key-value pair
        # Look for patterns like: "key": "value" or "key": number
        last_complete = max(
            json_str.rfind('",'),
            json_str.rfind('"\n'),
            json_str.rfind('" '),
            json_str.rfind('"}'),
            json_str.rfind('"]'),
        )
        if last_complete > 0:
            json_str = json_str[:last_complete + 1]
        else:
            # Try to close the unterminated string
            json_str = json_str.rstrip() + '"'

    # V4.9: Repair truncated JSON (common with rate limits)
    # Count brackets to check for truncation
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')

    # If truncated mid-structure, try to close it
    if open_braces > close_braces or open_brackets > close_brackets:
        # Remove incomplete trailing content after last complete value
        last_good = max(
            json_str.rfind('",'),
            json_str.rfind('": '),
            json_str.rfind('true'),
            json_str.rfind('false'),
            json_str.rfind('null'),
            json_str.rfind('],'),
            json_str.rfind('},'),
            json_str.rfind('}'),
            json_str.rfind(']'),
        )

        if last_good > 0 and last_good < len(json_str) - 1:
            json_str = json_str[:last_good + 1]

        # Remove any trailing comma after truncation
        json_str = json_str.rstrip().rstrip(',')

        # Recount after cleanup
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')

        # Add missing closing brackets (arrays first, then objects)
        missing_brackets = open_brackets - close_brackets
        missing_braces = open_braces - close_braces
        json_str += ']' * max(0, missing_brackets) + '}' * max(0, missing_braces)

    # Final cleanup pass for trailing commas
    json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

    return json_str.strip()


class GeminiClient:
    """
    Unified Gemini client for financial analysis.

    V4.9: Uses new google-genai SDK.

    Usage:
        client = GeminiClient()
        await client.initialize()

        # Simple chat
        response = await client.chat("Analyze NVDA stock")

        # JSON response
        data = await client.chat_json("Return sentiment as JSON", schema={"sentiment": "float"})

        # With system prompt
        response = await client.chat(
            "Analyze this news",
            system_prompt="You are a financial analyst"
        )
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Google AI API key (or GOOGLE_AI_API_KEY env var)
            model: Model name (default: gemini-3-flash-preview)
        """
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        self.model_name = model or os.getenv('GEMINI_MODEL', DEFAULT_MODEL)
        self._client = None
        self._initialized = False

    async def initialize(self):
        """Initialize the Gemini client"""
        if self._initialized:
            return True

        if not self.api_key:
            logger.warning("No GOOGLE_AI_API_KEY found - Gemini client disabled")
            return False

        try:
            # V4.9: Use new google-genai Client
            self._client = genai.Client(api_key=self.api_key)

            # Use model from configuration (set in __init__ from env or parameter)
            # No need to override - self.model_name is already set correctly
            logger.info(f"Using model: {self.model_name}")

            self._initialized = True
            logger.info(f"GeminiClient initialized with {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return False

    def is_available(self) -> bool:
        """Check if client is ready"""
        return self._initialized and self._client is not None

    async def chat(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 1.0,
        max_tokens: int = 1500
    ) -> str:
        """
        Send a chat message and get a response.

        V4.9.1: Gemini 3 recommends temperature=1.0 (default).
        Lower values may cause loops or degraded performance.

        Args:
            prompt: User message
            system_prompt: Optional system instruction
            temperature: Creativity (Gemini 3 recommends 1.0)
            max_tokens: Maximum response length

        Returns:
            Response text
        """
        if not self.is_available():
            if not await self.initialize():
                return ""

        try:
            # Build contents with system instruction
            contents = []
            if system_prompt:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=f"[System]: {system_prompt}")]
                ))
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text="Understood. I will follow these instructions.")]
                ))

            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            ))

            # Configure generation
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )

            # Generate response (run in thread pool for async)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
            )

            if response and response.text:
                return response.text
            return ""

        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            return ""

    async def chat_json(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 1.0,
        max_tokens: int = 2000
    ) -> Dict:
        """
        Send a chat message and parse JSON response.

        V4.9.1: Uses temperature=1.0 (Gemini 3 recommended default).
        Increased max_tokens to 2000 to avoid truncation.
        Added retry logic for better reliability.

        Args:
            prompt: User message (should ask for JSON output)
            system_prompt: Optional system instruction
            temperature: Creativity (lower for JSON)
            max_tokens: Maximum response length

        Returns:
            Parsed JSON dict, or empty dict on error
        """
        # Add stronger JSON instruction to system prompt
        json_system = system_prompt or ""
        json_system += "\n\nCRITICAL: You MUST respond with valid, complete JSON only. No markdown code blocks, no explanation, no truncation. Keep responses concise to fit within token limits."

        # Try up to 2 times
        for attempt in range(2):
            response = await self.chat(
                prompt,
                system_prompt=json_system,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if not response:
                continue

            try:
                cleaned = _clean_json(response)
                result = json.loads(cleaned)
                return result
            except json.JSONDecodeError as e:
                if attempt == 0:
                    # First attempt failed, try with simpler prompt
                    logger.info(f"JSON parse attempt 1 failed: {e}")
                    logger.info(f"Raw Gemini response: {response[:500]}")
                    prompt = prompt + "\n\nRespond with MINIMAL, valid JSON only. No text before or after the JSON."
                    # V4.9.1: Don't lower temperature (Gemini 3 recommends 1.0)
                else:
                    logger.warning(f"Failed to parse Gemini JSON response: {e}")
                    logger.warning(f"Raw Gemini response: {response[:300]}")

        return {}

    async def analyze_financial(
        self,
        content: str,
        analysis_type: str = "sentiment"
    ) -> Dict:
        """
        Specialized financial analysis.

        Args:
            content: Text to analyze (news, social posts, etc.)
            analysis_type: "sentiment", "news", or "trend"

        Returns:
            Analysis result dict
        """
        prompts = {
            "sentiment": """Analyze the sentiment of this financial content.
Return JSON with:
{
    "sentiment_score": float between -1.0 (very negative) and +1.0 (very positive),
    "sentiment_label": "bullish" | "bearish" | "neutral",
    "confidence": float 0-1,
    "key_themes": ["theme1", "theme2"],
    "summary": "one sentence summary"
}

Content:
""",
            "news": """Analyze this financial news for trading signals.
Return JSON with:
{
    "impact_score": float -100 to +100,
    "urgency": "high" | "medium" | "low",
    "category": "earnings" | "product" | "management" | "macro" | "regulatory" | "other",
    "affected_symbols": ["SYMBOL1", "SYMBOL2"],
    "key_insight": "main takeaway",
    "trade_implication": "buy" | "sell" | "hold" | "watch"
}

News:
""",
            "trend": """Analyze this content for emerging market trends.
Return JSON with:
{
    "sentiment": float -1 to +1,
    "momentum": "accelerating" | "stable" | "decelerating",
    "catalysts": ["catalyst1", "catalyst2"],
    "symbols": ["SYMBOL1", "SYMBOL2"],
    "key_insight": "main insight",
    "investment_thesis": "brief thesis if opportunity detected"
}

Content:
"""
        }

        system_prompt = "You are an expert financial analyst. Be precise and objective."
        prompt = prompts.get(analysis_type, prompts["sentiment"]) + content

        return await self.chat_json(prompt, system_prompt=system_prompt)

    async def close(self):
        """Clean up resources"""
        self._client = None
        self._initialized = False


# Singleton instance
_gemini_instance: Optional[GeminiClient] = None


async def get_gemini_client() -> GeminiClient:
    """Get or create the GeminiClient singleton"""
    global _gemini_instance

    if _gemini_instance is None:
        _gemini_instance = GeminiClient()
        await _gemini_instance.initialize()

    return _gemini_instance


# Quick test
if __name__ == "__main__":
    async def test():
        client = GeminiClient()
        if await client.initialize():
            print(f"Model: {client.model_name}")

            # Test simple chat
            response = await client.chat("What is NVIDIA's stock symbol?")
            print(f"Response: {response[:200]}")

            # Test JSON
            data = await client.analyze_financial(
                "NVIDIA reported record earnings, beating expectations by 20%",
                "news"
            )
            print(f"Analysis: {json.dumps(data, indent=2)}")
        else:
            print("Failed to initialize - check GOOGLE_AI_API_KEY")

    asyncio.run(test())
