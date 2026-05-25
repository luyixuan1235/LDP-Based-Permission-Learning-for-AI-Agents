"""
Quick connectivity check for OpenAI-compatible chat endpoints.

This script validates the current .env configuration before running
long IC-only / IC+CF experiments.
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv


def main() -> int:
    """Run a small chat completion request and report status."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is missing.")
        print("Set it in .env first, then rerun this script.")
        return 1

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("OPENAI_MODEL", "o4-mini").strip()

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    print("=" * 60)
    print("API Connectivity Check")
    print("=" * 60)
    print(f"Model:    {model}")
    print(f"Base URL: {base_url or '(OpenAI default)'}")

    try:
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=8,
        )

        text = (response.choices[0].message.content or "").strip()
        print("\nSUCCESS: Request completed.")
        print(f"Response preview: {text}")
        print("=" * 60)
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print("\nFAILED: Unable to complete test request.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        print("\nTroubleshooting:")
        print("1) Check OPENAI_API_KEY is valid.")
        print("2) Check OPENAI_BASE_URL and model name match your provider.")
        print("3) Confirm your network can access the API endpoint.")
        print("=" * 60)
        return 2


if __name__ == "__main__":
    sys.exit(main())
