#!/usr/bin/env python3
"""
Quick test of Mistral AI integration.

Usage:
    # Set environment variable first
    export MISTRAL_API_KEY=your_key_here

    # Then run the test
    python3 test_mistral.py
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_mistral_basic():
    """Test basic Mistral LLM call."""
    print("Testing Mistral AI integration...")

    # Check API key
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY not found in environment")
        print("   Please set it in .env file or export MISTRAL_API_KEY=your_key")
        return False

    print(f"✅ API key found (ends with: ...{api_key[-8:]})")

    # Import and test
    from src.llm import MistralLLM

    print("\n1. Testing basic text generation...")
    llm = MistralLLM(temperature=0.0)
    response = llm.generate("Say hello in exactly one word")

    print(f"   Response: {response.content}")
    print(f"   Provider: {response.provider}")
    print(f"   Model: {response.model}")
    print(f"   Tokens: {response.usage['input_tokens']} in, {response.usage['output_tokens']} out")

    # Test JSON mode
    print("\n2. Testing JSON mode...")
    response = llm.generate(
        'Return a JSON object with keys "name" and "age" for a person named Alice who is 30',
        json_mode=True
    )

    print(f"   Response: {response.content[:100]}...")

    import json
    try:
        data = json.loads(response.content)
        print(f"   ✅ Valid JSON: {data}")
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON: {e}")
        return False

    print("\n✅ All tests passed!")
    return True


def test_mistral_config():
    """Test loading Mistral from config."""
    print("\nTesting Mistral via config.yaml...")

    from src.llm import get_llm

    # Create config for Mistral
    config = {
        "provider": "mistral",
        "model": "mistral-large-latest",
        "temperature": 0.2,
        "max_tokens": 100
    }

    llm = get_llm(config)
    response = llm.generate("What is 2+2? Answer in one sentence.")

    print(f"   Response: {response.content}")
    print(f"   Provider: {response.provider}")
    print(f"   ✅ Config-based loading works!")

    return True


if __name__ == "__main__":
    print("="*60)
    print("MISTRAL AI INTEGRATION TEST")
    print("="*60)

    try:
        if test_mistral_basic() and test_mistral_config():
            print("\n" + "="*60)
            print("✅ ALL TESTS PASSED - Mistral integration working!")
            print("="*60)
        else:
            print("\n❌ Some tests failed")
            exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
