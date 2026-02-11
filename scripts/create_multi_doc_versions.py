#!/usr/bin/env python3
"""Create default multi-document summarization versions."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.versions import create_version, get_default_multi_doc_summarization_config


def create_gemini_version():
    """Create Gemini multi-doc summarization version."""
    config = get_default_multi_doc_summarization_config()
    config['multi_doc_summarization']['method'] = 'gemini'
    config['multi_doc_summarization']['llm_model'] = 'gemini-2.0-flash'

    version_id = create_version(
        name='gemini-flash',
        description='Multi-doc summarization using Gemini 2.0 Flash (faster, cheaper)',
        configuration=config,
        analysis_type='multi_doc_summarization'
    )
    print(f"✅ Created Gemini version: {version_id}")
    return version_id


def create_openai_version():
    """Create OpenAI multi-doc summarization version."""
    config = get_default_multi_doc_summarization_config()
    config['multi_doc_summarization']['method'] = 'openai'
    config['multi_doc_summarization']['llm_model'] = 'gpt-4o'

    version_id = create_version(
        name='openai-gpt4o',
        description='Multi-doc summarization using OpenAI GPT-4o (more detailed)',
        configuration=config,
        analysis_type='multi_doc_summarization'
    )
    print(f"✅ Created OpenAI version: {version_id}")
    return version_id


def main():
    """Create both default versions."""
    print("Creating multi-document summarization versions...\n")

    try:
        gemini_id = create_gemini_version()
        print()
        openai_id = create_openai_version()
        print()
        print("✨ Successfully created both versions!")
        print(f"\nGemini version ID: {gemini_id}")
        print(f"OpenAI version ID: {openai_id}")
        print("\nYou can now use these versions in the Article Insights page.")

    except ValueError as e:
        if "already exists" in str(e):
            print("⚠️  Versions already exist. Skipping creation.")
        else:
            print(f"❌ Error: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
