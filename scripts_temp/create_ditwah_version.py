from src.versions import create_version, get_default_ditwah_claims_config

version_id = create_version(
    "ditwah-v1",
    "DITWAH claims with local LLM",
    get_default_ditwah_claims_config(),
    analysis_type="ditwah_claims",
)

print(f"Version ID: {version_id}")

