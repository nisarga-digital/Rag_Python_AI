from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def validate_api_key(api_key: str) -> str:
    """Validate the Gemini API key before any model call."""
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file or enter it in the app."
        )

    api_key = api_key.strip()
    if len(api_key) < 20:
        raise ValueError(
            "GEMINI_API_KEY looks incomplete. Double-check the full key was copied into .env."
        )
    return api_key


def get_embeddings(api_key: str) -> Embeddings:
    api_key = validate_api_key(api_key)
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )


def get_llm(api_key: str, temperature: float, max_tokens: int) -> ChatGoogleGenerativeAI:
    api_key = validate_api_key(api_key)
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
