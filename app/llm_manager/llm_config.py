from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
    """Configuration specific to the Local LLM Generation layer."""
    # Ollama Local LLM Settings
    OLLAMA_HOST: str = "http://localhost:11434"
    LLM_MODEL: str = "phi3.5" #"llama3.2:3b" #"qwen2.5:7b" #"phi3" #"mistral"
    LLM_TEMPERATURE: float = 0.0
    LLM_CONTEXT_WINDOW: int = 4096
    LLM_SYSTEM_PROMPT: str = (
        "You are a highly precise, professional documentary assistant. "
        "Answer the user's question using ONLY the provided 'Context Information'.\n\n"
        "STRICT RULES:\n"
        "1. NO HALLUCINATIONS: If the answer is not contained in the context, explicitly state: "
        "'I don't have enough information based on the provided documents.'\n"
        "2. MANDATORY CITATIONS: You MUST cite the source document and page number for every factual claim you make.\n"
        "3. CITATION FORMAT: Use the exact format [Source: filename.pdf, Page: X] at the end of the relevant sentence.\n"
        "4. Be concise, direct, and structure your answer with bullet points if explaining multiple concepts."
    )
    LLM_NO_CONTEXT_RESPONSE: str = "I could not find any relevant information in the provided documents to answer your question."

    model_config = SettingsConfigDict(env_file=".env")

llm_settings = LLMSettings()