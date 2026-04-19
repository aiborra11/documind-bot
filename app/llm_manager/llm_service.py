from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------- Project Imports ---------
from app.utils.utils import get_logger
from app.llm_manager.llm_config import llm_settings

logger = get_logger(__name__)


class PromptBuilder:
    """
    Handles dynamic prompt construction and context formatting.
    Designed to easily scale with new prompt templates and variables.
    """

    def __init__(self) -> None:
        # Create a dynamic LCEL-compatible prompt template
        self.qa_prompt_template = ChatPromptTemplate.from_messages([
            ("system", llm_settings.LLM_SYSTEM_PROMPT),
            ("human", "Context Information:\n{context}\n\nUser Question: {question}")
        ])

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formats the retrieved chunks explicitly injecting filename and page number.
        """
        if not chunks:
            return ""

        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "").strip()
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown file")
            page = metadata.get("page", "Unknown page")
            
            chunk_str = (
                f"--- Document [{i+1}] ---\n"
                f"Source: {source} | Page: {page}\n"
                f"Content: {content}\n"
            )
            formatted_chunks.append(chunk_str)

        return "\n".join(formatted_chunks)
    

class LLMService:
    """
    Handles language model interactions using LangChain and LCEL.
    """

    def __init__(self, prompt_builder: PromptBuilder) -> None:
        """
        Initializes the LCEL chain connecting the Prompt, the LLM, and the Output Parser.
        """
        self._prompt_builder = prompt_builder
        self._no_context_response = llm_settings.LLM_NO_CONTEXT_RESPONSE
        
        # 1. Initialize LangChain's Ollama integration
        self._llm = ChatOllama(
            base_url=llm_settings.OLLAMA_HOST,
            model=llm_settings.LLM_MODEL,
            temperature=llm_settings.LLM_TEMPERATURE,
            num_ctx=llm_settings.LLM_CONTEXT_WINDOW
        )
        
        # 2. Build the LCEL Chain: Prompt -> LLM -> String Output
        self._chain = self._prompt_builder.qa_prompt_template | self._llm | StrOutputParser()

        logger.info(
            f"LCEL LLM Service initialized. Host: {llm_settings.OLLAMA_HOST} | "
            f"Model: {llm_settings.LLM_MODEL}"
        )

    async def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Asynchronously generates a response executing the LCEL chain.
        """
        logger.info(f"Generating LangChain LLM response for query: '{query}'")

        if not retrieved_chunks:
            return self._no_context_response

        # Format context using the builder
        context_block = self._prompt_builder.format_context(retrieved_chunks)

        try:
            # Execute the LCEL chain asynchronously
            final_answer = await self._chain.ainvoke({
                "context": context_block,
                "question": query
            })
            
            logger.info("Successfully generated response from LCEL chain.")
            return final_answer

        except Exception as e:
            logger.error(f"Error communicating with LLM via LangChain: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate response. Check if Ollama is running.")
        