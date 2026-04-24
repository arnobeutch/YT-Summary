"""RAG engine to summarize meeting transcripts using LangChain, ChromaDB, and Ollama."""

from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

import my_constants
from my_logger import my_logger

CHROMA_PERSIST_DIR = Path("chroma_db")


def build_vectorstore_from_utterances(
    utterances: list[tuple[str, str]],
    model: str,
) -> Chroma:
    """Return persistent Chroma vector store updated with new transcript.

    Args:
        utterances (list[tuple[str, str]]): Speaker-tagged utterances.
        model (str): Ollama model name.

    Returns:
        Chroma: Persistent vector store.

    """
    text_chunks = [f"{speaker} : {text}" for speaker, text in utterances]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.create_documents(text_chunks)  # pyright: ignore[reportUnknownMemberType]  # langchain metadatas param typed as dict[Unknown, Unknown]

    embeddings = OllamaEmbeddings(model=model)

    # Load existing or create new Chroma DB
    if CHROMA_PERSIST_DIR.exists():
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
    else:
        vectorstore = Chroma.from_documents(  # pyright: ignore[reportUnknownMemberType]  # langchain client_settings / collection_metadata typed as Unknown
            documents=documents,
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
        return vectorstore  # noqa: RET504

    # Append new documents
    vectorstore.add_documents(documents)

    return vectorstore


def generate_summary(
    utterances: list[tuple[str, str]],
    language: str,
    model: str,
    prompt: str | None = None,
) -> str:
    """Return markdown-formatted summary from structured utterances.

    Args:
        utterances: Speaker-tagged utterances.
        language: 'fr' or 'en' — used only when ``prompt`` is not given.
        model: Model name for Ollama (e.g., 'mistral').
        prompt: Override the built-in language-defaulted prompt (e.g. for
            summary-mode templates).

    Returns:
        str: Markdown summary of the meeting in the requested language.

    """
    vectorstore = build_vectorstore_from_utterances(utterances, model)

    my_logger.info("Generating summary...")
    llm = ChatOllama(model=model)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        return_source_documents=False,
    )

    if prompt is None:
        prompt = (
            my_constants.RAG_FRENCH_PROMPT if language == "fr" else my_constants.RAG_ENGLISH_PROMPT
        )
    result = qa_chain.invoke({"query": prompt})
    return result["result"]
