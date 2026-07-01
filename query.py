import os
import sys
import requests
import chromadb
# from chromadb.config import Settings


# Config
PERSIST_DIR        = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
OLLAMA_BASE_URL    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_MODEL       = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
COLLECTION_NAME    = "sts_codex"
TOP_K              = 5


# system prompt
def build_system_prompt() -> str:
    return (
        "You are a Slay the Spire 2 assistant helping a player during a run. "
        "Answer questions about cards, relics, potions, monsters, and events "
        "using only the context provided below. "
        "Keep your answers concise — a few sentences at most. "
        "Only expand with more detail if the player explicitly asks for it. "
        "If the context does not contain enough information to answer, "
        "say so clearly rather than guessing. "
        "Always cite the source URL for any information you use."
    )


# Ollama helpers
# ollama exposes its own api for generation and embedding
def embed(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def generate(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# retrieval helpers
def retrieve(
    query: str,
    col: chromadb.Collection,
    k: int = TOP_K,
    entity_type: str | None = None,
) -> tuple[list[str], list[dict], list[float]]:
    """
    Search Chroma for the top-k chunks most semantically similar to the query.

    Args:
        query:       The user's question.
        col:         The Chroma collection to search.
        k:           Number of chunks to retrieve.
        entity_type: Optional filter — restricts search to one entity type.
                     e.g. "card", "monster", "relic"

    Returns:
        docs:      The text chunks retrieved.
        metadatas: The metadata dicts for each chunk.
        distances: Similarity scores — lower means more relevant.
    """
    query_embedding = embed(query)

    where = {"type": entity_type} if entity_type else None

    # Query the collection for the top-k most similar chunks
    # chroma handles the embedding search and returns the results
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return docs, metadatas, distances


#  Prompt assembly
def build_prompt(query: str, docs: list[str], metadatas: list[dict]) -> str:
    """
    Combine the system prompt, retrieved context chunks, and the user's
    question into a single string for the generation model.
    """
    context_blocks = []
    for doc, meta in zip(docs, metadatas):
        block = (
            f"[{meta['type'].upper()}] {meta.get('name', meta['id'])}\n"
            f"{doc}\n"
            f"Source: {meta['url']}"
        )
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)
    

    return (
        f"{build_system_prompt()}\n\n"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== ANSWER ==="
    )



# Main query function
def ask(
    query: str,
    entity_type: str | None = None,
    k: int = TOP_K,
) -> dict:
    """
    Ask the assistant a question and return the answer with sources
    and retrieval confidence scores.

    Args:
        query:       The player's question.
        entity_type: Optional type filter e.g. "card", "monster", "relic".
        k:           Number of chunks to retrieve.

    Returns:
        {
            "answer":     str,
            "sources":    list[str],   # source URLs of retrieved chunks
            "confidence": list[float], # distances (lower = more relevant)
        }
    """
    # Load the Chroma collection from disk
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    try:
        col = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return {
            "answer": (
                "No index found. Please run ingest.py first to build the database."
            ),
            "sources":    [],
            "confidence": [],
        }

    # Retrieve relevant chunks
    docs, metadatas, distances = retrieve(query, col, k=k, entity_type=entity_type)

    if not docs:
        return {
            "answer":     "No relevant information found in the index.",
            "sources":    [],
            "confidence": [],
        }

    # Build prompt and generate
    prompt = build_prompt(query, docs, metadatas)
    answer = generate(prompt)

    return {
        "answer":     answer,
        "sources":    [m["url"] for m in metadatas],
        "confidence": distances,
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for url in result["sources"]:
        print(f"  {url}")

    print("\nRETRIEVAL CONFIDENCE (lower = more relevant):")
    for i, dist in enumerate(result["confidence"]):
        print(f"  Chunk {i + 1}: {dist:.4f}")
    print()


def parse_type_prefix(raw: str) -> tuple[str, str | None]:
    """
    Check if the user prefixed their question with an entity type filter.
    e.g. "card: what does Bash do?" -> ("what does Bash do?", "card")
    """
    # Valid entity types for filtering
    valid_types = {
        "card", "relic", "monster", "encounter",
        "potion", "power", "keyword", "event", "enchantment", "character",
    }
    # may jjust get rid of this and ignore any colon
    if ":" in raw:
        prefix, rest = raw.split(":", 1)
        if prefix.strip().lower() in valid_types:
            return rest.strip(), prefix.strip().lower()
    return raw, None


def repl() -> None:
    print("=== STS Assistant ===")
    print("Ask anything about your run. Type 'quit' to exit.")
    print("Tip: prefix with a type to narrow results, e.g. 'monster: Slime Boss'\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw or raw.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        query, entity_type = parse_type_prefix(raw)

        print("Thinking...\n")
        result = ask(query, entity_type=entity_type)
        print_result(result)


# Entry point
if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Question: {question}\n")
        result = ask(question)
        print_result(result)
    else:
        repl()