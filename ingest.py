import os
import time
import requests
import chromadb
from chromadb.config import Settings
from datetime import datetime

# config

BASE             = "https://spire-codex.com"
HEADERS          = {"User-Agent": "sts-rag-ingester/0.1"}
PERSIST_DIR      = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
OLLAMA_BASE_URL  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
BATCH            = 32
COLLECTION_NAME  = "sts_codex"

# Endpoints to pull from
# ingested into vector db

ENDPOINTS = {
    "card":        "/api/cards",
    "relic":       "/api/relics",
    "monster":     "/api/monsters",
    "encounter":   "/api/encounters",
    "potion":      "/api/potions",
    "power":       "/api/powers",
    "keyword":     "/api/keywords",
    "event":       "/api/events",
    "enchantment": "/api/enchantments",
    "character":   "/api/characters",
}


# fetching 
def fetch_json(path: str) -> list:
    r = requests.get(BASE + path, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def gather_items() -> list[dict]:
    #Pull each entry from every configured endpoint
    items = []
    for entity_type, endpoint in ENDPOINTS.items():
        print(f"  Fetching {entity_type}s from {endpoint} ...")
        try:
            data = fetch_json(endpoint)
            for raw in data:
                items.append({"type": entity_type, "raw": raw})
            print(f"    → {len(data)} records")
        except Exception as e:
            print(f"Failed ({e}), skipping.")
    return items



# Normalise each entity into (id, text_chunk, metadata).
# The text chunk is what gets embedded — include everything useful here.

def normalize(item: dict) -> tuple[str, str, dict]:
    raw  = item["raw"]
    typ  = item["type"]
    # id
    doc_id = str(raw.get("id", "unknown"))

    # text chunk
    name   = raw.get("name") or raw.get("title") or doc_id
    desc   = raw.get("description") or raw.get("flavor") or raw.get("wiki_intro") or ""

    # Card-specific extras
    card_parts = []
    if typ == "card":
        if raw.get("type"):         card_parts.append(f"Type: {raw['type']}")
        if raw.get("rarity"):       card_parts.append(f"Rarity: {raw['rarity']}")
        if raw.get("character"):    card_parts.append(f"Character: {raw['character']}")
        if raw.get("cost") is not None: card_parts.append(f"Cost: {raw['cost']}")
        if raw.get("keywords"):     card_parts.append(f"Keywords: {', '.join(raw['keywords'])}")
        if raw.get("upgraded_description"):
            card_parts.append(f"Upgraded: {raw['upgraded_description']}")

    # Monster/encounter extras
    monster_parts = []
    if typ == "monster":
        if raw.get("hp"): monster_parts.append(f"HP: {raw['hp']}")
        if raw.get("moves"):
            moves = ", ".join(
                m.get("name", "") for m in raw["moves"] if isinstance(m, dict)
            )
            if moves:
                monster_parts.append(f"Moves: {moves}")

    # Relic extras
    relic_parts = []
    if typ == "relic":
        if raw.get("rarity"):       relic_parts.append(f"Rarity: {raw['rarity']}")
        if raw.get("character"):    relic_parts.append(f"Character: {raw['character']}")
        if raw.get("flavor"):       relic_parts.append(f"Flavor: {raw['flavor']}")

    extras = " | ".join(card_parts + monster_parts + relic_parts)

    text = f"{typ.upper()}: {name}\n{desc}"
    if extras:
        text += f"\n{extras}"

# Metadata for any additional fields
    metadata = {
        "id":       doc_id,
        "name":     name,
        "type":     typ,
        "source":   "spire-codex",
        "url":      f"{BASE}/{typ}s/{doc_id}",
        "accessed": datetime.utcnow().isoformat() + "Z",
    }
    # Chroma only accepts str/int/float/bool metadata values.
    # Flatten any useful scalar fields from raw here.
    for field in ("rarity", "character", "type", "cost", "hp"):
        val = raw.get(field)
        if isinstance(val, (str, int, float, bool)):
            metadata[field] = val

    return doc_id, text, metadata


# Ollama embedding helper
def embed(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

# Ingestion pipeline
def ingest():
    print(f"=== STS Codex Ingest ===")
    print(f"Chroma DB  : {PERSIST_DIR}")
    print(f"Embed model: {OLLAMA_EMBED_MODEL}\n")

    # Chroma client
    client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR)
    )
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    print("Fetching data from Spire Codex API")
    items   = gather_items()
    records = [normalize(it) for it in items]

    if not records:
        print("No records fetched. Exiting.")
        return 

    ids, docs, metadatas = zip(*records)
    ids, docs, metadatas = list(ids), list(docs), list(metadatas)

    # Skip already-indexed documents so reruns are safe
    try: 
        existing = set(col.get(ids=ids)["ids"])
    except Exception:
        existing = set()

    new_idx   = [i for i, _id in enumerate(ids) if _id not in existing]
    ids       = [ids[i] for i in new_idx]
    docs      = [docs[i] for i in new_idx]
    metadatas = [metadatas[i] for i in new_idx]

    print(f"\n{len(existing)} already indexed, {len(ids)} new records to embed.\n")

    if not ids:
        print("Nothing to do. Index is up to date.")
        client.persist()
        return

    # Embed and add in batches
    for i in range(0, len(docs), BATCH):
        batch_docs  = docs[i : i + BATCH]
        batch_ids   = ids[i : i + BATCH]
        batch_meta  = metadatas[i : i + BATCH]

        embeddings = []
        for j, doc in enumerate(batch_docs):
            print(f"  Embedding [{i + j + 1}/{len(docs)}] {batch_ids[j]}")
            embeddings.append(embed(doc))
            time.sleep(0.05)   # be polite to the local Ollama server

        col.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings,
        )

    client.persist()
    print(f"\n✓ Done. Index persisted to {PERSIST_DIR}")


if __name__ == "__main__":
    ingest()