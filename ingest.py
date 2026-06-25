import os
import time
import requests
import chromadb
from chromadb.config import Settings
from datetime import datetime


BASE               = "https://spire-codex.com"
HEADERS            = {"User-Agent": "sts-rag-ingester/0.1"}
PERSIST_DIR        = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
OLLAMA_BASE_URL    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
BATCH              = 32
COLLECTION_NAME    = "sts_codex"


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


SCORE_ENDPOINTS = {
    "cards":   "/api/runs/scores/cards",
    "relics":  "/api/runs/scores/relics",
    "potions": "/api/runs/scores/potions",
}


# helper for gather items by first getting all raw JSON data
def fetch_json(path: str) -> dict | list:
    r = requests.get(BASE + path, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()

# store all items in a dict for fast lookup when normalizing each entity
def gather_items() -> list[dict]:
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

# gather scores for cards, relicss and potions from communbity score API
def gather_scores() -> dict:
    all_scores = {}
    for label, endpoint in SCORE_ENDPOINTS.items():
        print(f"  Fetching community scores for {label} ...")
        try:
            data = fetch_json(endpoint)        # returns {ID: {score, elo, picks, wins, win_rate}}
            all_scores.update(data)
            print(f" {len(data)} score entries")
        except Exception as e:
            print(f"Failed ({e}), skipping scores for {label}.")
    return all_scores


# ---------------------------------------------------------------------------
# Normalise each entity into (id, text_chunk, metadata).
# scores is the pre-fetched community score for this specific entity, or None.
# ---------------------------------------------------------------------------
# uses an item as the paramter 
# uses score as a parameter to add community score information to the text chunk and metadata

def normalize(item: dict, score: dict | None) -> tuple[str, str, dict]:
    raw = item["raw"]
    typ = item["type"]

    doc_id = str(raw.get("id", "unknown"))

    name = raw.get("name") or raw.get("title") or doc_id
    desc = raw.get("description") or raw.get("flavor") or raw.get("wiki_intro") or ""

    # Card
    card_parts = []
    if typ == "card":
        if raw.get("type"):
            card_parts.append(f"Type: {raw['type']}")
        if raw.get("rarity"):
            card_parts.append(f"Rarity: {raw['rarity']}")
        if raw.get("character"):
            card_parts.append(f"Character: {raw['character']}")
        if raw.get("cost") is not None:
            card_parts.append(f"Cost: {raw['cost']}")
        if raw.get("keywords"):
            card_parts.append(f"Keywords: {', '.join(raw['keywords'])}")
        if raw.get("upgraded_description"):
            card_parts.append(f"Upgraded: {raw['upgraded_description']}")

    # Monster
    monster_parts = []
    if typ == "monster":
        if raw.get("hp"):
            monster_parts.append(f"HP: {raw['hp']}")
        if raw.get("moves"):
            moves = ", ".join(
                m.get("name", "") for m in raw["moves"] if isinstance(m, dict)
            )
            if moves:
                monster_parts.append(f"Moves: {moves}")

    # Relic
    relic_parts = []
    if typ == "relic":
        if raw.get("rarity"):
            relic_parts.append(f"Rarity: {raw['rarity']}")
        if raw.get("character"):
            relic_parts.append(f"Character: {raw['character']}")
        if raw.get("flavor"):
            relic_parts.append(f"Flavor: {raw['flavor']}")

    # Potion
    potion_parts = []
    if typ == "potion":
        if raw.get("rarity"):
            potion_parts.append(f"Rarity: {raw['rarity']}")
        if raw.get("character"):
            potion_parts.append(f"Character: {raw['character']}")
        if raw.get("target"):
            potion_parts.append(f"Target: {raw['target']}")

    # Event
    event_parts = []
    if typ == "event":
        if raw.get("act"):
            event_parts.append(f"Act: {raw['act']}")
        if raw.get("options"):
            option_names = [
                o.get("text", "") for o in raw["options"] if isinstance(o, dict)
            ]
            if option_names:
                event_parts.append(f"Options: {' | '.join(option_names)}")

    # Power
    power_parts = []
    if typ == "power":
        if raw.get("character"):
            power_parts.append(f"Character: {raw['character']}")

    # --- Community scores --------------------------------------------------
    # This is the key addition: if we have a community score for this entity,
    # append it to the text chunk so the embedding captures performance data.
    # The model can then reason about win rates when recommending cards/relics.
    score_parts = []
    if score:
        if score.get("score") is not None:
            score_parts.append(f"Codex Score: {score['score']}/100")
        if score.get("win_rate") is not None:
            score_parts.append(f"Win Rate: {score['win_rate']}%")
        if score.get("picks") is not None:
            score_parts.append(f"Picked: {score['picks']} times")

    # --- Assemble text chunk -----------------------------------------------
    all_extras = card_parts + monster_parts + relic_parts + potion_parts + event_parts + power_parts
    extras_str = " | ".join(all_extras)

    text = f"{typ.upper()}: {name}\n{desc}"
    if extras_str:
        text += f"\n{extras_str}"
    if score_parts:
        text += f"\nCommunity Stats: {' | '.join(score_parts)}"

    # --- Metadata ----------------------------------------------------------
    metadata = {
        "id":       doc_id,
        "name":     name,
        "type":     typ,
        "source":   "spire-codex",
        "url":      f"{BASE}/{typ}s/{doc_id}",
        "accessed": datetime.utcnow().isoformat() + "Z",
    }
    # Flatten scalar fields from raw
    for field in ("rarity", "character", "type", "cost", "hp"):
        val = raw.get(field)
        if isinstance(val, (str, int, float, bool)):
            metadata[field] = val

    # Store score in metadata too so it can be filtered on in query.py
    if score and score.get("score") is not None:
        metadata["community_score"] = score["score"]
    if score and score.get("win_rate") is not None:
        metadata["win_rate"] = score["win_rate"]

    return doc_id, text, metadata


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ---------------------------------------------------------------------------
# Main ingest routine
# ---------------------------------------------------------------------------

def ingest():
    print(f"=== STS Codex Ingest ===")
    print(f"Chroma DB  : {PERSIST_DIR}")
    print(f"Embed model: {OLLAMA_EMBED_MODEL}\n")

    # Chroma client
    client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR)
    )
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    # Step 1 — fetch community scores upfront so normalize() can use them
    print("Fetching community scores...")
    scores = gather_scores()
    print(f"  → {len(scores)} total score entries loaded\n")

    # Step 2 — fetch all game entities
    print("Fetching game data from Spire Codex API...")
    items = gather_items()

    # Step 3 — normalise, passing each entity its score (or None if not found)
    # scores.get(id) does a single dictionary lookup — fast and clean
    records = [normalize(it, scores.get(it["raw"].get("id"))) for it in items]

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

    # Step 4 — embed and store in batches
    for i in range(0, len(docs), BATCH):
        batch_docs = docs[i : i + BATCH]
        batch_ids  = ids[i : i + BATCH]
        batch_meta = metadatas[i : i + BATCH]

        embeddings = []
        for j, doc in enumerate(batch_docs):
            print(f"  Embedding [{i + j + 1}/{len(docs)}] {batch_ids[j]}")
            embeddings.append(embed(doc))
            time.sleep(0.05)

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