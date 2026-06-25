import requests
import json

endpoints = [
    "https://spire-codex.com/api/runs/scores/cards",
    "https://spire-codex.com/api/runs/scores/relics",
    "https://spire-codex.com/api/runs/scores/potions",
]

for url in endpoints:
    r = requests.get(url)
    print(f"{r.status_code} -> {url}")
    if r.status_code == 200:
        print(json.dumps(r.json(), indent=2)[:800])
    print()