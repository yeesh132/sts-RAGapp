import requests
import json

endpoints = [
    "https://spire-codex.com/api/runs/scores/card?character=IRONCLAD",
    "https://spire-codex.com/api/runs/scores/card?limit=10",
    "https://spire-codex.com/api/runs/scores/card?ascension=0",
    "https://spire-codex.com/api/runs/stats/card/BASH?character=IRONCLAD",
    "https://spire-codex.com/api/runs/stats/card/BASH?ascension=0",
    "https://spire-codex.com/api/runs/top/card/IRONCLAD",
]

for url in endpoints:
    r = requests.get(url)
    print(f"{r.status_code} -> {url}")
    if r.status_code == 200:
        print(json.dumps(r.json(), indent=2)[:500])
    elif r.status_code == 422:
        print(json.dumps(r.json(), indent=2))  # 422 gives validation details
    print()