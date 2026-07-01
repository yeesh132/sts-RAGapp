import requests
data = requests.get("https://spire-codex.com/api/monsters").json()
print([m.get("id") for m in data])