
import requests
import json

try:
    response = requests.post("http://localhost:8000/manual-update", 
                             json={"emotion": "happy"},
                             headers={"Content-Type": "application/json"})
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
