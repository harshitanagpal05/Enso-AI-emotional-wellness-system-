
import requests

url = "http://localhost:8000/manual-update"
data = {"emotion": "happy"}
response = requests.post(url, json=data)
print(response.json())
