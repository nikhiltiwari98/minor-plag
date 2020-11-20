import requests

url = 'http://localhost:5000/cosineSimilarity_api'
r = requests.post(url,json={'plagtext':'bad'})

print(r.json())