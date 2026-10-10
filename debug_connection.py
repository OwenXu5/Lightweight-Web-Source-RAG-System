import os
import httpx
import requests
import json
from dotenv import load_dotenv
import logging

# logging.basicConfig(level=logging.DEBUG)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

print(f"DEBUG: API Base: {base_url}")

if not api_key or not base_url:
    print("ERROR: Missing API_KEY or API_BASE")
    exit(1)

url = f"{base_url}/embeddings"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "input": "test",
    "model": "ecnu-embedding-small"
}

print("\n--- Test 1: requests (verify=False) ---")
try:
    resp = requests.post(url, headers=headers, json=data, verify=False)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        print("Success!")
    else:
        print(f"Failed: {resp.text[:100]}")
except Exception as e:
    print(f"Requests Error: {e}")

print("\n--- Test 2: httpx (verify=False, http2=False) ---")
try:
    with httpx.Client(verify=False, http2=False) as client:
        resp = client.post(url, headers=headers, json=data)
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            print("Success!")
        else:
            print(f"Failed: {resp.text[:100]}")
except Exception as e:
    print(f"HTTPX Error: {e}")
