import requests
import json

# Ensure this URL matches the Flask backend's address
LOGIN_URL = "http://localhost:5000/login"

# Define the user data you want to embed in the token
user_data = {
    "user_id": "executive_leader_007",
    "role": "Chief Strategy Officer",
    "org": "Innovation Group"
}

try:
    response = requests.post(LOGIN_URL, json=user_data)
    response_data = response.json()

    if response.status_code == 200 and 'redirect_url' in response_data:
        print("✅ Login successful. COPY and PASTE the URL below into your browser:")
        print("-" * 50)
        print(response_data['redirect_url'])
        print("-" * 50)
    else:
        print(f"❌ Login failed. Status: {response.status_code}. Response: {response_data}")

except requests.exceptions.RequestException as e:
    print(f"❌ Error connecting to Flask backend. Is 'python api.py' running? Error: {e}")