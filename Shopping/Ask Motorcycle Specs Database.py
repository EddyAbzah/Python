import os
import requests


# The base URL for the Motorcycle Specs Database API
url = "https://motorcycle-specs-database.p.rapidapi.com/model/make-id/100/category/Sport"

# Headers including your RapidAPI Key
headers = {
    "X-RapidAPI-Key": os.getenv('RAPID_API_KEY'),
    "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com",
    "Content-Type": "application/json"
}

# Make the GET request to retrieve data
response = requests.get(url, headers=headers)

# Check the response status
if response.status_code == 200:
    data = response.json()  # Convert the response to JSON
    # Process the data here (for example, print it)
    print("Motorcycle Data Retrieved:")
    print(data)  # or you can process and extract specific info from 'data'
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
    print(response.text)  # Print the error message if the request failed
