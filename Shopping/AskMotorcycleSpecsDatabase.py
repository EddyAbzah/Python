import os
import requests


BASE_URL = "https://motorcycle-specs-database.p.rapidapi.com"
API_HOST = "motorcycle-specs-database.p.rapidapi.com"
API_KEY = os.getenv('RAPID_API_KEY')


def get_motorcycle_data(motorcycle_brand, motorcycle_model):
    url = f"{BASE_URL}/make/{motorcycle_brand}/model/{motorcycle_model}"
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": API_HOST,
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError if the response code is not 200
        data = response.json()
        return data
    except requests.exceptions.RequestException as error:
        print(f"Error fetching data: {error}")
        return None


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens a nested dictionary and combines nested keys using a separator (default: '_').
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # If the value is a list, join the list values as strings
            items.append((new_key, ', '.join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)


def pick_year(data, motorcycle_year):
    if data:
        for entry in data:
            if entry["articleCompleteInfo"]["yearName"] == motorcycle_year:
                flattened_dict = flatten_dict(entry)
                dict_out = dict([(f"{key.split("_")[-1].replace("Name", "")}", value) for key, value in flattened_dict.items()])
                return dict_out
    print("AskMotorcycleSpecsDatabase: pick_year() didn't match the picked year")
    return []


def main(motorcycle_brand, motorcycle_model, motorcycle_year):
    data = get_motorcycle_data(motorcycle_brand, motorcycle_model)
    data = pick_year(data, motorcycle_year)
    return data


if __name__ == "__main__":
    motorcycle_brand = "Honda"
    motorcycle_model = "CB500X"
    motorcycle_year = 2020

    motorcycle_data = main(motorcycle_brand, motorcycle_model, motorcycle_year)
    if motorcycle_data:
        for key, value in motorcycle_data.items():
            print(f"{key.split("_")[-1].replace("Name", "")}: {value}")
        print()
        print("\t".join(motorcycle_data.keys()))
        print("\t".join(str(v) for v in motorcycle_data.values()) + "\n")
    else:
        print("No data to display.")
