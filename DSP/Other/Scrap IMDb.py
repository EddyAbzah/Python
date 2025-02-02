import os
import requests
from bs4 import BeautifulSoup


def get_episode_titles():
    """Scrape episode titles from IMDb for the given season."""
    url = f"https://www.imdb.com/title/{imdb_id}/episodes/?season={season}"
    headers = {"User-Agent": "Mozilla/5.0"}  # Avoid getting blocked by IMDb
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Failed to fetch IMDb page. Check the IMDb ID and season number.")
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    episodes = soup.find_all("div", class_="ipc-title__text")

    episode_titles = {}
    for episode in episodes:
        text = episode.text.strip()
        if "∙" in text:
            parts = text.split("∙")
            episode_number = int(parts[0].split("E")[1])
            episode_name = parts[1].strip()
            episode_titles[episode_number] = episode_name
    return episode_titles


def rename_episodes(episode_titles, just_print=False):
    """Rename episode files based on IMDb episode titles."""
    files = [f for f in os.listdir(directory) if f.lower().endswith(('mp4', 'mkv', 'avi'))]
    if sort_episode_titles:
        files.sort()


    if len(files) != len(episode_titles):
        print("Warning: Number of files doesn't match the number of episodes!")
        return

    for i, filename in enumerate(files):
        episode_number = i + 1
        if episode_number not in episode_titles:
            continue

        file_ext = os.path.splitext(filename)[1]
        new_name = f"S{season:02}E{episode_number:02} - {episode_titles[episode_number]}{file_ext}"
        if just_print:
            print(f"{filename} → {new_name}")
        else:
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_name}")


if __name__ == '__main__':
    directory = r""
    imdb_id = ""
    season = 1
    sort_episode_titles = False     # sorting takes case sensitivity into account

    titles = get_episode_titles()
    if titles:
        rename_episodes(titles, just_print=True)
    user_input = input("Do you want to continue? (yes / no): ")
    if "y" in user_input.lower():
        rename_episodes(titles)
