import requests
import json
import time
import argparse
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

########################################
# MODE & API CONFIGURATION
########################################
# MODE can be "local" or "api"
MODE = os.getenv("MODE", "local").lower()
MODE = ""
if MODE == "api":
    API_URL = os.getenv("OPENROUTER_API_URL")  # e.g., "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "DeepSeek V3 0324")
else:
    API_URL = "http://10.219.205.24:1234/v1/completions"

########################################
# PROMPT & CALL FUNCTIONS
########################################
def create_prompt(game_record):
    """
    Build a prompt for summarizing the game's description and a few reviews.
    """
    description = game_record.get("detailed_description") or game_record.get("short_description", "")
    reviews = game_record.get("reviews", [])
    
    # Use up to 3 reviews for brevity.
    review_texts = [rev.get("review", "") for rev in reviews[:3]]
    review_block = "\n".join(review_texts)

    prompt = (
        "Based on the following Steam game information, provide a single, concise summary in no more than 100 words that focuses solely on the gameplay mechanics, unique features, and overall tone. "
        "Return only the final summary as plain text with no headings, bullet points, or internal chain-of-thought details.\n\n"
        "Game Description:\n"
        f"{description}\n\n"
        "User Reviews (sample):\n"
        f"{review_block}\n\n"
        "Final Summary:"
    )
    return prompt

def call_lm_studio(prompt, max_tokens=8000, temperature=0.7, top_p=0.9, timeout=180):
    """
    Sends the prompt to the selected API.

    - In local mode (LM Studio), uses a payload with "prompt".
    - In API mode (OpenRouter), uses the chat completions payload with a "messages" array.

    Returns the generated text from the first choice.
    """
    headers = {"Content-Type": "application/json"}

    if MODE == "api":
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        current_api_url = API_URL  # e.g., "https://openrouter.ai/api/v1/chat/completions"
    else:
        # Local LM Studio mode
        local_model = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-1.7b")

        payload = {
            "model": local_model,  # Required when multiple models are loaded
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        current_api_url = API_URL

    try:
        response = requests.post(current_api_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if MODE == "api":
            choices = data.get("choices", [])
            if choices and isinstance(choices, list):
                return choices[0].get("message", {}).get("content", "").strip()
        else:
            choices = data.get("choices", [])
            if choices and isinstance(choices, list):
                return choices[0].get("text", "").strip()
        return ""
    except Exception as e:
        print(f"Error calling LM Studio API: {e}")
        return ""

########################################
# REVIEW FILTERING FUNCTIONS
########################################
def is_good_review(review_text):
    """
    Uses LM Studio (or OpenRouter API) to classify whether a review is high-quality/helpful.
    Returns True if the model answers 'Yes', False otherwise.
    Also prints the classification prompt and the LLM's answer for inspection.
    """
    prompt = (
        "Determine if the following review is a high-quality, helpful review for a game. "
        "Return only 'Yes' if it is helpful, or 'No' if it is not.\n\n"
        f"Review: {review_text}\n\nAnswer:"
    )
    print(f"Review classification prompt:\n{prompt}\n")
    answer = call_lm_studio(prompt, max_tokens=1000, temperature=0.0, top_p=1.0, timeout=60)
    print(f"Review classification answer: {answer}\n")
    return answer.strip().lower().startswith("yes")

def filter_reviews(reviews, max_reviews=100):
    """
    Filters a list of reviews by using LM Studio to decide which reviews are "good".
    Then sorts the good reviews by 'votes_up' (descending) and returns up to max_reviews.
    """
    good_reviews = []
    for review in reviews:
        text = review.get("review", "")
        if len(text.split()) < 5:
            continue
        if is_good_review(text):
            good_reviews.append(review)
            time.sleep(0.5)
    good_reviews = sorted(good_reviews, key=lambda r: r.get("votes_up", 0), reverse=True)
    return good_reviews[:max_reviews]

########################################
# DATA ACQUISITION FUNCTIONS
########################################
def get_app_list():
    """
    Fetch the complete list of Steam apps using the Steam Web API (方案1).
    Uses access_token from environment variable as query parameter.
    Returns a list of dictionaries with keys "appid" and "name".
    """
    print("Fetching app list from Steam API...")

    # Get access token from environment
    access_token = os.getenv("STEAM_ACCESS_TOKEN")

    if not access_token:
        print("Error: STEAM_ACCESS_TOKEN not found in environment variables")
        return []

    url = f"https://api.steampowered.com/IStoreService/GetAppList/v1/?access_token={access_token}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # The new API returns data in a different structure
        # Response format: {"response": {"apps": [...], "have_more_results": true/false}}
        apps_data = data.get("response", {}).get("apps", [])

        # Convert to the same format as the old API for compatibility
        apps = []
        for app in apps_data:
            apps.append({
                "appid": app.get("appid"),
                "name": app.get("name", "")
            })

        print(f"Fetched {len(apps)} apps.")
        return apps
    except Exception as e:
        print(f"Error fetching app list: {e}")
        return []

def get_store_data(appid, country="us", language="en"):
    """
    Fetch store details for a given appid using the Steam Storefront API.
    Returns the 'data' dictionary if successful; otherwise, returns None.
    """
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc={country}&l={language}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if str(appid) in data and data[str(appid)].get("success"):
            return data[str(appid)].get("data", {})
        else:
            return None
    except Exception as e:
        print(f"Error fetching store data for appid {appid}: {e}")
        return None

def get_reviews(appid, num_per_page=100):
    """
    Fetch reviews for a given appid using Steam's reviews endpoint.
    Returns a list of review objects (may be empty if none available).
    """
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1&num_per_page={num_per_page}&language=english"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        reviews = data.get("reviews", [])
        return reviews
    except Exception as e:
        print(f"Error fetching reviews for appid {appid}: {e}")
        return []

def sanitize_text(text):
    """
    Replace unusual line terminators or other control characters in a string.
    """
    if not isinstance(text, str):
        return text
    return text.replace('\u2028', ' ').replace('\u2029', ' ')

def sanitize_data(obj):
    """
    Recursively sanitize all strings in a nested data structure.
    """
    if isinstance(obj, str):
        return sanitize_text(obj)
    elif isinstance(obj, list):
        return [sanitize_data(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: sanitize_data(value) for key, value in obj.items()}
    else:
        return obj

def save_game_data(game_data, output_file):
    """
    Append a single game's data as a JSON line to the output file,
    after sanitizing unusual characters.
    """
    try:
        game_data = sanitize_data(game_data)
        with open(output_file, "a", encoding="utf-8") as f:
            json_line = json.dumps(game_data, ensure_ascii=False)
            f.write(json_line + "\n")
            f.flush()
    except Exception as e:
        print(f"Error saving game data for appid {game_data.get('appid')}: {e}")

def load_processed_appids(checkpoint_file):
    """
    Load already processed app IDs from the checkpoint file.
    Returns a set of app IDs as strings.
    """
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        print(f"Loading already processed app IDs from {checkpoint_file}...")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                processed_ids.add(line.strip())
    return processed_ids

def append_checkpoint(appid, checkpoint_file):
    """
    Append an appid to the checkpoint file.
    """
    try:
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(str(appid) + "\n")
            f.flush()
    except Exception as e:
        print(f"Error writing appid {appid} to checkpoint: {e}")

########################################
# MAIN DATA ACQUISITION LOGIC
########################################
def main(limit=None, sleep_time=1, output_file="steam_games_data.jsonl", checkpoint_file="processed_ids.txt"):
    apps = get_app_list()
    if not apps:
        print("No apps found. Exiting.")
        return

    print(f"Total apps fetched: {len(apps)}")
    processed_appids = load_processed_appids(checkpoint_file)
    print(f"Already processed apps: {len(processed_appids)}")

    if limit is not None:
        apps = apps[:limit]
        print(f"Processing limit set to {limit} apps.")

    new_games = 0
    skipped_apps = 0

    for app in tqdm(apps, desc="Processing apps"):
        appid_str = str(app.get("appid"))
        if appid_str in processed_appids:
            skipped_apps += 1
            continue

        print(f"Processing appid {appid_str}")
        store_data = get_store_data(appid_str)

        # Mark as processed regardless of outcome.
        append_checkpoint(appid_str, checkpoint_file)
        processed_appids.add(appid_str)

        if store_data and store_data.get("type") == "game":
            game_info = {
                "appid": appid_str,
                "name": store_data.get("name"),
                "short_description": store_data.get("short_description"),
                "detailed_description": store_data.get("detailed_description"),
                "release_date": store_data.get("release_date", {}).get("date"),
                "developers": store_data.get("developers"),
                "publishers": store_data.get("publishers"),
                "header_image": store_data.get("header_image"),
                "website": store_data.get("website"),
                "store_data": store_data,  # Full store data
                "reviews": []
            }

            raw_reviews = get_reviews(appid_str)
            print(f"Fetched {len(raw_reviews)} reviews for appid {appid_str}")

            # Skip LLM filtering for now, just save raw reviews (max 100)
            # good_reviews = filter_reviews(raw_reviews, max_reviews=100)
            # print(f"Filtered down to {len(good_reviews)} good reviews for appid {appid_str}")

            # Sort by votes_up and take top 100
            sorted_reviews = sorted(raw_reviews, key=lambda r: r.get("votes_up", 0), reverse=True)
            game_info["reviews"] = sorted_reviews[:100]
            print(f"Saved top {len(game_info['reviews'])} reviews (sorted by votes_up)")

            save_game_data(game_info, output_file)
            new_games += 1
            print(f"Saved game: appid {appid_str} - {store_data.get('name')}")
        else:
            print(f"Skipping appid {appid_str} as it is not a game or store data is unavailable.")

        time.sleep(sleep_time)

    print(f"Finished processing apps. New games summarized: {new_games}. Skipped: {skipped_apps}.")
    print(f"Data saved to {output_file}")
    print(f"Checkpoint file updated: {checkpoint_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steam Games Data Acquisition with Review Filtering and Progress Reporting")
    parser.add_argument("--limit", type=int, help="Limit the number of apps to process (for testing)", default=None)
    parser.add_argument("--sleep", type=float, help="Sleep time (in seconds) between API calls", default=1)
    parser.add_argument("--output", type=str, help="Output JSONL file path", default="steam_games_data.jsonl")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file path", default="processed_ids.txt")
    args = parser.parse_args()
    main(limit=args.limit, sleep_time=args.sleep, output_file=args.output, checkpoint_file=args.checkpoint)
