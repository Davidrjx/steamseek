import requests
import json
import time
import argparse
import os
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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
def get_app_list(max_results=50000, last_appid=0):
    """
    Fetch a batch of Steam apps using the Steam Web API.
    This function makes a single API call and returns one page of results.

    Args:
        max_results: Maximum number of apps to fetch per call (default: 50000)
        last_appid: Continue from this appid (for pagination)

    Returns:
        tuple: (apps_list, has_more_results, last_appid)
            - apps_list: List of app dictionaries with keys "appid" and "name"
            - has_more_results: Boolean indicating if more results are available
            - last_appid: The last appid in this batch (for next pagination call)
    """
    # Get API key from environment
    steam_web_api_key = os.getenv("STEAM_API_KEY")

    if not steam_web_api_key:
        print("Error: STEAM_API_KEY not found in environment variables")
        return ([], False, 0)

    url = f"https://api.steampowered.com/IStoreService/GetAppList/v1/"
    params = {
        "key": steam_web_api_key,
        "max_results": max_results,
        "include_games": 1,
        "include_dlc": 1,
        "include_software": 1,
        "include_videos": 1,
        "include_hardware": 1
    }

    if last_appid > 0:
        params["last_appid"] = last_appid

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Parse response
        # Response format: {"response": {"apps": [...], "have_more_results": true/false, "last_appid": xxx}}
        apps_data_res = data.get("response", {})
        if not apps_data_res:
            print("Warning: No response data from Steam GetAppList Web API")
            return ([], False, 0)

        apps_data = apps_data_res.get("apps", [])
        if not apps_data:
            print("No apps data returned from Steam GetAppList Web API")
            return ([], False, 0)

        has_more_results = apps_data_res.get("have_more_results", False)
        last_appid = apps_data_res.get("last_appid", 0)

        # Convert to standard format
        apps = []
        for app in apps_data:
            apps.append({
                "appid": app.get("appid"),
                "name": app.get("name", "")
            })

        print(f"Fetched {len(apps)} apps. Has more: {has_more_results}, Last appid: {last_appid}")
        return (apps, has_more_results, last_appid)

    except Exception as e:
        print(f"Error fetching app list: {e}")
        import traceback
        traceback.print_exc()
        return ([], False, 0)


def main(limit=None, sleep_time=1, output_file="steam_games_data.jsonl", checkpoint_file="processed_ids.txt", batch_size=10000, max_workers=10):
    """
    Main function to fetch and process Steam games with external pagination loop and concurrent processing.

    Args:
        limit: Maximum number of apps to process (None for all)
        sleep_time: Sleep time between processing each app
        output_file: Output JSONL file path
        checkpoint_file: Checkpoint file to track processed apps
        batch_size: Number of apps to fetch per API call
        max_workers: Maximum number of concurrent threads (default: 10)
    """
    # Load already processed app IDs
    processed_appids = load_processed_appids(checkpoint_file)
    print(f"Already processed apps: {len(processed_appids)}")

    new_games = 0
    skipped_apps = 0
    total_processed = 0
    page = 1
    last_appid = 0

    # External pagination loop
    while True:
        # Check limit
        if limit is not None and total_processed+1 > limit:
            print(f"\nReached processing limit of {limit} apps.")
            print(f"Finished processing. New games: {new_games}, Skipped: {skipped_apps}")
            return
        # 
        print(f"\n{'='*60}")
        print(f"Fetching batch {page} (starting from appid {last_appid})...")
        print(f"{'='*60}\n")

        # Fetch one batch from API
        apps_batch, has_more, last_appid = get_app_list(max_results=batch_size, last_appid=last_appid)

        if not apps_batch:
            print("No apps returned. Stopping.")
            break

        # Calculate actual workers before printing
        # Reduce concurrency to avoid triggering Steam's rate limiting (max 5 workers recommended)
        actual_workers = min(min(max_workers, 5), len(apps_batch))  # Limit to 5 threads max
        if actual_workers != max_workers:
            print(f"Note: Limiting concurrent threads to {actual_workers} to avoid rate limiting")
        print(f"Processing batch of {len(apps_batch)} apps with {actual_workers} concurrent threads...")

        # Create thread-safe locks
        checkpoint_lock = Lock()
        file_lock = Lock()

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all tasks
            future_to_app = {
                executor.submit(
                    process_single_app,
                    app,
                    processed_appids,
                    output_file,
                    checkpoint_file,
                    checkpoint_lock,
                    file_lock,
                    sleep_time
                ): app for app in apps_batch
            }

            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_app), total=len(apps_batch), desc=f"Processing batch {page}"):
                app = future_to_app[future]
                try:
                    appid_str, status, is_game = future.result()

                    # Update processed set (thread-safe)
                    processed_appids.add(appid_str)

                    if status == 'skipped':
                        skipped_apps += 1
                    elif status == 'success':
                        new_games += 1
                        total_processed += 1
                    elif status == 'not_game':
                        total_processed += 1
                    elif status == 'error':
                        total_processed += 1

                except Exception as e:
                    appid = app.get("appid")
                    print(f"Exception for appid {appid}: {e}")

        print(f"\nBatch {page} complete. Total new games so far: {new_games}, Skipped: {skipped_apps}")
        page += 1

        # Check if there are more results
        if not has_more:
            print("\nNo more results available from Steam API.")
            break

        # Small delay between API calls to avoid rate limiting
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"All batches processed!")
    print(f"Final stats - New games: {new_games}, Skipped: {skipped_apps}")
    print(f"Data saved to: {output_file}")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"{'='*60}")


def get_store_data(appid, country="us", language="en", max_retries=3, retry_delay=2):
    """
    Fetch store details for a given appid using the Steam Storefront API.
    Returns the 'data' dictionary if successful; otherwise, returns None.

    Args:
        appid: Steam app ID
        country: Country code (default: "us")
        language: Language code (default: "en")
        max_retries: Maximum number of retry attempts for 403 errors
        retry_delay: Base delay in seconds between retries (uses exponential backoff)
    """
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc={country}&l={language}"

    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            # Handle non 200 errors with retry
            if response.status_code in  [403, 429, 500, 504, 502, 503]:
                print(f"  Fetching store data for appid {appid} (attempt {attempt + 1}/{max_retries}), status {response.status_code}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"  ⚠ {response.status_code} error for appid {appid}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Max retries reached, silent fail
                    return None

            # Check if response has content
            if not response.content:
                # Silent fail for empty responses
                return None

            # Try to parse JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                # Silent fail for non-JSON responses (HTML error pages, etc.)
                return None
            except ValueError:
                # Silent fail for other JSON parsing errors
                return None

            # Check if data is None or not a dict
            if data is None or not isinstance(data, dict):
                return None

            if str(appid) in data and data[str(appid)].get("success"):
                return data[str(appid)].get("data", {})
            else:
                return None

        except requests.exceptions.Timeout:
            # Silent fail for timeouts
            return None
        except requests.exceptions.RequestException:
            # Silent fail for network errors
            return None
        except Exception as e:
            # Only log unexpected errors
            print(f"Unexpected error for appid {appid}: {e}")
            return None

    return None

def get_reviews(appid, num_per_page=100, max_retries=3, retry_delay=2):
    """
    Fetch reviews for a given appid using Steam's reviews endpoint.
    Returns a list of review objects (may be empty if none available).

    Args:
        appid: Steam app ID
        num_per_page: Number of reviews to fetch per page
        max_retries: Maximum number of retry attempts for 403 errors
        retry_delay: Base delay in seconds between retries (uses exponential backoff)
    """
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1&num_per_page={num_per_page}&language=english"

    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)

            # Handle 403 errors with retry
            if response.status_code != 200:
                print(f"  Fetching reviews for appid {appid} (attempt {attempt + 1}/{max_retries}), status {response.status_code}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"  ⚠ {response.status_code} error fetching reviews for appid {appid}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Max retries reached, return empty list
                    return []

            # Check if response has content
            if not response.content:
                return []

            # Try to parse JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                return []
            except ValueError:
                return []

            # Check if data is None or not a dict
            if data is None or not isinstance(data, dict):
                return []

            reviews = data.get("reviews", [])
            return reviews

        except requests.exceptions.Timeout:
            return []
        except requests.exceptions.RequestException:
            return []
        except Exception as e:
            print(f"Unexpected error fetching reviews for appid {appid}: {e}")
            return []

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

def append_checkpoint(appid, checkpoint_file, lock=None):
    """
    Append an appid to the checkpoint file.
    Thread-safe when lock is provided.
    """
    try:
        if lock:
            with lock:
                with open(checkpoint_file, "a", encoding="utf-8") as f:
                    f.write(str(appid) + "\n")
                    f.flush()
        else:
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(str(appid) + "\n")
                f.flush()
    except Exception as e:
        print(f"Error writing appid {appid} to checkpoint: {e}")

def process_single_app(app, processed_appids, output_file, checkpoint_file, checkpoint_lock, file_lock, sleep_time=0):
    """
    Process a single app: fetch store data and reviews, save to file.
    Thread-safe function for concurrent execution.

    Returns:
        tuple: (appid_str, status, is_game)
            - appid_str: The app ID as string
            - status: 'skipped', 'not_game', 'success', or 'error'
            - is_game: Boolean indicating if it's a game
    """
    appid_str = str(app.get("appid"))

    # Check if already processed (thread-safe read from set)
    if appid_str in processed_appids:
        return (appid_str, 'skipped', False)

    try:
        # Fetch store data
        store_data = get_store_data(appid_str)

        # Mark as processed regardless of outcome
        append_checkpoint(appid_str, checkpoint_file, lock=checkpoint_lock)

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
                "store_data": store_data,
                "reviews": []
            }

            raw_reviews = get_reviews(appid_str)

            # Sort by votes_up and take top 100
            sorted_reviews = sorted(raw_reviews, key=lambda r: r.get("votes_up", 0), reverse=True)
            game_info["reviews"] = sorted_reviews[:100]

            # Save game data (thread-safe file write)
            if file_lock:
                with file_lock:
                    save_game_data(game_info, output_file)
            else:
                save_game_data(game_info, output_file)

            # Add delay to avoid rate limiting (minimum 0.5s recommended)
            delay = max(sleep_time, 0.5)
            time.sleep(delay)

            return (appid_str, 'success', True)
        else:
            return (appid_str, 'not_game', False)

    except Exception as e:
        print(f"Error processing appid {appid_str}: {e}")
        return (appid_str, 'error', False)

########################################
# MAIN DATA ACQUISITION LOGIC
########################################
def main_bak(limit=None, sleep_time=1, output_file="steam_games_data.jsonl", checkpoint_file="processed_ids.txt"):
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
    parser = argparse.ArgumentParser(description="Steam Games Data Acquisition with Concurrent Processing")
    parser.add_argument("--limit", type=int, help="Limit the number of apps to process (for testing)", default=None)
    parser.add_argument("--sleep", type=float, help="Sleep time (in seconds) between processing each app", default=0)
    parser.add_argument("--output", type=str, help="Output JSONL file path", default="steam_games_data.jsonl")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file path", default="processed_ids.txt")
    parser.add_argument("--batch-size", type=int, help="Number of apps to fetch per API call", default=10000)
    parser.add_argument("--workers", type=int, help="Number of concurrent threads (default: 10)", default=10)
    args = parser.parse_args()
    main(
        limit=args.limit,
        sleep_time=args.sleep,
        output_file=args.output,
        checkpoint_file=args.checkpoint,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
