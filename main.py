import json
import os
import time
from typing import Dict, Any
from review_filtering import ReviewFilter
from summarization import GameSummarizer
from config import *

def load_processed_ids() -> set:
    """
    Loads already processed app IDs from checkpoint file.
    """
    processed_ids = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            processed_ids = set(line.strip() for line in f)
    return processed_ids

def save_processed_id(app_id: str):
    """
    Saves an app ID to the checkpoint file.
    """
    with open(CHECKPOINT_FILE, 'a') as f:
        f.write(f"{app_id}\n")

def save_game_summary(game_data: Dict[str, Any]):
    """
    Saves processed game data to the summaries file.
    """
    with open(SUMMARIES_FILE, 'a', encoding='utf-8') as f:
        json.dump(game_data, f, ensure_ascii=False)
        f.write('\n')  # Add newline for JSONL format

def process_game(game_data: Dict[str, Any], review_filter: ReviewFilter, summarizer: GameSummarizer) -> Dict[str, Any]:
    """
    Processes a single game by filtering reviews and generating a summary.
    """
    try:
        # Extract the game ID
        app_id = str(game_data.get('appid', ''))
        if not app_id:
            print("Warning: Game data missing app ID")
            return None

        # Get and filter reviews
        reviews = game_data.get('reviews', [])
        print(f"Processing {len(reviews)} reviews for game {app_id}")
        
        filtered_reviews = review_filter.filter_reviews(reviews)
        print(f"Filtered down to {len(filtered_reviews)} quality reviews")
        
        # Update game data with filtered reviews
        game_data['filtered_reviews'] = filtered_reviews

        # Generate summary
        if filtered_reviews or game_data.get('detailed_description'):
            summary = summarizer.summarize_game(game_data)
            if summary:
                game_data['summary'] = summary
                print(f"Successfully generated summary for game {app_id}")
            else:
                print(f"Failed to generate summary for game {app_id}")
        else:
            print(f"Insufficient content for summary generation for game {app_id}")

        return game_data

    except Exception as e:
        print(f"Error processing game {game_data.get('appid')}: {e}")
        return None

def main():
    """
    Main function to process all games.
    """
    # Initialize components
    review_filter = ReviewFilter()
    summarizer = GameSummarizer()
    
    # Load already processed IDs
    processed_ids = load_processed_ids()
    print(f"Found {len(processed_ids)} already processed games")

    # Process games from input file
    games_processed = 0
    games_failed = 0

    try:
        with open(GAMES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Parse game data
                    game_data = json.loads(line.strip())
                    app_id = str(game_data.get('appid', ''))

                    # Skip if already processed
                    if app_id in processed_ids:
                        print(f"Skipping already processed game {app_id}")
                        continue

                    print(f"\nProcessing game {app_id}: {game_data.get('name', 'Unknown')}")
                    
                    # Process the game
                    processed_data = process_game(game_data, review_filter, summarizer)
                    
                    if processed_data:
                        # Save results
                        save_game_summary(processed_data)
                        save_processed_id(app_id)
                        games_processed += 1
                        print(f"Successfully processed game {app_id}")
                    else:
                        games_failed += 1
                        print(f"Failed to process game {app_id}")

                    # Add a small delay to avoid overwhelming APIs
                    time.sleep(1)

                except json.JSONDecodeError as e:
                    print(f"Error parsing game data: {e}")
                    games_failed += 1
                    continue
                except Exception as e:
                    print(f"Unexpected error processing game: {e}")
                    games_failed += 1
                    continue

    except FileNotFoundError:
        print(f"Input file {GAMES_FILE} not found!")
        return
    except Exception as e:
        print(f"Fatal error: {e}")
        return
    finally:
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {games_processed} games")
        print(f"Failed to process: {games_failed} games")

if __name__ == "__main__":
    main()