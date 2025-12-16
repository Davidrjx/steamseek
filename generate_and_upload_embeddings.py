#!/usr/bin/env python3
"""
Generate embeddings for games and upload to Pinecone
Handles rate limiting and errors gracefully
"""

import argparse
import json
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "game-knowledge"
BATCH_SIZE = 10  # Process 10 games at a time
DELAY_BETWEEN_BATCHES = 2  # Wait 2 seconds between batches to avoid rate limits

# Initialize APIs
openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def get_embedding(text: str):
    """Get embedding with retry logic for rate limits"""
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

def create_embedding_text(game):
    """Create text for embedding from game data"""
    parts = []

    if game.get('name'):
        parts.append(f"Game: {game['name']}")

    if game.get('short_description'):
        parts.append(game['short_description'])

    if game.get('genres'):
        parts.append(f"Genres: {', '.join(game['genres'])}")

    if game.get('developers'):
        parts.append(f"Developers: {', '.join(game['developers'])}")

    return " ".join(parts)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate embeddings for games and upload to Pinecone',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'input_file',
        help='Path to the JSONL file containing game data (e.g., data/steam_games_data.jsonl)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Number of games to process in each batch (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--delay',
        type=int,
        default=DELAY_BETWEEN_BATCHES,
        help=f'Seconds to wait between batches (default: {DELAY_BETWEEN_BATCHES})'
    )

    args = parser.parse_args()
    games_file = args.input_file
    batch_size = args.batch_size
    delay = args.delay

    print(f"file: {games_file}, bsize: {batch_size}, delay: {delay}")

    print("="*60)
    print("Generate and Upload Embeddings to Pinecone")
    print("="*60)

    # Load games
    print(f"\nLoading games from {games_file}...")
    games = []
    with open(games_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                game = json.loads(line.strip())
                games.append(game)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(games)} games")

    # Check current index stats
    stats = index.describe_index_stats()
    print(f"\nCurrent index stats:")
    print(f"  Total vectors: {stats.total_vector_count}")
    print(f"  Dimension: {stats.dimension}")

    # Process games in batches
    total_processed = 0
    total_failed = 0
    total_skipped = 0

    for i in range(0, len(games), batch_size):
        batch = games[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(games) + batch_size - 1) // batch_size

        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} games)...")

        # check which games already exist in pinecone
        batch_appids = [str(game.get('appid', '')) for game in batch if game.get('appid')]
        existing_ids = set()

        if batch_appids:
            print(f"  Checking for existing vectors/records in pinecone...")
            try:
                fetch_results = index.fetch(ids=batch_appids, namespace="")
                existing_ids = set(fetch_results.vectors.keys())
                if existing_ids:
                    print(f"  Found {len(existing_ids)} existing vectors/records in pinecone")
            except Exception as e:
                print(f"  warning: checking for existing vectors/records in pinecone: {e}, continuing...")

        vectors_to_upsert = []

        for game in batch:
            try:
                appid = str(game.get('appid', ''))
                name = game.get('name', 'Unknown')

                if not appid:
                    print(f"  ⚠️  Skipping game with no appid")
                    total_failed += 1
                    continue

                print(f"  Processing: {name} (appid: {appid})")

                if appid in existing_ids:
                    print(f"  ✓ Already exists in Pinecone, skipping")
                    total_skipped += 1
                    continue

                # Create text for embedding
                embedding_text = create_embedding_text(game)

                # Get embedding
                print(f"    Getting embedding...")
                embedding = get_embedding(embedding_text)
                print(f"    ✓ Got embedding ({len(embedding)} dimensions)")

                # Prepare vector for Pinecone
                vector_data = {
                    'id': appid,
                    'values': embedding,
                    'metadata': {
                        'appid': appid,
                        'name': name,
                        'ai_summary': game.get('short_description', ''),
                    }
                }
                vectors_to_upsert.append(vector_data)
                total_processed += 1

            except openai.RateLimitError as e:
                print(f"    ❌ Rate limit hit: {e}")
                print(f"    Waiting 60 seconds before retry...")
                time.sleep(60)
                # Retry this game
                try:
                    embedding = get_embedding(embedding_text)
                    vector_data = {
                        'id': appid,
                        'values': embedding,
                        'metadata': {
                            'appid': appid,
                            'name': name,
                            'ai_summary': game.get('short_description', ''),
                        }
                    }
                    vectors_to_upsert.append(vector_data)
                    total_processed += 1
                except Exception as retry_error:
                    print(f"    ❌ Failed after retry: {retry_error}")
                    total_failed += 1

            except Exception as e:
                print(f"    ❌ Error: {e}")
                total_failed += 1
                continue

        # Upload batch to Pinecone
        if vectors_to_upsert:
            print(f"\n  Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
            try:
                index.upsert(vectors=vectors_to_upsert, namespace="")
                print(f"  ✓ Batch uploaded successfully")
            except Exception as e:
                print(f"  ❌ Failed to upload batch: {e}")
                total_failed += len(vectors_to_upsert)
                total_processed -= len(vectors_to_upsert)

        # Wait between batches to avoid rate limits
        if i + batch_size < len(games):
            print(f"\n  Waiting {delay} seconds before next batch...")
            time.sleep(delay)

    # Final stats
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Successfully processed: {total_processed} games")
    print(f"Failed: {total_failed} games, Skipped: {total_skipped} games")

    # Check final index stats
    stats = index.describe_index_stats()
    print(f"\nFinal index stats:")
    print(f"  Total vectors: {stats.total_vector_count}")
    print(f"  Dimension: {stats.dimension}")

    # Create checkpoint file
    with open('pinecone_upload_complete.txt', 'w') as f:
        f.write(f"Upload completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total vectors: {stats.total_vector_count}\n")

    print("\n✓ Checkpoint file created: pinecone_upload_complete.txt")

if __name__ == "__main__":
    main()
