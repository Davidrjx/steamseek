#!/usr/bin/env python3
"""
Generate embeddings for games and upload to Pinecone
Handles rate limiting and errors gracefully
"""

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
GAMES_FILE = "data/steam_games_data.jsonl"
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
    print("="*60)
    print("Generate and Upload Embeddings to Pinecone")
    print("="*60)

    # Load games
    print(f"\nLoading games from {GAMES_FILE}...")
    games = []
    with open(GAMES_FILE, 'r', encoding='utf-8') as f:
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

    for i in range(0, len(games), BATCH_SIZE):
        batch = games[i:i+BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(games) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} games)...")

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
        if i + BATCH_SIZE < len(games):
            print(f"\n  Waiting {DELAY_BETWEEN_BATCHES} seconds before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES)

    # Final stats
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Successfully processed: {total_processed} games")
    print(f"Failed: {total_failed} games")

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
