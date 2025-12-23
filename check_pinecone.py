#!/usr/bin/env python3
"""Check Pinecone index status and data"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "game-knowledge"

print("="*60)
print("Checking Pinecone Index Status")
print("="*60)

# Connect to Pinecone
print(f"\nConnecting to Pinecone with API key: {PINECONE_API_KEY[:10]}...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# List all indexes
print("\nListing all indexes...")
indexes = pc.list_indexes()
print(f"Found {len(indexes)} index(es):")
for idx in indexes:
    print(f"  - {idx.name}")

# Check if our index exists
if INDEX_NAME not in [idx.name for idx in indexes]:
    print(f"\n❌ ERROR: Index '{INDEX_NAME}' does not exist!")
    print("You need to create the index or upload embeddings first.")
    exit(1)

print(f"\n✓ Index '{INDEX_NAME}' exists")

# Get index reference
index = pc.Index(INDEX_NAME)

# Get index stats
print(f"\nGetting stats for index '{INDEX_NAME}'...")
stats = index.describe_index_stats()

print("\nIndex Statistics:")
print(f"  Total vector count: {stats.total_vector_count}")
print(f"  Dimension: {stats.dimension}")
print(f"  Index fullness: {stats.index_fullness}")

if hasattr(stats, 'namespaces') and stats.namespaces:
    print(f"\nNamespaces:")
    for ns_name, ns_stats in stats.namespaces.items():
        ns_display = f"'{ns_name}'" if ns_name else "'default/empty'"
        print(f"  {ns_display}: {ns_stats.vector_count} vectors")
else:
    print(f"\n⚠️  No namespace information available")

# Test a query with a sample embedding
print("\n" + "="*60)
print("Testing Query")
print("="*60)

if stats.total_vector_count == 0:
    print("\n❌ Index is EMPTY! No vectors to query.")
    print("You need to upload embeddings to the index first.")
else:
    # Try to query with a simple test vector
    print(f"\nTrying to query with a test vector (3072 dimensions)...")
    test_vector = [0.1] * 3072  # Simple test vector

    try:
        results = index.query(
            vector=test_vector,
            top_k=5,
            include_metadata=True,
            namespace=""  # Query the default namespace
        )

        print(f"✓ Query successful!")
        print(f"  Returned {len(results.matches)} matches")

        if len(results.matches) > 0:
            print(f"\nSample matches:")
            for i, match in enumerate(results.matches[:3]):
                name = match.metadata.get('name', 'Unknown')
                appid = match.metadata.get('appid', 'Unknown')
                score = match.score
                print(f"  {i+1}. {name} (appid: {appid}, score: {score:.4f})")
        else:
            print("\n⚠️  Query returned 0 matches even though index has vectors!")
            print("This suggests a namespace mismatch or data issue.")

    except Exception as e:
        print(f"\n❌ Query failed with error: {e}")

print("\n" + "="*60)
print("Check Complete")
print("="*60)
