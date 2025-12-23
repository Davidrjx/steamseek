#!/usr/bin/env python3
"""
Export all appids from Pinecone index to a text file.
"""

import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def export_appids_from_pinecone(
    index_name="game-knowledge",
    output_file="appids_in_pinecone.txt",
    namespace=""
):
    """
    Export all appids from Pinecone index to a text file.

    Args:
        index_name: Name of the Pinecone index
        output_file: Output file path
        namespace: Pinecone namespace (default: "")
    """
    print(f"Connecting to Pinecone index: {index_name}")

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)

    # Get index stats
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    print(f"Total vectors in index: {total_vectors}")

    if total_vectors == 0:
        print("No vectors found in the index!")
        return

    # Collect all appids
    appids = set()

    print("\nFetching all vector IDs from Pinecone...")
    print("Note: This may take a while for large indexes...")

    # Method: Use list_paginated to get all IDs
    # This is more efficient than querying
    try:
        # List all vector IDs with pagination
        print("Using list API to fetch vector IDs...")

        # Get the first page
        results = index.list(namespace=namespace)

        # Process first batch
        for vector_id in results.vector_ids:
            appids.add(vector_id)

        print(f"Fetched {len(appids)} appids so far...")

        # Continue with pagination if there are more
        while results.pagination is not None and results.pagination.next is not None:
            results = index.list(
                namespace=namespace,
                pagination_token=results.pagination.next
            )

            for vector_id in results.vector_ids:
                appids.add(vector_id)

            print(f"Fetched {len(appids)} appids so far...")

        print(f"\nTotal unique appids collected: {len(appids)}")

    except Exception as e:
        print(f"Error using list API: {e}")
        print("\nFalling back to query-based method...")

        # Fallback: Use a dummy query to get all vectors
        # Create a zero vector for querying
        dummy_vector = [0.0] * 3072  # Dimension for text-embedding-3-large

        # Query in batches
        batch_size = 10000
        fetched = 0

        while fetched < total_vectors:
            results = index.query(
                vector=dummy_vector,
                top_k=min(batch_size, total_vectors - fetched),
                include_metadata=False,
                namespace=namespace
            )

            for match in results.matches:
                appids.add(match.id)

            fetched += len(results.matches)
            print(f"Fetched {len(appids)} unique appids so far... ({fetched}/{total_vectors})")

            # If no more results, break
            if len(results.matches) == 0:
                break

        print(f"\nTotal unique appids collected: {len(appids)}")

    # Sort appids for consistent output
    sorted_appids = sorted(appids, key=lambda x: int(x) if x.isdigit() else 0)

    # Write to file
    print(f"\nWriting appids to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for appid in sorted_appids:
            f.write(f"{appid}\n")

    print(f"Successfully exported {len(sorted_appids)} appids to {output_file}")

    # Print some statistics
    print("\n" + "="*60)
    print("Export Summary:")
    print(f"  Index name: {index_name}")
    print(f"  Namespace: '{namespace}'")
    print(f"  Total vectors in index: {total_vectors}")
    print(f"  Unique appids exported: {len(sorted_appids)}")
    print(f"  Output file: {output_file}")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export all appids from Pinecone index to a text file"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="game-knowledge",
        help="Pinecone index name (default: game-knowledge)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="appids_in_pinecone.txt",
        help="Output file path (default: appids_in_pinecone.txt)"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="",
        help="Pinecone namespace (default: empty string)"
    )

    args = parser.parse_args()

    export_appids_from_pinecone(
        index_name=args.index,
        output_file=args.output,
        namespace=args.namespace
    )
