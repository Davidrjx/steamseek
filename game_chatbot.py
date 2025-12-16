import json
import os
import logging
import time
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
import openai
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Configure logging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s [%(levelname)s] %(message)s')

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Use OpenAI's official API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Use OpenRouter as fallback

# Configure file paths
EMBEDDINGS_FILE = "embeddings.jsonl"  # Your embeddings file in root folder
UPLOAD_CHECKPOINT_FILE = "pinecone_upload_complete.txt"  # To track if upload is done

# Set OpenAI API key for official embedding endpoint
openai.api_key = OPENAI_API_KEY

# Check which API to use based on availability
USE_OPENROUTER = not OPENAI_API_KEY or len(OPENAI_API_KEY.strip()) == 0

class GameKnowledgeBase:
    def __init__(self, index_name: str = "game-knowledge"):
        """Initialize the knowledge base with Pinecone index."""
        self.index_name = index_name
        # Use dimension 3072 to match the text-embedding-3-large model
        self.dimension = 3072  
        logging.info("Initializing GameKnowledgeBase with index '%s'", self.index_name)
        self.initialize_index()
        
    def initialize_index(self):
        """Create or connect to Pinecone index."""
        logging.debug("Connecting to Pinecone with API key.")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # List existing indexes
        existing_indexes = self.pc.list_indexes()
        index_exists = self.index_name in [index.name for index in existing_indexes]
        
        if not index_exists:
            logging.info("Index '%s' does not exist. Creating new index...", self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logging.info("Index created successfully.")
        else:
            logging.info("Connecting to existing index '%s'...", self.index_name)
        
        self.index = self.pc.Index(self.index_name)
        logging.debug("Index '%s' is ready.", self.index_name)

    def check_if_uploaded(self) -> bool:
        """Check if embeddings have already been uploaded."""
        if os.path.exists(UPLOAD_CHECKPOINT_FILE):
            logging.debug("Upload checkpoint file found: %s", UPLOAD_CHECKPOINT_FILE)
            return True
        try:
            stats = self.index.describe_index_stats(namespace="")

            total_vectors = stats.total_vector_count
            logging.debug("Total vectors in index: %d", total_vectors)
            return total_vectors > 0
        except Exception as e:
            logging.error("Error checking index stats: %s", e)
            return False

    def upload_embeddings(self, embeddings_file: str = EMBEDDINGS_FILE, batch_size: int = 100):
        """Upload embeddings from file to Pinecone in batches."""
        logging.info("Starting embeddings upload from %s", embeddings_file)
        
        if self.check_if_uploaded():
            logging.info("Embeddings already uploaded to Pinecone. Skipping upload.")
            return

        total_uploaded = 0
        batch = []

        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    
                    # Include the ai_summary field in metadata.
                    vector_record = {
                        'id': record['appid'],
                        'values': record['embedding'],
                        'metadata': {
                            'name': record['name'],
                            'appid': record['appid'],
                            'ai_summary': record.get('ai_summary', 'No summary available'),
                            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    }
                    batch.append(vector_record)
                    
                    if len(batch) >= batch_size:
                        logging.debug("Uploading batch of %d vectors", len(batch))
                        self.index.upsert(vectors=batch, namespace="")

                        total_uploaded += len(batch)
                        logging.info("Uploaded %d vectors so far", total_uploaded)
                        batch = []
                
                if batch:
                    logging.debug("Uploading final batch of %d vectors", len(batch))
                    self.index.upsert(vectors=batch, namespace="")

                    total_uploaded += len(batch)
            
            with open(UPLOAD_CHECKPOINT_FILE, 'w') as f:
                f.write(f"Upload completed on {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info("Upload complete. Total vectors uploaded: %d", total_uploaded)
            
        except FileNotFoundError:
            logging.error("Embeddings file not found at %s", embeddings_file)
            raise
        except Exception as e:
            logging.error("Error during upload: %s", e)
            raise

    def update_embeddings(self, embeddings_file: str = EMBEDDINGS_FILE, batch_size: int = 100):
        """Update the Pinecone index with new or updated embeddings."""
        logging.info("Starting embeddings update from %s", embeddings_file)
        
        try:
            current_stats = self.index.describe_index_stats()
            logging.info("Current vectors in index: %d", current_stats.total_vector_count)
        except Exception as e:
            logging.error("Error retrieving current index stats: %s", e)
            current_stats = None
        
        total_processed = 0
        batch = []
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    
                    # Include ai_summary in metadata during update as well.
                    vector_record = {
                        'id': record['appid'],
                        'values': record['embedding'],
                        'metadata': {
                            'name': record['name'],
                            'appid': record['appid'],
                            'ai_summary': record.get('ai_summary', 'No summary available'),
                            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    }
                    batch.append(vector_record)
                    
                    if len(batch) >= batch_size:
                        logging.debug("Processing batch of %d vectors", len(batch))
                        self.index.upsert(vectors=batch, namespace="")

                        total_processed += len(batch)
                        logging.info("Processed batch of %d vectors", len(batch))
                        batch = []
                
                if batch:
                    logging.debug("Processing final batch of %d vectors", len(batch))
                    self.index.upsert(vectors=batch, namespace="")

                    total_processed += len(batch)
            
            new_stats = self.index.describe_index_stats()
            logging.info("Update complete!")
            if current_stats:
                logging.info("Previous vector count: %d", current_stats.total_vector_count)
            logging.info("New vector count: %d", new_stats.total_vector_count)
            logging.info("Total vectors processed: %d", total_processed)
            
            with open(UPLOAD_CHECKPOINT_FILE, 'w') as f:
                f.write(f"Last updated on {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except FileNotFoundError:
            logging.error("Embeddings file not found at %s", embeddings_file)
            raise
        except Exception as e:
            logging.error("Error during update: %s", e)
            raise

    def reset_index(self):
        """Delete and recreate the index for a fresh start."""
        logging.info("Resetting index '%s'", self.index_name)
        try:
            self.pc.delete_index(self.index_name)
            logging.info("Index '%s' deleted successfully.", self.index_name)
            if os.path.exists(UPLOAD_CHECKPOINT_FILE):
                os.remove(UPLOAD_CHECKPOINT_FILE)
                logging.debug("Removed checkpoint file: %s", UPLOAD_CHECKPOINT_FILE)
            
            self.initialize_index()
            logging.info("Index reset complete. Ready for fresh upload.")
        except Exception as e:
            logging.error("Error resetting index: %s", e)
            raise

class GameChatbot:
    def __init__(self, knowledge_base: GameKnowledgeBase):
        """Initialize chatbot with knowledge base and OpenAI client for chat completions."""
        self.kb = knowledge_base
        logging.info("Initializing GameChatbot")
        # self.chat_model = "gpt-4o-mini"  # Ensure this model is supported or change accordingly.
        # Instantiate the new OpenAI client
        import httpx

        # Get proxy from environment
        http_proxy = os.getenv('http_proxy') or os.getenv('HTTP_PROXY')
        https_proxy = os.getenv('https_proxy') or os.getenv('HTTPS_PROXY')

        if http_proxy or https_proxy:
            # Use mounts for separate HTTP/HTTPS proxies
            # Note: HTTPS proxy should still use http:// scheme (not https://)
            proxy_mounts = {
                "http://": httpx.HTTPTransport(proxy=http_proxy) if http_proxy else httpx.HTTPTransport(),
                "https://": httpx.HTTPTransport(proxy=https_proxy or http_proxy) if (https_proxy or http_proxy) else httpx.HTTPTransport(),
            }
            http_client = httpx.Client(mounts=proxy_mounts, timeout=60.0)
            self.openai_client = openai.OpenAI(http_client=http_client)
        else:
            self.openai_client = openai.OpenAI(timeout=60.0)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI or OpenRouter API."""
        logging.debug("Requesting embedding for text: %s", text)
        try:
            if USE_OPENROUTER:
                # Use OpenRouter API (supports global access)
                logging.debug("Using OpenRouter for embeddings")
                response = requests.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "openai/text-embedding-3-large",
                        "input": text
                    }
                )
                response.raise_for_status()
                result = response.json()

                # Debug: print the response to understand the structure
                logging.debug("OpenRouter response: %s", str(result)[:200])

                # Check if there's an error in the response
                if 'error' in result:
                    error_msg = result.get('error', {})
                    logging.error("OpenRouter error: %s", error_msg)
                    raise Exception(f"OpenRouter API error: {error_msg}")

                # Try to extract embedding from response
                if 'data' in result and len(result['data']) > 0:
                    embedding = result['data'][0]['embedding']
                elif 'embedding' in result:
                    embedding = result['embedding']
                else:
                    logging.error("Unexpected response format: %s", result)
                    raise Exception(f"Unexpected OpenRouter response format: {result}")

                logging.debug("Received embedding of length %d", len(embedding))
                return embedding
            else:
                # Use OpenAI directly
                logging.debug("Using OpenAI directly for embeddings")
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-large",
                    input=text,
                    encoding_format="float"
                )
                # Use attribute access for the response.
                embedding = response.data[0].embedding
                logging.debug("Received embedding of length %d", len(embedding))
                return embedding
        except Exception as e:
            logging.error("Error getting embedding: %s", e)
            raise

    def search_games(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant games using the query."""
        logging.info("Searching games for query: '%s'", query)
        print(f"\n=== SEARCH_GAMES called with query: '{query}', top_k: {top_k} ===")
        try:
            print("Step 1: Getting query embedding...")
            query_embedding = self.get_embedding(query)
            print(f"Step 1 DONE: Got embedding with {len(query_embedding)} dimensions")

            print("Step 2: Querying Pinecone index...")
            results = self.kb.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=""  # Explicitly set namespace
            )
            print(f"Step 2 DONE: Pinecone returned {len(results.matches)} matches")

            if len(results.matches) == 0:
                print("WARNING: No matches found in Pinecone!")
            else:
                print(f"First match: {results.matches[0].metadata.get('name', 'Unknown')} (score: {results.matches[0].score})")

            logging.debug("Search returned %d matches", len(results.matches))
            return results.matches
        except Exception as e:
            logging.error("Error during game search: %s", e)
            print(f"ERROR in search_games: {e}")
            import traceback
            traceback.print_exc()
            raise

    # def chat(self, user_message: str) -> str:
    #     """Generate a response using the chatbot."""
    #     print("Chat initiated with user message: '%s'", user_message)
    #     system_message = (
    #         "You are a knowledgeable gaming assistant. When users ask about games, you should:\n"
    #         "1. Analyze their question to understand what information you need\n"
    #         "2. Use the search results from the game database to provide accurate information\n"
    #         "3. Cite specific games when providing examples or recommendations\n"
    #         "4. Reference specific features, mechanics, and content from the game summaries\n"
    #         "Remember to be helpful and informative, and always base your responses on the actual game data provided."
    #     )

    #     try:
    #         relevant_games = self.search_games(user_message)
    #         context = "Here are some relevant games from our database:\n\n"
    #         for game in relevant_games:
    #             context += f"Game: {game.metadata['name']} (ID: {game.metadata['appid']})\n"
    #             context += f"Summary: {game.metadata.get('ai_summary', 'No summary available')}\n\n"
    #             context += "---\n\n"  # Separator between games for clarity

    #         logging.debug("Context for chatbot: %s", context)
            
    #         response = self.openai_client.chat.completions.create(
    #             model=self.chat_model,
    #             messages=[
    #                 {"role": "system", "content": system_message},
    #                 {"role": "user", "content": f"Context:\n{context}\n\nUser question: {user_message}"}
    #             ]
    #         )
    #         chatbot_response = response.choices[0].message.content
    #         logging.info("Chatbot response received")
    #         return chatbot_response
    #     except Exception as e:
    #         logging.error("Error in chat processing: %s", e)
    #         raise

def chat_loop(chatbot):
    """
    Handles the interactive chat loop with the user.
    This function:
    1. Takes user input
    2. Sends it to the chatbot
    3. Displays the response
    4. Continues until the user types 'quit'
    """
    logging.info("Starting chat loop")
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                logging.info("User requested to quit chat.")
                print("\nThank you for chatting! Goodbye!")
                break
            
            if not user_input:
                logging.warning("Empty input received, prompting user again.")
                print("Please enter a question or type 'quit' to exit.")
                continue
            
            try:
                logging.debug("Sending user input to chatbot")
                print("\nThinking...")
                response = chatbot.chat(user_input)
                print("\nAssistant:", response)
            except Exception as e:
                logging.error("Error getting response: %s", e)
                print("\nError getting response:", e)
                print("Please try again or type 'quit' to exit.")
    except KeyboardInterrupt:
        logging.info("Chat loop interrupted by user.")
        print("\n\nChat interrupted by user. Goodbye!")

def semantic_search_query(query: str, top_k: int = 10):
    """
    A simple helper that:
    1. Instantiates our GameKnowledgeBase and GameChatbot
    2. Performs a semantic search for 'query' via chatbot.search_games()
    3. Returns a list of dictionaries with the top matches
    """
    print(f"\n{'='*60}")
    print(f"SEMANTIC_SEARCH_QUERY called")
    print(f"Query: '{query}'")
    print(f"Top K: {top_k}")
    print(f"{'='*60}\n")

    print("Creating GameKnowledgeBase...")
    kb = GameKnowledgeBase()
    print("Creating GameChatbot...")
    chatbot = GameChatbot(kb)

    print(f"Calling chatbot.search_games with query: '{query}'")
    pinecone_results = chatbot.search_games(query, top_k=top_k)
    print(f"Received {len(pinecone_results)} results from Pinecone")

    results = []
    for i, match in enumerate(pinecone_results):
        meta = match.metadata
        appid = meta.get('appid')
        name = meta.get('name')
        ai_summary = meta.get('ai_summary', '')
        score = match.score  # similarity

        if i < 3:  # Print first 3 results
            print(f"  Result {i+1}: {name} (appid: {appid}, score: {score:.4f})")

        results.append({
            "appid": appid,
            "name": name,
            "ai_summary": ai_summary,
            "similarity_score": score
        })

    # Sort by descending similarity score
    results.sort(key=lambda x: x["similarity_score"], reverse=True)

    print(f"\n{'='*60}")
    print(f"SEMANTIC_SEARCH_QUERY completed")
    print(f"Total results: {len(results)}")
    if len(results) == 0:
        print("⚠️  WARNING: Returning EMPTY results!")
    print(f"{'='*60}\n")

    return results


def main():
    logging.info("Starting main program")
    kb = GameKnowledgeBase()
    
    while True:
        print("\n=== Game Knowledge Base Management ===")
        print("1. Upload fresh embeddings (deletes existing data)")
        print("2. Update existing embeddings")
        print("3. Start chat")
        print("4. Show database stats")
        print("5. Exit")
        
        choice = input("\nChoose an operation (1-5): ").strip()
        
        if choice == "1":
            try:
                kb.reset_index()
                kb.upload_embeddings()
            except Exception as e:
                logging.error("Error during fresh upload: %s", e)
        elif choice == "2":
            try:
                kb.update_embeddings()
            except Exception as e:
                logging.error("Error during embeddings update: %s", e)
        elif choice == "3":
            try:
                chatbot = GameChatbot(kb)
                stats = kb.index.describe_index_stats()
                print("\nPinecone index status:")
                print(f"Total vectors in index: {stats.total_vector_count}")
                print("\nChat with the gaming assistant (type 'quit' to exit)")
                chat_loop(chatbot)
            except Exception as e:
                logging.error("Error during chat initiation: %s", e)
                print("Error starting chat:", e)
        elif choice == "4":
            try:
                stats = kb.index.describe_index_stats()
                print("\nCurrent Database Statistics:")
                print(f"Total vectors: {stats.total_vector_count}")
                if os.path.exists(UPLOAD_CHECKPOINT_FILE):
                    with open(UPLOAD_CHECKPOINT_FILE, 'r') as f:
                        last_update = f.read()
                    print(f"Last update: {last_update}")
            except Exception as e:
                logging.error("Error retrieving database stats: %s", e)
                print("Error retrieving database stats:", e)
        elif choice == "5":
            logging.info("User selected exit. Terminating program.")
            print("\nGoodbye!")
            break
        else:
            logging.warning("Invalid choice entered: %s", choice)
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
