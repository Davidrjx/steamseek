import nltk
nltk.download('punkt_tab', quiet=True)

import json
import os
import time
import logging
import random
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import concurrent.futures
import itertools
from openai import OpenAI, RateLimitError, APIError
import signal
import sys
from dotenv import load_dotenv
from review_filtering import ReviewFilter

@dataclass
class ProcessingState:
    """Track processing state for resumability"""
    last_processed_game: Optional[str] = None
    processed_count: int = 0
    failed_games: List[str] = None
    start_time: float = None

    def __post_init__(self):
        self.failed_games = self.failed_games or []
        self.start_time = self.start_time or time.time()

class GameSummarizer:
    def __init__(self,
                 input_file: str,
                 output_file: str,
                 state_file: str,
                 openrouter_key: Optional[str] = None,
                 model: str = "mistralai/mistral-nemo",
                 max_workers: int = 5):
        # Save file paths and settings to instance variables
        self.input_file = input_file
        self.output_file = output_file
        self.state_file = state_file
        self.model = model
        self.max_workers = max_workers

        # Load environment variables
        load_dotenv()

        # Get API key with fallback to environment variable
        api_key = openrouter_key or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key must be provided either via --api-key argument or OPENROUTER_API_KEY environment variable")

        # Initialize OpenAI client with OpenRouter base URL
        # Note: Using synchronous client as we're using threads
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Initialize review filter
        self.review_filter = ReviewFilter()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('game_summarizer_threaded.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Load or initialize state
        self.state = self._load_state()

        # Setup interrupt handling
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _load_state(self) -> ProcessingState:
        """Load processing state from file or create new state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return ProcessingState(**data)
            except Exception as e:
                self.logger.warning(f"Error loading state file: {e}. Starting fresh.")
        return ProcessingState()

    def _save_state(self):
        """Save current processing state to file."""
        with open(self.state_file, 'w') as f:
            json.dump({
                'last_processed_game': self.state.last_processed_game,
                'processed_count': self.state.processed_count,
                'failed_games': self.state.failed_games,
                'start_time': self.state.start_time
            }, f)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals by saving state and exiting gracefully."""
        self.logger.info("\nInterrupt received. Saving state and exiting...")
        self._save_state()
        stats = self._get_statistics()
        self.logger.info("Final Statistics:")
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
        sys.exit(0)

    def _filter_reviews(self, reviews: List[Dict]) -> List[Dict]:
        """
        Filter reviews using the ReviewFilter class.
        Returns the top filtered reviews sorted by quality score.
        """
        return self.review_filter.filter_reviews(reviews)

    def _prepare_summary_prompt(self, game_data: Dict) -> str:
        """
        Prepare prompt for the LLM to generate game summary.
        Includes game metadata and filtered reviews.
        """
        # Get basic game info
        name = game_data.get('name', 'Unknown Game')
        description = game_data.get('short_description', '')
        detailed_desc = game_data.get('detailed_description', '')

        # Filter and get top reviews
        filtered_reviews = self._filter_reviews(game_data.get('reviews', []))
        top_reviews = sorted(filtered_reviews,
                           key=lambda x: x.get('quality_score', 0),
                           reverse=True)[:5]  # Use top 5 highest quality reviews

        # Construct prompt
        prompt = f"""Create a concise summary (400-500 words) that covers:
1. Core gameplay mechanics and systems
2. Unique features or innovations
3. Game style, pace, and overall tone
4. Multiplayer/singleplayer aspects
5. Notable elements mentioned in reviews
6. Any unusual or creative weapon/item usage

Focus on specific gameplay elements rather than marketing language.

Game: {name}

Official Description:
{description}

Detailed Description:
{detailed_desc}

Top Player Reviews:
"""
        for i, review in enumerate(top_reviews, 1):
            review_text = review.get('review', '').strip()
            is_positive = review.get('voted_up', False)
            sentiment = "Positive" if is_positive else "Negative"
            prompt += f"\nReview {i} ({sentiment}):\n{review_text}\n"

        prompt += "\nSummary:"
        return prompt

    def _generate_summary(self, game_data: Dict) -> Optional[str]:
        """
        Generate summary for a single game using OpenRouter LLM.
        Includes robust retry logic for RateLimitError (429) and APIError (5xx).
        Returns the summary as a string, or None if generation fails after retries.
        """
        max_retries = 4
        base_delay = 1.0  # Initial delay in seconds

        prompt = self._prepare_summary_prompt(game_data)
        messages = [
            {"role": "system", "content": "You are a skilled video game analyst who creates balanced, informative summaries of games based on official descriptions and player reviews."},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800
                )
                return completion.choices[0].message.content

            except RateLimitError as e:
                if attempt == max_retries:
                    self.logger.error(f"Rate limit exceeded for {game_data.get('name')} after {max_retries} retries: {e}")
                    return None

                # Exponential backoff with jitter
                delay = (base_delay * (2 ** attempt)) + (random.uniform(0, 1))
                self.logger.warning(f"Rate limit hit for {game_data.get('name')}. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)

            except APIError as e:
                # Handle 5xx errors or other transient API issues
                if attempt == max_retries:
                    self.logger.error(f"API Error for {game_data.get('name')} after {max_retries} retries: {e}")
                    return None

                delay = (base_delay * (2 ** attempt)) + (random.uniform(0, 1))
                self.logger.warning(f"API Error for {game_data.get('name')}: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)

            except Exception as e:
                # Non-retriable errors (e.g. invalid request, authentication)
                self.logger.error(f"Error generating summary for {game_data.get('name')}: {e}")
                return None

        return None

    def _get_statistics(self) -> Dict[str, Any]:
        """Calculate processing statistics."""
        elapsed = time.time() - self.state.start_time
        return {
            'Processed Games': self.state.processed_count,
            'Failed Games': len(self.state.failed_games),
            'Elapsed Time': f"{elapsed/3600:.2f} hours",
            'Processing Rate': f"{self.state.processed_count/elapsed:.2f} games/second" if elapsed > 0 else "N/A"
        }

    def process_single_game(self, line: str) -> tuple[Optional[Dict], Optional[str], Optional[str]]:
        """
        Process a single game line.
        Returns (game_data, summary, error_message)
        """
        try:
            game_data = json.loads(line)
            # Generate summary
            summary = self._generate_summary(game_data)
            return game_data, summary, None
        except json.JSONDecodeError as e:
            return None, None, f"JSON Error: {e}"
        except Exception as e:
            return None, None, f"Error: {e}"

    def process_games(self):
        """Main processing loop using ThreadPoolExecutor for concurrency with batching."""
        try:
            with open(self.input_file, 'r') as infile:
                # Use itertools.islice to skip previously processed lines efficiently
                # This doesn't read skipped lines into memory
                lines_iterator = itertools.islice(infile, self.state.processed_count, None)

                # Determine batch size based on worker count
                batch_size = self.max_workers * 2

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    while True:
                        # Read a batch of lines
                        batch_lines = list(itertools.islice(lines_iterator, batch_size))
                        if not batch_lines:
                            break

                        self.logger.info(f"Processing batch of {len(batch_lines)} games...")

                        # Process batch in parallel
                        # map guarantees order is preserved in the results
                        results = executor.map(self.process_single_game, batch_lines)

                        for game_data, summary, error in results:
                            if error:
                                self.logger.error(f"Error processing line: {error}")
                                self.state.processed_count += 1
                                continue

                            if not game_data:
                                self.state.processed_count += 1
                                continue

                            game_id = game_data.get('appid')

                            if summary:
                                # Build output
                                output_data = {
                                    "appid": game_id,
                                    "name": game_data.get("name"),
                                    "short_description": game_data.get("short_description"),
                                    "ai_summary": summary,
                                    "summary_generated_at": datetime.now().isoformat(),
                                    "summary_model": self.model
                                }

                                # Write to output immediately to ensure data safety
                                with open(self.output_file, 'a') as outfile:
                                    json.dump(output_data, outfile)
                                    outfile.write('\n')

                                self.logger.info(f"Successfully processed {game_data.get('name')} ({game_id})")
                                self.state.last_processed_game = game_id
                                self.state.processed_count += 1
                            else:
                                self.logger.error(f"Failed to generate summary for {game_data.get('name')} ({game_id})")
                                self.state.failed_games.append(game_id)
                                self.state.processed_count += 1

                        # Save state after each batch
                        self._save_state()

            self.logger.info("Processing completed!")

        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            self._save_state()
            raise

def main():
    """Script entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate game summaries using OpenRouter LLM (Threaded)")
    parser.add_argument("--input", default="data/steam_games_data.jsonl",
                        help="Input JSONL file path")
    parser.add_argument("--output", default="data/game_summaries_threaded.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--state", default="data/summarizer_state_threaded.json",
                        help="State file path for resuming")
    parser.add_argument("--api-key", required=False,
                        help="OpenRouter API key")
    parser.add_argument("--model",
                        default="mistralai/mistral-nemo",
                        help="Model identifier to use")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of concurrent worker threads")

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    summarizer = GameSummarizer(
        input_file=args.input,
        output_file=args.output,
        state_file=args.state,
        openrouter_key=args.api_key,
        model=args.model,
        max_workers=args.workers
    )

    summarizer.process_games()

if __name__ == "__main__":
    main()
