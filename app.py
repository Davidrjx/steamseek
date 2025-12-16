from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, flash
from markupsafe import Markup
import os
import json
import logging
import markdown  # pip install markdown
import time
from collections import OrderedDict
from threading import Thread
import uuid
import requests
import urllib.parse  # For URL encoding
# from dotenv import load_dotenv

# # Load environment variables before importing firebase_config
# load_dotenv()

# Import Firebase and Flask-Login
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from firebase_config import User, firebase_auth, db

# Import our data loader and helper functions
from data_loader import build_steam_data_index, load_summaries, get_game_data_by_appid
from game_chatbot import semantic_search_query
from llm_processor import (generate_game_analysis, rerank_search_results, OPENROUTER_API_KEY, 
                          optimize_search_query, deep_search_generate_variations, 
                          deep_search_generate_summary)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your-secret-key")  # Required for session support
# app.config["SERVER_NAME"] = os.environ.get("FLASK_SERVER_NAME", "127.0.0.1:5000")  # Commented out to fix access issues

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Specify the route for the login page

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# --- Explicit Logger Configuration ---
# Remove existing basicConfig if it causes conflicts, or adjust level
# logging.basicConfig(level=logging.INFO) # Keep or remove based on testing
if not app.debug: # Optionally configure differently for production
    # Example: Log to a file in production
    # handler = logging.FileHandler('app.log')
    # handler.setLevel(logging.INFO)
    # app.logger.addHandler(handler)
    app.logger.setLevel(logging.DEBUG)
    pass 
else:
     # Ensure logger level is set for debug mode
    app.logger.setLevel(logging.INFO)
app.logger.info("Flask logger initialized with level: %s", app.logger.getEffectiveLevel())

# Check OpenRouter API Key
if not OPENROUTER_API_KEY:
    print("\n===========================================================")
    print("WARNING: OPENROUTER_API_KEY not set in .env file")
    print("LLM re-ranking will not work without a valid API key")
    print("===========================================================\n")
else:
    print(f"\nOpenRouter API Key found (masked): {OPENROUTER_API_KEY[:4]}...{OPENROUTER_API_KEY[-4:]}")
# ------------------------------------

# Custom Jinja filter to render markdown as HTML
def markdown_filter(text):
    if not text:
        return ""
    try:
        return Markup(markdown.markdown(text))
    except Exception as e:
        print(f"Error rendering markdown: {e}")
        return Markup(f"<p>Error rendering markdown: {e}</p><pre>{text}</pre>")
app.jinja_env.filters['markdown'] = markdown_filter

# Define file paths
STEAM_DATA_FILE = "data/steam_games_data.jsonl"
SUMMARIES_FILE = "data/summaries.jsonl"  # Pre-run AI summaries for semantic search
ANALYSIS_CACHE_FILE = "data/analysis_cache.jsonl"  # Detailed analysis cache for dashboard

# TESTING flag for development - set to True to enable synthetic summaries for testing
TESTING_ENABLE_SYNTHETIC_SUMMARIES = True

# Deep Search background process state
deep_search_status = {
    "active": False,
    "progress": 0,
    "total_steps": 0,
    "current_step": "",
    "results": [],
    "grand_summary": "",
    "original_query": "",
    "completed": False,
    "error": None,
    "session_id": None,  # Add a session ID to track which search process is current
    "results_served": False  # Add a flag to track if results have been served to the client
}

# Build the index map once at startup
logging.basicConfig(level=logging.INFO)
index_map = build_steam_data_index(STEAM_DATA_FILE)

def force_https(url: str) -> str:
    if url.startswith("http://"):
        return "https://" + url[7:]
    return url

#############################################
# Analysis Cache Helper Functions
#############################################
def load_analysis_cache(file_path: str) -> dict:
    """Load the detailed analysis cache from an external file."""
    cache = {}
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    appid = obj.get("appid")
                    if appid is not None:
                        cache[int(appid)] = obj
                except Exception as e:
                    app.logger.warning(f"Error parsing analysis cache line: {e}")
    return cache

def save_analysis_cache(cache: dict, file_path: str):
    """Save the detailed analysis cache to an external file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for analysis in cache.values():
                f.write(json.dumps(analysis) + "\n")
    except Exception as e:
        app.logger.error(f"Error saving analysis cache: {e}")

#############################################
# Helper function to run deep search in the background
#############################################
def deep_search_background_task(query, search_params):
    global deep_search_status
    
    # Store the original query for reference and later matching
    original_query = query.strip()
    
    # Generate a unique session ID for this search
    session_id = str(uuid.uuid4())
    deep_search_status["session_id"] = session_id
    deep_search_status["original_query"] = original_query  # Make sure to set this explicitly
    
    try:
        print(f"\n==== STARTING DEEP SEARCH FOR: '{original_query}' (Session: {session_id}) ====\n")
        
        # Step 1: Generate keyword variations
        deep_search_status["current_step"] = "Generating search variations"
        deep_search_status["progress"] = 10
        variations = deep_search_generate_variations(original_query)
        
        # Check if the search is still valid (not cancelled or replaced)
        if deep_search_status["session_id"] != session_id:
            print(f"Deep search session {session_id} was replaced. Terminating.")
            return
            
        # Include the original query as the first variation
        if original_query not in variations:
            variations.insert(0, original_query)
        
        # Limit the number of variations to prevent excessive API calls
        MAX_VARIATIONS = 6
        if len(variations) > MAX_VARIATIONS:
            print(f"Limiting search variations from {len(variations)} to {MAX_VARIATIONS}")
            variations = variations[:MAX_VARIATIONS]
        
        total_variations = len(variations)
        deep_search_status["total_steps"] = total_variations + 2  # +2 for initial setup and final summary
        
        # Step 2: Run searches for each variation
        combined_results = OrderedDict()  # Use OrderedDict to avoid duplicate games
        successful_variations = 0
        
        for i, variation in enumerate(variations):
            # Check if the search is still valid
            if deep_search_status["session_id"] != session_id:
                print(f"Deep search session {session_id} was replaced. Terminating.")
                return
                
            step_num = i + 1  # +1 because we started at step 1 with variation generation
            deep_search_status["progress"] = int(10 + (70 * step_num / total_variations))
            deep_search_status["current_step"] = f"Searching with variation {step_num}/{total_variations}: '{variation}'"
            
            # Add a small delay to ensure the frontend can see the progress update
            time.sleep(0.5)
            
            try:
                # Get the search results for this variation
                results, _ = perform_search(
                    variation,
                    search_params["genre"],
                    search_params["year"],
                    search_params["platform"],
                    search_params["price"],
                    "Relevance",  # Always use relevance sort for deep search variations
                    False,  # Don't use AI Enhanced for variations
                    False,  # Not a deep search (to avoid recursion)
                    False,  # Don't need to save results to status (we'll combine them)
                )
                
                # Add these results to our combined set, avoiding duplicates
                for result in results:
                    appid = result["appid"]
                    if appid not in combined_results:
                        combined_results[appid] = result
                
                successful_variations += 1
            except Exception as e:
                print(f"Error during search for variation '{variation}': {str(e)}")
                import traceback
                print(traceback.format_exc())
                # Continue with the next variation
            
            # Short sleep to prevent rate limiting
            time.sleep(0.5)
        
        # Check if the search is still valid
        if deep_search_status["session_id"] != session_id:
            print(f"Deep search session {session_id} was replaced. Terminating.")
            return
            
        # Convert OrderedDict to list
        all_results = list(combined_results.values())
        
        # If we didn't get any successful searches, report the error
        if successful_variations == 0:
            deep_search_status["error"] = "All search variations failed. Please try again."
            deep_search_status["progress"] = 100
            deep_search_status["current_step"] = "Failed to complete any searches"
            deep_search_status["completed"] = True
            deep_search_status["active"] = False
            return
        
        # Step 3: Generate the summary and final ranking
        deep_search_status["current_step"] = "Generating final summary and ranking"
        deep_search_status["progress"] = 90
        
        if all_results:
            try:
                # Generate the summary and get the reranked order
                # Important: Use the original query here, not a variation
                ranked_appids, grand_summary = deep_search_generate_summary(original_query, all_results)
                
                # Check if the search is still valid
                if deep_search_status["session_id"] != session_id:
                    print(f"Deep search session {session_id} was replaced. Terminating.")
                    return
                
                # Reorder the results based on the ranking
                reranked_results = []
                appid_to_result = {result["appid"]: result for result in all_results}
                
                # First add all the ranked appids in the specified order
                for appid in ranked_appids:
                    if appid in appid_to_result:
                        reranked_results.append(appid_to_result[appid])
                        
                # Then add any remaining results that weren't in the ranking
                for result in all_results:
                    if result["appid"] not in ranked_appids:
                        reranked_results.append(result)
                        
                # Update the status with the final results (only if this is still the active search)
                if deep_search_status["session_id"] == session_id:
                    # Make sure all necessary fields are updated
                    deep_search_status.update({
                        "results": reranked_results,
                        "grand_summary": grand_summary,
                        "original_query": original_query,
                        "progress": 100,
                        "current_step": "Completed",
                        "completed": True,
                        "active": False,
                        "results_served": False,  # Reset the served flag
                        "error": None
                    })
                    print(f"Final result count: {len(reranked_results)}, Grand summary length: {len(grand_summary)}")
            except Exception as e:
                print(f"Error generating final summary: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
                # If summary generation fails, still return the results but with a default message
                if deep_search_status["session_id"] == session_id:
                    # Make sure all necessary fields are updated
                    deep_search_status.update({
                        "results": all_results,
                        "grand_summary": f"Found {len(all_results)} games matching your query. The summary generation encountered an error: {str(e)}",
                        "original_query": original_query,
                        "progress": 100,
                        "current_step": "Completed (with errors)",
                        "completed": True,
                        "active": False,
                        "results_served": False,
                        "error": str(e)
                    })
        else:
            if deep_search_status["session_id"] == session_id:
                # Make sure all necessary fields are updated
                deep_search_status.update({
                    "results": [],
                    "grand_summary": "No games found matching your query.",
                    "original_query": original_query,
                    "progress": 100,
                    "current_step": "Completed (no results)",
                    "completed": True,
                    "active": False,
                    "results_served": False,
                    "error": "No games found matching your query."
                })
        
        # Add a delay to make sure final status update is seen
        time.sleep(1)
        
        print(f"\n==== DEEP SEARCH COMPLETED FOR: '{original_query}' (Session: {session_id}) ====\n")
        print(f"Ready for viewing: query='{deep_search_status['original_query']}', result count={len(deep_search_status['results'])}")
    except Exception as e:
        print(f"Unexpected error in deep search background task: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Update status to show the error
        deep_search_status.update({
            "error": f"An unexpected error occurred: {str(e)}",
            "progress": 100,
            "current_step": "Error occurred",
            "completed": True,
            "active": False
        })

#############################################
# Search Helper with Filtering and Sorting
#############################################
def perform_search(query, selected_genre="All", selected_year="All", selected_platform="All", 
                  selected_price="All", sort_by="Relevance", use_ai_enhanced=False, 
                  use_deep_search=False, save_to_status=True, limit=50):
    app.logger.info(f"--- Entering perform_search --- Query: '{query}', Sort By: '{sort_by}', AI Enhanced: {use_ai_enhanced}, Deep Search: {use_deep_search}") # DEBUG
    
    # Make sure the query is properly stripped of whitespace
    query = query.strip()
    
    # If limit is None, set it to a default of 50
    if limit is None:
        limit = 50
    
    # If requesting a deep search, we'll start the background process and return empty results
    if use_deep_search:
        global deep_search_status
        
        # Check if we already have a completed deep search for this query that hasn't been served
        if deep_search_status["completed"] and not deep_search_status["results_served"] and deep_search_status["original_query"].lower() == query.lower():
            # Use the completed deep search results instead of starting a new search
            print(f"Using existing completed deep search results for query: '{query}'")
            return deep_search_status["results"], "Deep Search completed. Here are your results."
        
        # If a deep search is already running, just return empty results
        if deep_search_status["active"]:
            return [], "A Deep Search is already in progress."
        
        # Check if this is a restart of an identical search
        if deep_search_status["completed"] and deep_search_status["original_query"].lower() == query.lower():
            print(f"Preventing automatic restart of deep search for: '{query}'")
            return [], "This search was already completed. Refresh the page to start a new deep search."
        
        # Reset deep search status with a completely new dictionary to prevent partial updates
        deep_search_status.clear()
        deep_search_status.update({
            "active": True,
            "progress": 0,
            "total_steps": 0,
            "current_step": "Initializing Deep Search",
            "results": [],
            "grand_summary": "",
            "original_query": query,  # Set the original query
            "completed": False,
            "error": None,
            "session_id": None,  # Will be set in the background task
            "results_served": False  # Reset the served flag
        })
        
        print(f"Initialized new deep search for: '{query}'")
        
        # Start the background task
        search_params = {
            "genre": selected_genre,
            "year": selected_year,
            "platform": selected_platform,
            "price": selected_price,
        }
        thread = Thread(target=deep_search_background_task, args=(query, search_params))
        thread.daemon = True
        thread.start()
        
        # Return empty results - the client will poll for updates
        return [], "Deep Search started. Please wait while we find the best results for you."
    
    # Regular search process
    summaries_dict = load_summaries(SUMMARIES_FILE)
    print(f"Perform search loaded {len(summaries_dict)} summaries") # NEW DEBUG
    
    # Apply AI optimization to the query if enabled
    actual_search_query = query
    optimization_explanation = ""
    
    if use_ai_enhanced and query.strip():
        try:
            actual_search_query, optimization_explanation = optimize_search_query(query)
            print(f"Original query: '{query}'")
            print(f"Optimized query: '{actual_search_query}'")
            print(f"Explanation: {optimization_explanation}")
        except Exception as e:
            print(f"Error optimizing query: {e}")
            # Fall back to original query if optimization fails
            pass
    
    # 1. Get initial semantic search results using the actual search query
    initial_top_k = 50
    limit_for_reranking = 50 # Changed from 25 to 50 games for re-ranking
    raw_results = semantic_search_query(actual_search_query, top_k=initial_top_k)

    if not raw_results:
        app.logger.info("Semantic search returned no results.") # DEBUG
        return [], optimization_explanation
    else:
        app.logger.info(f"Semantic search returned {len(raw_results)} raw results.") # DEBUG
        # NEW DEBUG - Show a sample of the raw results
        if len(raw_results) > 0:
            sample = raw_results[0]
            print(f"Sample raw result: appid={sample.get('appid')}, name={sample.get('name')}")

    # 2. Prepare candidates for potential LLM re-ranking and track original order
    candidates_for_reranking = []
    original_semantic_order_appids = []
    missing_summaries_count = 0
    
    for r in raw_results:
        appid = r.get("appid")
        if not appid: continue
        appid_int = int(appid)
        original_semantic_order_appids.append(appid_int)
        # Prepare candidate only if it's within the limit we send to the LLM
        if len(candidates_for_reranking) < limit_for_reranking:
             # Get the actual game data to access more information if needed
             game_data = None
             if TESTING_ENABLE_SYNTHETIC_SUMMARIES:
                 game_data = get_game_data_by_appid(appid_int, STEAM_DATA_FILE, index_map)
             
             summary_obj = summaries_dict.get(appid_int, {})
             ai_summary = summary_obj.get("ai_summary", "")
             
             if ai_summary:
                 # We have a real AI summary from the summaries file
                 candidates_for_reranking.append({"appid": appid_int, "ai_summary": ai_summary})
             elif TESTING_ENABLE_SYNTHETIC_SUMMARIES and game_data:
                 # TESTING MODE: Generate a synthetic summary for testing
                 missing_summaries_count += 1
                 
                 # Create a basic description from the game data
                 name = game_data.get("name", "Unknown Game")
                 description = game_data.get("short_description", "No description available.")
                 
                 # Create a synthetic summary that's good enough for testing
                 synthetic_summary = f"SYNTHETIC SUMMARY FOR TESTING:\n{name} is a game on Steam. {description}"
                 
                 if missing_summaries_count <= 3:
                     print(f"Generated synthetic summary for: {name} (appid: {appid_int})")
                 
                 candidates_for_reranking.append({
                     "appid": appid_int, 
                     "ai_summary": synthetic_summary
                 })
             else:
                 missing_summaries_count += 1
                 # Only print the first few missing ones to avoid console spam
                 if missing_summaries_count <= 5:
                     print(f"Missing AI summary for appid {appid_int} (name: {r.get('name', 'Unknown')})")

    print(f"Missing summaries for {missing_summaries_count} out of {len(original_semantic_order_appids)} search results")
    print(f"Final candidate count for re-ranking: {len(candidates_for_reranking)}")
    app.logger.info(f"Prepared {len(candidates_for_reranking)} candidates for re-ranking (limit: {limit_for_reranking}).")

    # 3. Determine the processing order of appids
    processing_order_appids = original_semantic_order_appids # Default: semantic order

    # --- DEBUG Check before IF ---
    app.logger.info(f"Checking condition for re-ranking: sort_by == 'Relevance' ({sort_by == 'Relevance'}), len(candidates_for_reranking) > 0 ({len(candidates_for_reranking) > 0})")
    # --- END DEBUG ---

    if sort_by == "Relevance" and candidates_for_reranking:
        app.logger.info(f"Attempting LLM re-ranking for query: '{actual_search_query}'") # Expected log
        print(f"\n>> ATTEMPTING LLM RE-RANKING for query: '{actual_search_query}'")
        print(f">> Number of candidates: {len(candidates_for_reranking)}")
        
        try:
            # Before calling the function
            print(f">> First few candidate AppIDs: {[c['appid'] for c in candidates_for_reranking[:3]]}")
            print(">> First candidate summary (truncated): " + candidates_for_reranking[0]['ai_summary'][:100] + "...")
            
            # Call the new re-ranking function
            app.logger.info("Calling rerank_search_results...") # DEBUG
            print(">> Calling rerank_search_results function now...")
            
            ordered_appids_from_llm, llm_comment = rerank_search_results(actual_search_query, candidates_for_reranking)
            
            app.logger.info("rerank_search_results call completed.") # DEBUG
            print(">> rerank_search_results call completed.")

            if ordered_appids_from_llm is not None:
                app.logger.info(f"LLM Re-ranking successful. Comment: {llm_comment}") # Expected log
                print(f">> SUCCESS! LLM comment: {llm_comment}")
                
                # Create the new order: Start with LLM's order, then append remaining semantic results
                # that weren't in the LLM's list, maintaining their relative semantic order.
                llm_ordered_set = set(ordered_appids_from_llm)
                # Append remaining original IDs not covered by LLM ranking
                remaining_semantic_appids = [appid for appid in original_semantic_order_appids if appid not in llm_ordered_set]
                processing_order_appids = ordered_appids_from_llm + remaining_semantic_appids
                
                print(f">> New processing order (first few): {processing_order_appids[:5]}")
            else:
                 # Re-ranking failed, log the reason (comment might contain error)
                 app.logger.warning(f"LLM re-ranking failed or returned invalid data. Reason: {llm_comment}. Falling back to semantic order.") # Expected log
                 print(f">> FAILURE: {llm_comment}")
                 print(">> Falling back to semantic search order.")
                 # Keep the default semantic order assigned earlier
        except Exception as e:
            app.logger.error(f"Exception during LLM re-ranking call: {e}. Falling back to semantic order.", exc_info=True) # Expected log
            print(f">> EXCEPTION: {e}")
            import traceback
            print(traceback.format_exc())
            print(">> Falling back to semantic search order.")
            # Keep the default semantic order assigned earlier
        
        print(">> LLM RE-RANKING ATTEMPT COMPLETE")
    else:
        app.logger.info("Skipping LLM re-ranking based on sort_by or empty candidates.") # DEBUG
        print(f">> Skipping LLM re-ranking. sort_by={sort_by}, candidates={len(candidates_for_reranking)}")

    # 4. Fetch full data, filter, and build results based on the determined processing_order_appids
    results_dict = {} # Use dict to store results before final sorting
    processed_count = 0
    max_results_to_display = 25 # Limit final results shown on page? Adjust as needed.

    for appid in processing_order_appids:
        # Optional: Stop processing if we have enough results for the page
        # if processed_count >= max_results_to_display:
        #    break

        # --- Fetch full game data ---
        game_data = get_game_data_by_appid(appid, STEAM_DATA_FILE, index_map)
        if not game_data:
            app.logger.warning(f"Could not retrieve game data for appid {appid} during search processing.")
            continue

        # --- Extract data needed for filtering and display (reuse existing logic) ---
        reviews = game_data.get("reviews", [])
        total_reviews = len(reviews)
        positive_count = sum(1 for review in reviews if review.get("voted_up"))
        pos_percent = (positive_count / total_reviews * 100) if total_reviews > 0 else 0

        media = [] # Extract media... (keep existing logic)
        if game_data.get("header_image"): 
            media.append(force_https(game_data["header_image"]))
        if isinstance(game_data.get("screenshots"), list):
            for s in game_data["screenshots"]:
                if isinstance(s, dict) and s.get("path_full"):
                    media.append(force_https(s["path_full"]))
                else:
                    media.append(force_https(str(s)))
        store_data = game_data.get("store_data", {})
        if isinstance(store_data, dict):
            movies = store_data.get("movies", [])
            for movie in movies:
                webm_max = movie.get("webm", {}).get("max")
                mp4_max = movie.get("mp4", {}).get("max")
                if webm_max:
                    media.append(force_https(webm_max))
                elif mp4_max:
                    media.append(force_https(mp4_max))
                else:
                    thumb = movie.get("thumbnail")
                    if thumb:
                        media.append(force_https(thumb))

        summary_obj = summaries_dict.get(appid, {}) # Fetch summary again or pass from raw_results if needed
        ai_summary = summary_obj.get("ai_summary", "")

        genres = [] # Extract genres... (keep existing logic)
        if "store_data" in game_data and isinstance(game_data["store_data"], dict):
            genre_list = game_data["store_data"].get("genres", [])
            genres = [g.get("description") for g in genre_list if g.get("description")]

        release_date_str = game_data.get("release_date", "") # Extract year... (keep existing logic)
        year = "Unknown"
        if release_date_str:
            try: 
                year = release_date_str.split(",")[-1].strip()
            except: 
                pass

        platforms = game_data.get("store_data", {}).get("platforms", {}) # Extract platforms...

        is_free = game_data.get("store_data", {}).get("is_free", False) # Extract price...
        price = 0.0
        if not is_free:
            price_overview = game_data.get("store_data", {}).get("price_overview", {})
            if price_overview: 
                price = price_overview.get("final", 0) / 100.0

        # --- Apply Filters ---
        if selected_genre != "All" and selected_genre not in genres: 
            continue
        if selected_year != "All" and year != selected_year: 
            continue
        if selected_platform != "All":
            platform_key = selected_platform.lower()
            if not platforms.get(platform_key, False): 
                continue
        if selected_price == "Free" and not is_free: 
            continue
        if selected_price == "Paid" and is_free: 
            continue

        # --- If filters pass, store the result ---
        results_dict[appid] = {
            "appid": appid,
            "name": game_data.get("name", "Unknown"),
            "media": media,
            "genres": genres,
            "release_year": year,
            "platforms": platforms,
            "is_free": is_free,
            "price": price,
            "pos_percent": pos_percent,
            "total_reviews": total_reviews,
            "ai_summary": ai_summary # Keep summary for potential display
        }
        processed_count += 1

    # 5. Create the final list, respecting the processing order
    # This ensures that if sort_by=="Relevance", the LLM/semantic order is maintained after filtering
    final_results = [results_dict[appid] for appid in processing_order_appids if appid in results_dict]

    # 6. Apply final explicit sorting ONLY if the user chose something other than "Relevance"
    if sort_by != "Relevance":
        app.logger.info(f"Applying final sort: {sort_by}")
        if sort_by == "Name (A-Z)":
            final_results.sort(key=lambda x: x["name"])
        elif sort_by == "Release Date (Newest)":
            final_results.sort(key=lambda x: int(x["release_year"]) if x["release_year"].isdigit() else 0, reverse=True)
        elif sort_by == "Release Date (Oldest)":
            final_results.sort(key=lambda x: int(x["release_year"]) if x["release_year"].isdigit() else float('inf'))
        elif sort_by == "Price (Low to High)":
            final_results.sort(key=lambda x: x["price"])
        elif sort_by == "Price (High to Low)":
            final_results.sort(key=lambda x: x["price"], reverse=True)
        elif sort_by == "Review Count (High to Low)":
            final_results.sort(key=lambda x: x["total_reviews"], reverse=True)
        elif sort_by == "Positive Review % (High to Low)":
            final_results.sort(key=lambda x: x["pos_percent"], reverse=True)

    # Limit the final results based on the user's selection
    if limit and limit < len(final_results):
        app.logger.info(f"Limiting final results from {len(final_results)} to {limit}")
        final_results = final_results[:limit]

    # If this is a deep search and we need to save to status
    if save_to_status and use_deep_search:
        deep_search_status["results"] = final_results

    app.logger.info(f"--- Exiting perform_search --- Returning {len(final_results)} final results.") # DEBUG
    return final_results, optimization_explanation

#############################################
# Deep Search Status Route for AJAX polling
#############################################
@app.route("/deep_search_status")
def get_deep_search_status():
    """Returns the current status of a deep search as JSON for polling."""
    global deep_search_status
    
    status_copy = dict(deep_search_status)  # Make a copy to avoid thread issues
    
    # Ensure all necessary fields are present
    if "progress" not in status_copy:
        status_copy["progress"] = 0
    
    if "current_step" not in status_copy:
        status_copy["current_step"] = "Initializing..."
    
    if "completed" not in status_copy:
        status_copy["completed"] = False
    
    # Add a client_friendly field to indicate search is ready for viewing
    status_copy["ready_for_viewing"] = status_copy.get("completed", False) and not status_copy.get("results_served", False)
    
    # For security/performance reasons, don't include the full results
    # in the status JSON (they can be large)
    if "results" in status_copy:
        # Just include the count instead of the full results
        status_copy["result_count"] = len(status_copy["results"])
        del status_copy["results"]
    
    # Print status when a search is ready for viewing
    if status_copy["ready_for_viewing"]:
        print(f"Deep search is ready for viewing: query='{status_copy.get('original_query', '')}', result_count={status_copy.get('result_count', 0)}")
    
    return jsonify(status_copy)

#############################################
# Authentication Routes
#############################################
@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = None
    if request.method == 'POST':
        return redirect(url_for('auth_google'))
    return render_template('login.html', error=error_message)

@app.route('/auth/google')
def auth_google():
    """Start the Google OAuth flow by redirecting to Google sign-in page"""
    # Generate a random state value for CSRF protection
    state = str(uuid.uuid4())
    session['oauth_state'] = state
    
    # Fix: Always use localhost in the redirect URI since that's what's configured in Google OAuth
    client_id = os.environ.get('GOOGLE_CLIENT_ID')
    
    # IMPORTANT: Always use localhost, not 127.0.0.1, to match Google OAuth settings
    # Even if the user is accessing via 127.0.0.1, we need to use localhost in the redirect
    #(old style)# redirect_uri = "http://localhost:5000/auth/google/callback"
    redirect_uri = url_for('auth_google_callback', _external=True)
    
    # Print detailed debug information
    print("\n=== GOOGLE AUTH DEBUG INFO ===")
    print(f"Client ID: {client_id}")
    print(f"Current Host: {request.host}")
    print(f"Using Redirect URI: {redirect_uri}")
    print(f"State: {state}")
    print("===============================\n")

    # URL encode parameters for the auth URL
    params = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': 'email profile',
        'state': state
    }
    encoded_params = urllib.parse.urlencode(params)
    
    # Build the auth URL with the correctly encoded parameters
    auth_url = f"https://accounts.google.com/o/oauth2/auth?{encoded_params}"
    
    print(f"Full auth URL: {auth_url}")
    
    return redirect(auth_url)

@app.route('/auth/google/callback')
def auth_google_callback():
    """Handle the callback from Google OAuth"""
    # Verify state parameter to prevent CSRF attacks
    if 'oauth_state' not in session or request.args.get('state') != session['oauth_state']:
        flash('Authentication failed: Invalid state parameter.', 'danger')
        return redirect(url_for('login'))
    
    # Check for error parameter from Google
    if request.args.get('error'):
        error = request.args.get('error')
        print(f"OAuth error returned from Google: {error}")
        flash(f'Authentication failed: {error}', 'danger')
        return redirect(url_for('login'))
    
    # Exchange the authorization code for tokens
    try:
        code = request.args.get('code')
        if not code:
            flash('Authentication failed: No authorization code received.', 'danger')
            return redirect(url_for('login'))
        
        print(f"\n=== GOOGLE AUTHENTICATION FLOW ===")
        print(f"Received code from Google. Length: {len(code)}")
            
        # Exchange the code for tokens
        token = exchange_code_for_token(code)
        if not token:
            flash('Authentication failed: Could not retrieve token from Google.', 'danger')
            return redirect(url_for('login'))
        
        print(f"Token retrieved successfully. Length: {len(token)}")
        
        # Verify and use the token to get user info
        user_info = verify_id_token(token)
        if not user_info:
            flash('Authentication failed: Could not verify user identity.', 'danger')
            return redirect(url_for('login'))
        
        # Get required user information
        uid = user_info.get('sub')
        email = user_info.get('email')
        name = user_info.get('name', email.split('@')[0] if email else 'User')
        picture = user_info.get('picture', '')
        
        print(f"User information retrieved: ID={uid}, Email={email}, Name={name}")
        
        if not uid or not email:
            flash('Authentication failed: Missing user information from Google.', 'danger')
            return redirect(url_for('login'))
            
        # Create a User object
        try:
            user_obj = User(uid=uid, email=email, display_name=name, photo_url=picture)
            
            # Save user to Firestore
            result = user_obj.create_or_update()
            if result:
                print(f"User saved to Firestore: {email}")
            else:
                print(f"Warning: Failed to save user to Firestore: {email}")
                # Continue anyway - we can still log the user in
                
            # Log the user in with Flask-Login
            login_user(user_obj)
            print(f"User logged in successfully: {email}")
            print(f"=================================\n")
            
            # Redirect to the home page
            flash(f'Welcome, {name}!', 'success')
            return redirect(url_for('search'))
        except Exception as user_error:
            print(f"Error creating user object: {user_error}")
            import traceback
            traceback.print_exc()
            flash('Authentication failed: Error creating user account.', 'danger')
            return redirect(url_for('login'))
    except Exception as e:
        print(f"Error in Google callback: {e}")
        import traceback
        traceback.print_exc()
        flash('Authentication failed. Please try again.', 'danger')
        return redirect(url_for('login'))

def exchange_code_for_token(code):
    """Exchange the authorization code for an ID token"""
    try:
        # Set up the proper OAuth token exchange with Google
        token_url = 'https://oauth2.googleapis.com/token'
        
        # IMPORTANT: Always use localhost, not 127.0.0.1, to match Google OAuth settings
        # The redirect_uri must match exactly what we sent in the auth request
        #(old style)# redirect_uri = "http://localhost:5000/auth/google/callback"
        redirect_uri = url_for('auth_google_callback', _external=True)
        
        # Print debug information
        print("\n=== TOKEN EXCHANGE DEBUG INFO ===")
        print(f"Code: {code[:10]}... (truncated)")
        print(f"Current Host: {request.host}")
        print(f"Using Redirect URI: {redirect_uri}")
        print("==================================\n")
        
        # Send request to Google to exchange code for tokens
        data = {
            'code': code,
            'client_id': os.environ.get('GOOGLE_CLIENT_ID'),
            'client_secret': os.environ.get('GOOGLE_CLIENT_SECRET'),
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        # URL encode all parameters properly
        encoded_data = urllib.parse.urlencode(data)
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        response = requests.post(token_url, data=encoded_data, headers=headers)
        
        if response.status_code != 200:
            print(f"Token exchange failed. Status: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        token_data = response.json()
        print(f"Token exchange successful. Available tokens:")
        # Log which tokens we received (but don't print the actual values)
        for key in token_data:
            print(f"  - {key}: {'[present]' if token_data.get(key) else '[missing]'}")
        
        # Return the ID token or access token (many implementations use access token)
        if 'id_token' in token_data:
            return token_data.get('id_token')
        elif 'access_token' in token_data:
            # We can use the access token to fetch user info directly
            return token_data.get('access_token')
        else:
            print("Error: No usable token found in the response")
            return None
    except Exception as e:
        print(f"Error exchanging code for token: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_id_token(token):
    """Verify the token from Google and return the user's information.
    This function handles both ID tokens and access tokens."""
    try:
        # Print debug information
        print(f"\n=== TOKEN VERIFICATION ===")
        print(f"Token received: {token[:20]}... (truncated)")
        print(f"Token type appears to be: {'ID token' if len(token) > 500 else 'Access token'}")
        
        # First try to get user info directly from Google using the token
        # This works for both access tokens and ID tokens
        try:
            if len(token) < 500:  # Likely an access token based on length
                print("Using token as access token to fetch user info from Google")
                # Use the access token to get user info
                user_info_url = "https://www.googleapis.com/oauth2/v3/userinfo"
                headers = {"Authorization": f"Bearer {token}"}
                response = requests.get(user_info_url, headers=headers)
                
                if response.status_code == 200:
                    user_data = response.json()
                    user_info = {
                        'sub': user_data.get('sub'),
                        'email': user_data.get('email'),
                        'name': user_data.get('name'),
                        'picture': user_data.get('picture')
                    }
                    print(f"User info retrieved directly from Google: {user_info['email']}")
                    return user_info
                else:
                    print(f"Failed to get user info with access token: {response.status_code}")
                    print(f"Response: {response.text}")
            else:
                # Try to verify the token with Firebase Auth
                print("Attempting to verify token with Firebase Auth")
                decoded_token = firebase_auth.verify_id_token(token)
                
                # Extract user information
                user_info = {
                    'sub': decoded_token.get('sub', ''),
                    'email': decoded_token.get('email', ''),
                    'name': decoded_token.get('name', ''),
                    'picture': decoded_token.get('picture', '')
                }
                
                print(f"Token verified with Firebase. User: {user_info['email']}")
                return user_info
        except Exception as e:
            print(f"Primary verification failed: {e}")
            import traceback
            traceback.print_exc()
            
        # Fallback: Verify with Google directly
        print(f"Attempting fallback verification with Google tokeninfo endpoint...")
        
        try:
            # Request information from Google's tokeninfo endpoint
            tokeninfo_url = 'https://oauth2.googleapis.com/tokeninfo'
            
            # Try as ID token first
            params = {'id_token': token}
            response = requests.get(tokeninfo_url, params=params)
            
            # If that fails, try as access token
            if response.status_code != 200:
                print(f"ID token verification failed. Trying as access token...")
                params = {'access_token': token}
                response = requests.get(tokeninfo_url, params=params)
            
            if response.status_code == 200:
                token_info = response.json()
                user_info = {
                    'sub': token_info.get('sub', token_info.get('user_id', '')),
                    'email': token_info.get('email', ''),
                    'name': token_info.get('name', token_info.get('email', '').split('@')[0]),
                    'picture': token_info.get('picture', '')
                }
                print(f"Fallback verification successful. User: {user_info['email']}")
                return user_info
            else:
                print(f"Fallback verification failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        except Exception as fallback_error:
            print(f"Fallback verification error: {fallback_error}")
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"Error in token verification: {e}")
        traceback.print_exc()
        return None

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('search'))

#############################################
# Routes
#############################################
@app.route("/", methods=["GET", "POST"])
def search():
    query = ""
    results = []
    optimization_explanation = ""
    grand_summary = ""
    # Default filter values
    selected_genre = "All"
    selected_year = "All"
    selected_platform = "All"
    selected_price = "All"
    sort_by = "Relevance"
    use_ai_enhanced = False
    use_deep_search = False
    show_previous_search = False
    deep_search_active = False
    regular_search_active = False
    restored_from_cache = False
    result_limit = 50
    previous_results = []  # Instead of getting from session, initialize as empty list
    
    global deep_search_status
    global regular_search_status

    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    no_reload = request.args.get('no_reload') == 'true' or request.form.get('no_reload') == 'true'
    
    print(f"Request type: {'AJAX' if is_ajax else 'Regular'}, no_reload: {no_reload}")

    # Special case for ?restore=true - just show the form with data from previous session,
    # without triggering a new search
    if request.method == "GET" and request.args.get("restore") == "true":
        # We'll set this flag to indicate in the template that we're just restoring the form
        # The frontend JS will handle the actual restoration of values
        print("Restore mode detected - will not trigger new search")
        restored_from_cache = True
        # Explicitly ensure results are an empty list
        results = []
        # Don't set any values from session - let the JS on the client side handle it
        # from localStorage instead
        query = ""

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        selected_genre = request.form.get("genre", "All")
        selected_year = request.form.get("release_year", "All")
        selected_platform = request.form.get("platform", "All")
        selected_price = request.form.get("price", "All")
        sort_by = request.form.get("sort_by", "Relevance")
        use_ai_enhanced = request.form.get("use_ai_enhanced") == "true"
        use_deep_search = request.form.get("use_deep_search") == "true"
        result_limit = int(request.form.get("result_limit", "50"))

        # Ensure only one of AI Enhanced or Deep Search is enabled
        if use_ai_enhanced and use_deep_search:
            use_ai_enhanced = False  # Deep Search takes precedence
        
        # Check if we already have a completed deep search for this query that hasn't been served
        if deep_search_status["completed"] and not deep_search_status["results_served"] and deep_search_status["original_query"] == query:
            # Use the completed deep search results instead of starting a new search
            print(f"Using completed deep search results for query: '{query}'")
            results = deep_search_status["results"]
            grand_summary = deep_search_status["grand_summary"]
            deep_search_active = False
            deep_search_status["results_served"] = True  # Mark as served to prevent reuse
            use_deep_search = False  # Prevent starting a new deep search
            
            # Store search parameters in session
            session["last_search"] = {
                "query": query,
                "filters": {
                    "genre": selected_genre,
                    "release_year": selected_year,
                    "platform": selected_platform,
                    "price": selected_price,
                    "sort_by": sort_by,
                    "result_limit": result_limit,
                    "use_ai_enhanced": use_ai_enhanced,
                    "use_deep_search": False  # Set to False since we're using cached results
                }
            }
            
            # Don't save large result sets in session
            # session['previous_results'] = results  # REMOVED
        elif query:
            # Store search parameters in session
            session["last_search"] = {
                "query": query,
                "filters": {
                    "genre": selected_genre,
                    "release_year": selected_year,
                    "platform": selected_platform,
                    "price": selected_price,
                    "sort_by": sort_by,
                    "result_limit": result_limit,
                    "use_ai_enhanced": use_ai_enhanced,
                    "use_deep_search": use_deep_search
                }
            }
            
            # Store current results as previous results before starting a new search
            # We won't use session for this anymore
            # if 'results' in session and session['results']:
            #     previous_results = session['results']
            # else:
            #     previous_results = results  # Use current results as previous results
            
            # Don't store results in session anymore
            # session['previous_results'] = previous_results  # REMOVED
            
            session['previous_results'] = previous_results
            
            if use_deep_search:
                # Start a deep search
                deep_search_active = True
                # Reset deep search status
                deep_search_status.clear()
                deep_search_status.update({
                    "active": True,
                    "progress": 0,
                    "total_steps": 0,
                    "current_step": "Initializing Deep Search",
                    "results": [],
                    "grand_summary": "",
                    "original_query": query,  # Set the original query
                    "completed": False,
                    "error": None,
                    "session_id": None,  # Will be set in the background task
                    "results_served": False  # Reset the served flag
                })
                
                # Start the background task
                search_params = {
                    "genre": selected_genre,
                    "year": selected_year,
                    "platform": selected_platform,
                    "price": selected_price,
                }
                thread = Thread(target=deep_search_background_task, args=(query, search_params))
                thread.daemon = True
                thread.start()
                
                # Keep previous results visible while searching
                results = previous_results
                optimization_explanation = "Deep Search started. Please wait while we find the best results for you."
            elif use_ai_enhanced:
                # Start an AI enhanced search
                regular_search_active = True
                
                # Reset regular search status
                regular_search_status.clear()
                regular_search_status.update({
                    "active": True,
                    "progress": 0,
                    "current_step": "Initializing",
                    "search_type": "ai_enhanced",
                    "query": query,
                    "completed": False,
                    "error": None,
                    "session_id": str(uuid.uuid4()),  # Generate a new session ID
                    "optimization_explanation": ""
                })
                
                # Prepare search parameters
                search_params = {
                    "genre": selected_genre,
                    "year": selected_year,
                    "platform": selected_platform,
                    "price": selected_price,
                    "sort_by": sort_by,
                    "result_limit": result_limit
                }
                
                # Start the background task
                thread = Thread(target=regular_search_background_task, args=(query, search_params, use_ai_enhanced))
                thread.daemon = True
                thread.start()
                
                # Keep previous results visible while searching
                results = previous_results
                optimization_explanation = "AI Enhanced search started. Please wait for results..."
            else:
                # For standard search, run it immediately without background thread
                print(f"Running standard search for query: '{query}'")
                
                # Perform the search directly (not in background)
                # We'll use the existing perform_search function
                results, _ = perform_search(
                    query, 
                    selected_genre, 
                    selected_year, 
                    selected_platform, 
                    selected_price, 
                    sort_by,
                    use_ai_enhanced=False,
                    use_deep_search=False,
                    save_to_status=False,
                    limit=result_limit
                )
                
                # Don't save results in session anymore
                # session['previous_results'] = results  # REMOVED
                # session['results'] = results  # REMOVED
                
                print(f"Standard search completed with {len(results)} results")
                
                # If this is an AJAX request, return just the results HTML
                if is_ajax and no_reload:
                    print("Rendering partial results for AJAX request")
                    return render_template(
                        "search.html", 
                        query=query, 
                        results=results,
                        selected_genre=selected_genre, 
                        selected_year=selected_year,
                        selected_platform=selected_platform, 
                        selected_price=selected_price,
                        sort_by=sort_by,
                        use_ai_enhanced=False,
                        use_deep_search=False,
                        optimization_explanation="",
                        grand_summary="",
                        deep_search_active=False,
                        regular_search_active=False,
                        show_previous_search=False,
                        restored_from_cache=False,
                        result_limit=result_limit,
                        is_partial_results=True  # Flag to indicate this is a partial render
                    )
        else:
            session.pop("last_search", None)
    elif request.method == "GET":
        # Handle regular GET requests as before
        # ... [rest of the GET handling code remains the same]

        # Added special case for AJAX standard search
        print(f"###GET request - Query: '{query}', AJAX: {is_ajax}, No Reload: {no_reload}, use_ai_enhanced: {use_ai_enhanced}, use_deep_search: {use_deep_search}")
        if query and is_ajax and no_reload and not use_ai_enhanced and not use_deep_search:
            print(f"Running standard search via AJAX GET for query: '{query}'")
            
            # Perform the search directly
            results, _ = perform_search(
                query, 
                selected_genre, 
                selected_year, 
                selected_platform, 
                selected_price, 
                sort_by,
                use_ai_enhanced=False,
                use_deep_search=False,
                save_to_status=False,
                limit=result_limit
            )
            
            # Don't save results in session anymore
            # session['previous_results'] = results  # REMOVED
            # session['results'] = results  # REMOVED
            
            print(f"AJAX standard search completed with {len(results)} results")
            
            # Return just the results HTML for AJAX
            return render_template(
                "search.html", 
                query=query, 
                results=results,
                selected_genre=selected_genre, 
                selected_year=selected_year,
                selected_platform=selected_platform, 
                selected_price=selected_price,
                sort_by=sort_by,
                use_ai_enhanced=False,
                use_deep_search=False,
                optimization_explanation="",
                grand_summary="",
                deep_search_active=False,
                regular_search_active=False,
                show_previous_search=False,
                restored_from_cache=False,
                result_limit=result_limit,
                is_partial_results=True  # Flag to indicate this is a partial render
            )
        
        # Check if this is a view_results request from the JavaScript redirect
        view_results = request.args.get("view_results") == "true"
        
        # Handle explicit query parameters in URL
        query = request.args.get("q", "").strip()
        selected_genre = request.args.get("genre", "All")
        selected_year = request.args.get("release_year", "All")
        selected_platform = request.args.get("platform", "All")
        selected_price = request.args.get("price", "All")
        sort_by = request.args.get("sort_by", "Relevance")
        use_ai_enhanced = request.args.get("use_ai_enhanced") == "true"
        use_deep_search = request.args.get("use_deep_search") == "true"
        try:
            result_limit = int(request.args.get("result_limit", "50"))
        except ValueError:
            result_limit = 50
        # Flag to explicitly re-run search (not set by default)
        run_search = request.args.get("run_search") == "true"

        # Ensure only one of AI Enhanced or Deep Search is enabled
        if use_ai_enhanced and use_deep_search:
            use_ai_enhanced = False  # Deep Search takes precedence
        
        print(f"GET request - Query: '{query}', View Results: {view_results}, Run Search: {run_search}, Deep Search Status: completed={deep_search_status['completed']}, original_query='{deep_search_status['original_query']}'")
        
        # Special handling for view_results parameter - this means we're coming from
        # a completed deep search or regular search and should display its results without restarting it
        if view_results and query:
            # Check which search type was requested based on URL parameters
            # Priority: use_deep_search > use_ai_enhanced > fallback to any completed search

            # For deep search results (when explicitly requested)
            if use_deep_search and deep_search_status["completed"] and query.lower() == deep_search_status["original_query"].lower():
                print(f"Showing completed deep search results for query: '{query}' (view_results=true, use_deep_search=true)")
                results = deep_search_status["results"]
                grand_summary = deep_search_status["grand_summary"]
                deep_search_active = False
                deep_search_status["results_served"] = True  # Mark as served to prevent reuse
                use_deep_search = False  # Reset the flag since we're just viewing results

                # Don't save large result sets in session
                # session['previous_results'] = results  # REMOVED

            # For regular/AI enhanced search results (when explicitly requested)
            elif use_ai_enhanced and regular_search_status["completed"] and query.lower() == regular_search_status["query"].lower():
                print(f"Showing completed AI Enhanced search results for query: '{query}' (view_results=true, use_ai_enhanced=true)")

                # Use the stored results from the completed background task
                results = regular_search_status["results"]
                optimization_explanation = regular_search_status["optimization_explanation"]
                regular_search_active = False

                # Don't save large result sets in session
                # session['previous_results'] = results  # REMOVED

            # Fallback: try to use any completed search results
            elif regular_search_status["completed"] and query.lower() == regular_search_status["query"].lower():
                print(f"Showing completed regular search results for query: '{query}' (view_results=true, fallback to regular)")

                # Use the stored results from the completed background task
                results = regular_search_status["results"]
                optimization_explanation = regular_search_status["optimization_explanation"]
                regular_search_active = False

                # Don't save large result sets in session
                # session['previous_results'] = results  # REMOVED

            elif deep_search_status["completed"] and query.lower() == deep_search_status["original_query"].lower():
                print(f"Showing completed deep search results for query: '{query}' (view_results=true, fallback to deep)")
                results = deep_search_status["results"]
                grand_summary = deep_search_status["grand_summary"]
                deep_search_active = False
                deep_search_status["results_served"] = True  # Mark as served to prevent reuse

                # Don't save large result sets in session
                # session['previous_results'] = results  # REMOVED
            
            # Store search parameters in session
            session["last_search"] = {
                "query": query,
                "filters": {
                    "genre": selected_genre,
                    "release_year": selected_year,
                    "platform": selected_platform,
                    "price": selected_price,
                    "sort_by": sort_by,
                    "result_limit": result_limit,
                    "use_ai_enhanced": use_ai_enhanced,
                    "use_deep_search": False  # Set to false since we're using cached results
                }
            }
            
            print(f"Results prepared: {len(results)} games")
        # Check if we have a completed deep search with the same query that hasn't been served
        elif query and deep_search_status["completed"] and not deep_search_status["results_served"] and query.lower() == deep_search_status["original_query"].lower():
            # Use the completed deep search results instead of starting a new search
            print(f"Using completed deep search results for query: '{query}'")
            results = deep_search_status["results"]
            grand_summary = deep_search_status["grand_summary"]
            deep_search_active = False
            deep_search_status["results_served"] = True  # Mark as served to prevent reuse
            
            # Save these results for future reference
            # session['previous_results'] = results  # REMOVED - don't use session for large data
            
            # Store search parameters in session
            session["last_search"] = {
                "query": query,
                "filters": {
                    "genre": selected_genre,
                    "release_year": selected_year,
                    "platform": selected_platform,
                    "price": selected_price,
                    "sort_by": sort_by,
                    "result_limit": result_limit,
                    "use_ai_enhanced": use_ai_enhanced,
                    "use_deep_search": False  # Set to false since we're using cached results
                }
            }
            
            print(f"Results prepared: {len(results)} games, Grand Summary: {len(grand_summary)} chars")
        elif query and (run_search or request.args.get("q")):
            # Execute search if query is provided AND either run_search flag is set OR query is in the URL
            print(f"Running search for query: '{query}' (explicit run from URL parameters)")
            
            # Store parameters in session
            session["last_search"] = {
                "query": query,
                "filters": {
                    "genre": selected_genre,
                    "release_year": selected_year,
                    "platform": selected_platform,
                    "price": selected_price,
                    "sort_by": sort_by,
                    "result_limit": result_limit,
                    "use_ai_enhanced": use_ai_enhanced,
                    "use_deep_search": use_deep_search
                }
            }
            
            # Store current results as previous results before starting a new search
            previous_results = session.get('previous_results', [])
            
            if use_deep_search:
                # Start a deep search
                deep_search_active = True
                # Reset deep search status
                deep_search_status.clear()
                deep_search_status.update({
                    "active": True,
                    "progress": 0,
                    "total_steps": 0,
                    "current_step": "Initializing Deep Search",
                    "results": [],
                    "grand_summary": "",
                    "original_query": query,
                    "completed": False,
                    "error": None,
                    "session_id": None,  # Will be set in the background task
                    "results_served": False
                })
                
                # Start the background task
                search_params = {
                    "genre": selected_genre,
                    "year": selected_year,
                    "platform": selected_platform,
                    "price": selected_price,
                }
                thread = Thread(target=deep_search_background_task, args=(query, search_params))
                thread.daemon = True
                thread.start()
                
                # Keep previous results visible while searching
                results = previous_results
                optimization_explanation = "Deep Search started. Please wait while we find the best results for you."
            elif use_ai_enhanced:
                # Start a regular or AI enhanced search
                regular_search_active = True
                
                # Reset regular search status
                regular_search_status.clear()
                regular_search_status.update({
                    "active": True,
                    "progress": 0,
                    "current_step": "Initializing",
                    "search_type": "ai_enhanced",
                    "query": query,
                    "completed": False,
                    "error": None,
                    "session_id": str(uuid.uuid4()),  # Generate a new session ID
                    "optimization_explanation": ""
                })
                
                # Prepare search parameters
                search_params = {
                    "genre": selected_genre,
                    "year": selected_year,
                    "platform": selected_platform,
                    "price": selected_price,
                    "sort_by": sort_by,
                    "result_limit": result_limit
                }
                
                # Start the background task
                thread = Thread(target=regular_search_background_task, args=(query, search_params, use_ai_enhanced))
                thread.daemon = True
                thread.start()
                
                # Keep previous results visible while searching
                results = previous_results
                optimization_explanation = "AI Enhanced search started. Please wait for results..."
            else:
                # For standard search, run it immediately
                print(f"Running standard search for query: '{query}' (via GET request)")
                
                # Perform the search directly (not in background)
                results, _ = perform_search(
                    query, 
                    selected_genre, 
                    selected_year, 
                    selected_platform, 
                    selected_price, 
                    sort_by,
                    use_ai_enhanced=False,
                    use_deep_search=False,
                    save_to_status=False,
                    limit=result_limit
                )
                
                # Don't save results in session anymore
                # session['previous_results'] = results  # REMOVED
                # session['results'] = results  # REMOVED
        elif "last_search" in session and not query:
            # If there's a previous search in session and no new query provided,
            # just restore the previous search parameters (don't re-run the search)
            cached = session["last_search"]
            query = cached.get("query", "")
            if query:
                filters = cached.get("filters", {})
                selected_genre = filters.get("genre", "All")
                selected_year = filters.get("release_year", "All")
                selected_platform = filters.get("platform", "All")
                selected_price = filters.get("price", "All")
                sort_by = filters.get("sort_by", "Relevance")
                result_limit = filters.get("result_limit", 50)
                use_ai_enhanced = filters.get("use_ai_enhanced", False)
                use_deep_search = filters.get("use_deep_search", False)
                show_previous_search = True
                
                # We won't use session for storing results anymore
                # results = session['previous_results']  # REMOVED
                
                # Only show this message when we're displaying a previous search form
                if show_previous_search:
                    optimization_explanation = "Your previous search is ready to run again"
    
    # Save current results in session for future reference
    # session['results'] = results  # REMOVED - don't use session for large data
    
    # If AJAX request that just need the results, check template option
    # This is different from the partial results case - it's for when the template itself 
    # is requested to include only specific parts
    if 'template' in request.args and is_ajax:
        template_name = request.args.get('template')
        if template_name == 'results_only':
            print("Rendering results-only template for AJAX")
            return render_template(
                "search_results_partial.html", 
                results=results
            )
    
    # Don't save results in session anymore
    # session['results'] = results  # REMOVED
    
    print(f"Final template values: Results: {len(results)}, Has Grand Summary: {'Yes' if grand_summary else 'No'}")
    return render_template("search.html", 
                          query=query, 
                          results=results,
                          selected_genre=selected_genre, 
                          selected_year=selected_year,
                          selected_platform=selected_platform, 
                          selected_price=selected_price,
                          sort_by=sort_by,
                          use_ai_enhanced=use_ai_enhanced,
                          use_deep_search=use_deep_search,
                          optimization_explanation=optimization_explanation,
                          grand_summary=grand_summary,
                          deep_search_active=deep_search_active,
                          regular_search_active=regular_search_active,
                          show_previous_search=show_previous_search,
                          restored_from_cache=restored_from_cache,
                          result_limit=result_limit)

@app.route("/detail/<appid>")
def detail(appid):
    try:
        appid_int = int(appid)
    except ValueError:
        return "Invalid AppID", 400

    # Preserve original search query so that the "Back to Search" link works
    orig_query = request.args.get("q", "")
    # Check if user requested a refresh ("Analyze Again")
    refresh = request.args.get("refresh", "0")

    game_data = get_game_data_by_appid(appid_int, STEAM_DATA_FILE, index_map)
    if not game_data:
        return "Game not found", 404

    # Load the external analysis cache (separate from summaries.jsonl)
    analysis_cache = load_analysis_cache(ANALYSIS_CACHE_FILE)

    # Define required keys for a complete analysis
    required_keys = {"ai_summary", "feature_sentiment", "standout_features",
                     "community_feedback", "market_analysis", "feature_validation"}
    analysis_obj = analysis_cache.get(appid_int)

    if refresh == "1" or not analysis_obj or not required_keys.issubset(analysis_obj.keys()):
        app.logger.info("Generating new detailed analysis via LLM...")
        analysis = generate_game_analysis(game_data)
        # Ensure the analysis object contains the appid for later retrieval
        analysis["appid"] = appid_int
        analysis_cache[appid_int] = analysis
        # Save updated analysis cache externally
        save_analysis_cache(analysis_cache, ANALYSIS_CACHE_FILE)
    else:
        analysis = analysis_obj

    # Compute metrics from raw reviews
    reviews = game_data.get("reviews", [])
    total_reviews = len(reviews)
    positive_count = sum(1 for r in reviews if r.get("voted_up"))
    pos_percent = (positive_count / total_reviews * 100) if total_reviews > 0 else 0

    # Build playtime distribution
    playtime_buckets = {"<10h": 0, "10-50h": 0, "50-100h": 0, ">100h": 0}
    for r in reviews:
        hours = r.get("playtime_forever", 0) / 60
        if hours < 10:
            playtime_buckets["<10h"] += 1
        elif hours < 50:
            playtime_buckets["10-50h"] += 1
        elif hours < 100:
            playtime_buckets["50-100h"] += 1
        else:
            playtime_buckets[">100h"] += 1
    playtime_distribution = [{"name": k, "value": v} for k, v in playtime_buckets.items()]

    # Player growth data (fallback if not available)
    player_growth = game_data.get("player_growth")
    if not player_growth or not isinstance(player_growth, list):
        player_growth = [
            {"month": "Jan", "players": 125},
            {"month": "Feb", "players": 350},
            {"month": "Mar", "players": 410},
            {"month": "Apr", "players": 380},
            {"month": "May", "players": 425},
        ]
        player_growth_available = False
    else:
        player_growth_available = True

    # Compute media list for carousel (same as in search)
    media = []
    if game_data.get("header_image"):
        media.append(force_https(game_data["header_image"]))
    if isinstance(game_data.get("screenshots"), list):
        for s in game_data["screenshots"]:
            if isinstance(s, dict) and s.get("path_full"):
                media.append(force_https(s["path_full"]))
            else:
                media.append(force_https(str(s)))
    store_data = game_data.get("store_data", {})
    if isinstance(store_data, dict):
        movies = store_data.get("movies", [])
        for movie in movies:
            webm_max = movie.get("webm", {}).get("max")
            mp4_max = movie.get("mp4", {}).get("max")
            if webm_max:
                media.append(force_https(webm_max))
            elif mp4_max:
                media.append(force_https(mp4_max))
            else:
                thumb = movie.get("thumbnail")
                if thumb:
                    media.append(force_https(thumb))

    return render_template("detail.html",
                           game=game_data,
                           analysis=analysis,
                           pos_percent=pos_percent,
                           total_reviews=total_reviews,
                           playtime_distribution=playtime_distribution,
                           player_growth=player_growth,
                           player_growth_available=player_growth_available,
                           orig_query=orig_query,
                           media=media)

#############################################
# Game Lists Routes
#############################################
@app.route('/user/lists')
@login_required
def user_lists():
    """Show all lists for the current user"""
    lists = current_user.get_lists()
    return render_template('lists.html', lists=lists)

@app.route('/user/lists/<list_id>')
@login_required
def view_list(list_id):
    """Show games in a specific list"""
    # Get the list information
    lists = current_user.get_lists()
    list_info = None
    for lst in lists:
        if lst['id'] == list_id:
            list_info = lst
            break
    
    if not list_info:
        flash('List not found.', 'danger')
        return redirect(url_for('user_lists'))
    
    # Get games in the list
    games = current_user.get_games_in_list(list_id)
    
    # Load summaries for AI summary data
    summaries_dict = load_summaries(SUMMARIES_FILE)
    print(f"Loaded {len(summaries_dict)} summaries for list view")
    
    # Process each game to ensure it has media, especially header_image
    for game in games:
        # Get the appid as integer for lookup
        appid = int(game['appid'])
        
        # Load AI summary from summaries file if available
        summary_obj = summaries_dict.get(appid, {})
        if 'ai_summary' in summary_obj:
            ai_summary = summary_obj['ai_summary']
            # Ensure AI summary has proper formatting
            if ai_summary and isinstance(ai_summary, str):
                # Add paragraph breaks if needed
                if not ai_summary.startswith('#') and '\n\n' not in ai_summary:
                    ai_summary = ai_summary.replace('\n', '\n\n')
                game['ai_summary'] = ai_summary
            else:
                game['ai_summary'] = ai_summary
            print(f"Found AI summary in summaries file for {game['name']} (appid: {appid})")
        elif 'ai_summary' not in game and 'short_description' in game:
            # Use short description as fallback
            game['ai_summary'] = game['short_description']
            print(f"Using short description as fallback for {game['name']} (appid: {appid})")
        else:
            print(f"No AI summary or fallback available for {game['name']} (appid: {appid})")
            
        # Ensure media list exists
        if not game.get('media') or not isinstance(game['media'], list):
            game['media'] = []
            
        # Add header image as the first item if it exists and isn't already in media
        if game.get('header_image') and game['header_image'] not in game['media']:
            game['media'].insert(0, force_https(game['header_image']))
        
        # If there's store_data with a header_image, use that as fallback
        if not game['media'] and game.get('store_data', {}).get('header_image'):
            game['media'].insert(0, force_https(game['store_data']['header_image']))
            
        # Add screenshots from store_data
        store_data = game.get('store_data', {})
        if isinstance(store_data, dict):
            # Add screenshots
            screenshots = store_data.get('screenshots', [])
            for screenshot in screenshots:
                if isinstance(screenshot, dict) and screenshot.get('path_full'):
                    media_url = force_https(screenshot['path_full'])
                    if media_url not in game['media']:
                        game['media'].append(media_url)
            
            # Add videos
            movies = store_data.get('movies', [])
            for movie in movies:
                webm_max = movie.get('webm', {}).get('max')
                mp4_max = movie.get('mp4', {}).get('max')
                if webm_max:
                    media_url = force_https(webm_max)
                    if media_url not in game['media']:
                        game['media'].append(media_url)
                elif mp4_max:
                    media_url = force_https(mp4_max)
                    if media_url not in game['media']:
                        game['media'].append(media_url)
                else:
                    thumb = movie.get('thumbnail')
                    if thumb:
                        media_url = force_https(thumb)
                        if media_url not in game['media']:
                            game['media'].append(media_url)
            
        # Ensure essential fields have default values if missing
        if 'price' not in game:
            price_overview = game.get('store_data', {}).get('price_overview', {})
            game['price'] = price_overview.get('final', 0) / 100.0 if price_overview else 0.0
            
        if 'is_free' not in game:
            game['is_free'] = game.get('store_data', {}).get('is_free', False)
            
        if 'release_year' not in game:
            # Extract year from release_date
            release_date = game.get('release_date', '')
            if release_date:
                try:
                    year = release_date.split(',')[-1].strip()
                    game['release_year'] = year
                except:
                    game['release_year'] = 'Unknown'
            else:
                game['release_year'] = 'Unknown'
                
        # Add a flag for whether the game is released
        coming_soon = game.get('store_data', {}).get('release_date', {}).get('coming_soon', False)
        game['is_released'] = not coming_soon
    
    # Handle sorting options
    sort_by = request.args.get('sort_by', 'name')
    sort_order = request.args.get('order', 'asc')
    
    if sort_by == 'date_added':
        # Sort by date_added (using timestamp or default to 0)
        reverse = sort_order == 'desc'
        games.sort(key=lambda g: g.get('timestamp', 0), reverse=reverse)
    elif sort_by == 'price':
        # Sort by price
        reverse = sort_order == 'desc'
        games.sort(key=lambda g: 0 if g.get('is_free', False) else g.get('price', 0), reverse=reverse)
    elif sort_by == 'release_year':
        # Sort by release year, putting Unknown at the end
        reverse = sort_order == 'desc'
        def release_year_key(g):
            year = g.get('release_year', 'Unknown')
            if year == 'Unknown' or year == 'TBA' or not year.isdigit():
                return 9999 if not reverse else -9999
            return int(year)
        games.sort(key=release_year_key, reverse=reverse)
    else:
        # Default sort by name
        reverse = sort_order == 'desc'
        games.sort(key=lambda g: g.get('name', '').lower(), reverse=reverse)
    
    # Handle filtering options
    show_released_only = request.args.get('released_only') == 'true'
    if show_released_only:
        games = [game for game in games if game.get('is_released', True)]
    
    return render_template('list_detail.html', 
                           list=list_info, 
                           games=games,
                           sort_by=sort_by,
                           sort_order=sort_order,
                           show_released_only=show_released_only)

@app.route('/create_list', methods=['POST'])
@login_required
def create_list():
    """Create a new list"""
    list_name = request.form.get('list_name')
    
    if not list_name or not list_name.strip():
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'error': 'List name is required'})
        flash('List name is required.', 'danger')
        return redirect(url_for('user_lists'))
    
    # Create the list
    list_id = current_user.create_list(list_name.strip())
    
    if not list_id:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'error': 'Failed to create list'})
        flash('Failed to create list.', 'danger')
        return redirect(url_for('user_lists'))
    
    # Check if this is an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': True, 'list_id': list_id, 'name': list_name.strip()})
    
    flash(f'List "{list_name}" created successfully.', 'success')
    return redirect(url_for('user_lists'))

@app.route('/delete_list/<list_id>', methods=['POST'])
@login_required
def delete_list(list_id):
    """Delete a list"""
    result = current_user.delete_list(list_id)
    
    if result:
        flash('List deleted successfully.', 'success')
    else:
        flash('Failed to delete list.', 'danger')
    
    return redirect(url_for('user_lists'))

@app.route('/save_game/<int:appid>', methods=['POST'])
@login_required
def save_game(appid):
    """Save a game to one or more lists"""
    # Check if this is an AJAX/JSON request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type') == 'application/json'
    
    # Get list IDs - first try JSON payload, then form data
    list_ids = []
    if request.is_json:
        # Get from JSON body
        data = request.json
        list_ids = data.get('lists', [])
    else:
        # Get from form data
        list_ids = request.form.getlist('list_ids')
    
    if not list_ids:
        if is_ajax:
            return jsonify({'success': False, 'message': 'Please select at least one list.'})
        flash('Please select at least one list.', 'danger')
        return redirect(request.referrer or url_for('search'))
    
    # Get the game data
    game_data = get_game_data_by_appid(appid, STEAM_DATA_FILE, index_map)
    
    if not game_data:
        if is_ajax:
            return jsonify({'success': False, 'message': 'Game not found.'})
        flash('Game not found.', 'danger')
        return redirect(request.referrer or url_for('search'))
    
    # Add the game to each selected list
    success_count = 0
    for list_id in list_ids:
        if current_user.add_game_to_list(list_id, game_data):
            success_count += 1
    
    # Prepare response based on success
    if success_count == len(list_ids):
        message = f'Game added to {success_count} list(s) successfully.'
        success = True
    elif success_count > 0:
        message = f'Game added to {success_count} out of {len(list_ids)} list(s).'
        success = True
    else:
        message = 'Failed to add game to any lists.'
        success = False
    
    # Return JSON for AJAX requests
    if is_ajax:
        return jsonify({
            'success': success,
            'message': message
        })
    
    # Standard response for non-AJAX requests
    if success:
        flash(message, 'success')
    else:
        flash(message, 'danger')
    
    return redirect(request.referrer or url_for('search'))

@app.route('/remove_game/<list_id>/<int:appid>', methods=['POST'])
@login_required
def remove_game_from_list(list_id, appid):
    """Remove a game from a list"""
    result = current_user.remove_game_from_list(list_id, appid)
    
    if result:
        flash('Game removed from list successfully.', 'success')
    else:
        flash('Failed to remove game from list.', 'danger')
    
    return redirect(url_for('view_list', list_id=list_id))

@app.route('/api/game_lists/<int:appid>')
@login_required
def get_game_lists_api(appid):
    """API to get all lists and whether they contain a specific game"""
    print(f"\n=== GET GAME LISTS API CALLED ===")
    print(f"User: {current_user.email}")
    print(f"AppID: {appid}")
    
    lists = current_user.get_lists()
    print(f"Found {len(lists)} lists for user")
    
    # For each list, check if it contains the game
    result = []
    for lst in lists:
        has_game = current_user.is_game_in_list(lst['id'], appid)
        print(f"  List: {lst['name']} (ID: {lst['id']}) - Has game: {has_game}")
        result.append({
            'id': lst['id'],
            'name': lst['name'],
            'has_game': has_game
        })
    
    print(f"Returning {len(result)} lists\n")
    return jsonify(result)

@app.route('/api/save_results_as_list', methods=['POST'])
@login_required
def save_results_as_list():
    """API to save all search results as a new list with AI-generated name"""
    try:
        # Get data from the request
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
            
        query = data.get('query', '').strip()
        results = data.get('results', [])
        
        if not query:
            return jsonify({'success': False, 'message': 'Search query is required'}), 400
            
        if not results or not isinstance(results, list):
            return jsonify({'success': False, 'message': 'No valid search results provided'}), 400
            
        # Log how many games we're saving
        print(f"Saving {len(results)} games to a new list for query: '{query}'")
            
        # Generate a list name using LLM
        list_name = generate_list_name(query, results)
        if not list_name:
            # Fallback list name if generation fails
            list_name = f"Search: {query[:30]}"
        
        # Create the list
        list_id = current_user.create_list(list_name)
        if not list_id:
            return jsonify({'success': False, 'message': 'Failed to create list'}), 500
            
        print(f"Created new list '{list_name}' (ID: {list_id}) with {len(results)} games")
        
        # Add games to the list in reverse order to maintain correct chronological order
        # (last added game will have the most recent timestamp)
        success_count = 0
        failed_games = []
        
        # First, add games in bulk to avoid timestamp issues
        for i, game_data in enumerate(reversed(results)):
            if not isinstance(game_data, dict) or 'appid' not in game_data:
                failed_games.append(f"Invalid game data at position {i}")
                continue
                
            appid = game_data.get('appid')
            
            # Get full game data
            full_game_data = get_game_data_by_appid(appid, STEAM_DATA_FILE, index_map)
            if not full_game_data:
                failed_games.append(f"Game {appid} not found")
                continue
                
            # Add the game to the list
            if current_user.add_game_to_list(list_id, full_game_data):
                success_count += 1
            else:
                failed_games.append(f"Failed to add game {appid} to list")
        
        # Prepare response
        if success_count == len(results):
            message = f"All {success_count} games saved to list '{list_name}' successfully"
            success = True
        elif success_count > 0:
            message = f"{success_count} out of {len(results)} games saved to list '{list_name}'"
            success = True
        else:
            message = "Failed to add any games to the list"
            success = False
            
        return jsonify({
            'success': success,
            'message': message,
            'list_id': list_id,
            'list_name': list_name,
            'games_added': success_count,
            'failed_games': failed_games,
            'redirect_url': url_for('view_list', list_id=list_id)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500
        
def generate_list_name(query, results):
    """Generate a list name based on the search query and results using LLM"""
    try:
        # For simplicity, just take a couple top results to use as context
        top_results = results[:5]
        game_names = [game.get('name', '') for game in top_results if game.get('name')]
        
        # Create input for the LLM
        prompt = f"""Generate a short, catchy list name (maximum 40 characters) for a collection of games based on this search query: "{query}".
        
        The top games in this collection are:
        {', '.join(game_names)}
        
        The list name should capture the theme of these games and the user's search intent.
        Respond with ONLY the list name, nothing else."""
        
        # Use the OpenRouter API to generate the list name
        list_name = None
        try:
            from llm_processor import generate_completion
            list_name = generate_completion(prompt, max_tokens=30)
            
            # Clean up the response (remove quotes, ensure not too long)
            if list_name:
                list_name = list_name.strip('"\'').strip()
                if len(list_name) > 40:
                    list_name = list_name[:37] + "..."
        except Exception as llm_error:
            print(f"Error generating list name with LLM: {llm_error}")
            
        if not list_name or len(list_name) < 3:
            # Fallback if LLM doesn't provide a valid name
            if query and len(query) <= 35:
                return f"Games about {query}"
            else:
                return f"Search: {query[:30]}"
                
        return list_name
    except Exception as e:
        print(f"Error in generate_list_name: {e}")
        # Fallback to a simple name based on query
        return f"Search: {query[:30]}"

@app.route('/api/update_list/<list_id>', methods=['POST'])
@login_required
def update_list_api(list_id):
    """API endpoint to update list metadata (name, description, notes)"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
            
        field = data.get('field')
        value = data.get('value', '').strip()
        
        # Validate field
        if field not in ['name', 'description', 'notes']:
            return jsonify({'success': False, 'message': 'Invalid field'}), 400
            
        # Validate permissions (make sure user owns the list)
        lists = current_user.get_lists()
        list_exists = False
        for lst in lists:
            if lst['id'] == list_id:
                list_exists = True
                break
                
        if not list_exists:
            return jsonify({'success': False, 'message': 'List not found or access denied'}), 404
            
        # Update list in Firebase
        result = current_user.update_list_metadata(list_id, field, value)
        
        if result:
            return jsonify({'success': True, 'message': f'List {field} updated successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to update list metadata'}), 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/render_markdown', methods=['POST'])
def render_markdown_api():
    """API endpoint to render markdown to HTML"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
            
        markdown_text = data.get('markdown', '')
        if not markdown_text:
            return jsonify({'html': ''})
            
        # Use the existing markdown filter to render HTML
        html = markdown_filter(markdown_text)
        
        return jsonify({'success': True, 'html': html})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/game_note/<appid>', methods=['GET', 'POST', 'DELETE'])
@login_required
def game_note_api(appid):
    """API endpoint to manage game notes"""
    try:
        # GET request - retrieve the note
        if request.method == 'GET':
            note = current_user.get_game_note(appid)
            return jsonify({'success': True, 'note': note})
            
        # POST request - save or update the note
        elif request.method == 'POST':
            data = request.json
            if not data:
                return jsonify({'success': False, 'message': 'No data provided'}), 400
                
            note_text = data.get('note', '').strip()
            
            # Save the note
            result = current_user.save_game_note(appid, note_text)
            
            if result:
                # If note was saved successfully, return the rendered HTML version too
                html = markdown_filter(note_text) if note_text else ''
                return jsonify({
                    'success': True, 
                    'message': 'Note saved successfully',
                    'html': html
                })
            else:
                return jsonify({'success': False, 'message': 'Failed to save note'}), 500
                
        # DELETE request - delete the note
        elif request.method == 'DELETE':
            result = current_user.delete_game_note(appid)
            
            if result:
                return jsonify({'success': True, 'message': 'Note deleted successfully'})
            else:
                return jsonify({'success': False, 'message': 'Failed to delete note or note does not exist'}), 404
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# Regular search status tracking
regular_search_status = {
    "active": False,
    "progress": 0,
    "current_step": "",
    "search_type": "regular",  # 'regular' or 'ai_enhanced'
    "query": "",
    "completed": False,
    "error": None,
    "session_id": None,
    "optimization_explanation": "",
    "results": []  # Store completed results here
}

#############################################
# Regular Search Status Route for AJAX polling
#############################################
@app.route("/search_status")
def get_search_status():
    """Returns the current status of a regular or AI enhanced search as JSON for polling."""
    global regular_search_status
    
    status_copy = dict(regular_search_status)  # Make a copy to avoid thread issues
    
    # Ensure all necessary fields are present
    if "progress" not in status_copy:
        status_copy["progress"] = 0
    
    if "current_step" not in status_copy:
        status_copy["current_step"] = "Initializing..."
    
    if "completed" not in status_copy:
        status_copy["completed"] = False
    
    return jsonify(status_copy)

#############################################
# Regular/AI Enhanced search function to run in background
#############################################
def regular_search_background_task(query, search_params, use_ai_enhanced=False):
    global regular_search_status
    
    try:
        # Store the original query for reference
        original_query = query.strip()
        search_type = "ai_enhanced" if use_ai_enhanced else "regular"
        
        session_id = regular_search_status["session_id"]
        
        print(f"\n==== STARTING {search_type.upper()} SEARCH FOR: '{original_query}' (Session: {session_id}) ====\n")
        
        # Step 1: Initialize search
        regular_search_status["current_step"] = "Initializing search"
        regular_search_status["progress"] = 10
        
        # Add a small delay to ensure the frontend can see the progress update
        time.sleep(0.2)
        
        # Step 2: Apply AI optimization if enabled
        if use_ai_enhanced:
            regular_search_status["current_step"] = "Optimizing search query with AI"
            regular_search_status["progress"] = 20
            
            try:
                actual_search_query, optimization_explanation = optimize_search_query(query)
                
                # Check if the search is still valid
                if regular_search_status["session_id"] != session_id:
                    print(f"Search session {session_id} was replaced. Terminating.")
                    return None, None
                
                regular_search_status["optimization_explanation"] = optimization_explanation
                print(f"Original query: '{query}'")
                print(f"Optimized query: '{actual_search_query}'")
                print(f"Explanation: {optimization_explanation}")
                
                regular_search_status["progress"] = 30
                regular_search_status["current_step"] = "Performing semantic search with optimized query"
            except Exception as e:
                print(f"Error optimizing query: {e}")
                # Fall back to original query if optimization fails
                actual_search_query = query
                regular_search_status["current_step"] = "Falling back to original query due to optimization error"
                regular_search_status["progress"] = 30
        else:
            actual_search_query = query
            regular_search_status["current_step"] = "Performing semantic search"
            regular_search_status["progress"] = 30
        
        # Short delay for UI update
        time.sleep(0.2)
        
        # Step 3: Perform semantic search
        initial_top_k = 50
        limit_for_reranking = 50
        
        # Update progress for semantic search
        regular_search_status["current_step"] = "Searching for games"
        regular_search_status["progress"] = 40
        
        raw_results = semantic_search_query(actual_search_query, top_k=initial_top_k)
        
        # Check if the search is still valid
        if regular_search_status["session_id"] != session_id:
            print(f"Search session {session_id} was replaced. Terminating.")
            return None, None
        
        if not raw_results:
            regular_search_status["current_step"] = "No results found for your query"
            regular_search_status["progress"] = 100
            regular_search_status["completed"] = True
            regular_search_status["error"] = "No results found for your search query."
            regular_search_status["results"] = []  # Store empty results
            return [], regular_search_status["optimization_explanation"]
        
        # Update progress for preparing candidates
        regular_search_status["current_step"] = "Preparing search results"
        regular_search_status["progress"] = 50
        
        # Step 4: Prepare candidates for potential LLM re-ranking
        candidates_for_reranking = []
        original_semantic_order_appids = []
        
        # Load summaries for AI data
        summaries_dict = load_summaries(SUMMARIES_FILE)
        
        for r in raw_results:
            appid = r.get("appid")
            if not appid: continue
            appid_int = int(appid)
            original_semantic_order_appids.append(appid_int)
            
            # Prepare candidate only if it's within the limit we send to the LLM
            if len(candidates_for_reranking) < limit_for_reranking:
                 # Get the actual game data to access more information if needed
                 game_data = None
                 if TESTING_ENABLE_SYNTHETIC_SUMMARIES:
                     game_data = get_game_data_by_appid(appid_int, STEAM_DATA_FILE, index_map)
                 
                 summary_obj = summaries_dict.get(appid_int, {})
                 ai_summary = summary_obj.get("ai_summary", "")
                 
                 if ai_summary:
                     # We have a real AI summary from the summaries file
                     candidates_for_reranking.append({"appid": appid_int, "ai_summary": ai_summary})
                 elif TESTING_ENABLE_SYNTHETIC_SUMMARIES and game_data:
                     # Create a synthetic summary for testing
                     name = game_data.get("name", "Unknown Game")
                     description = game_data.get("short_description", "No description available.")
                     synthetic_summary = f"SYNTHETIC SUMMARY FOR TESTING:\n{name} is a game on Steam. {description}"
                     candidates_for_reranking.append({
                         "appid": appid_int, 
                         "ai_summary": synthetic_summary
                     })
        
        # Check if the search is still valid
        if regular_search_status["session_id"] != session_id:
            print(f"Search session {session_id} was replaced. Terminating.")
            return None, None
        
        # Step 5: Perform LLM re-ranking if needed
        processing_order_appids = original_semantic_order_appids  # Default: semantic order
        
        if search_params["sort_by"] == "Relevance" and candidates_for_reranking:
            regular_search_status["current_step"] = "Re-ranking results with AI for better relevance"
            regular_search_status["progress"] = 60
            
            try:
                ordered_appids_from_llm, llm_comment = rerank_search_results(actual_search_query, candidates_for_reranking)
                
                # Check if the search is still valid
                if regular_search_status["session_id"] != session_id:
                    print(f"Search session {session_id} was replaced. Terminating.")
                    return None, None
                
                if ordered_appids_from_llm is not None:
                    regular_search_status["current_step"] = "AI re-ranking successful"
                    
                    # Create the new order: Start with LLM's order, then append remaining semantic results
                    llm_ordered_set = set(ordered_appids_from_llm)
                    remaining_semantic_appids = [appid for appid in original_semantic_order_appids if appid not in llm_ordered_set]
                    processing_order_appids = ordered_appids_from_llm + remaining_semantic_appids
                else:
                     regular_search_status["current_step"] = "AI re-ranking failed, using default ordering"
            except Exception as e:
                print(f"Exception during LLM re-ranking call: {e}")
                regular_search_status["current_step"] = "Error in AI re-ranking, using default ordering"
        
        # Step 6: Prepare final results with filtering
        regular_search_status["current_step"] = "Applying filters and finalizing results"
        regular_search_status["progress"] = 80
        
        results_dict = {}  # Use dict to store results before final sorting
        
        for appid in processing_order_appids:
            # Get full game data
            game_data = get_game_data_by_appid(appid, STEAM_DATA_FILE, index_map)
            if not game_data:
                continue
            
            # Extract data needed for filtering and display
            reviews = game_data.get("reviews", [])
            total_reviews = len(reviews)
            positive_count = sum(1 for review in reviews if review.get("voted_up"))
            pos_percent = (positive_count / total_reviews * 100) if total_reviews > 0 else 0
            
            # Extract media
            media = []
            if game_data.get("header_image"): media.append(force_https(game_data["header_image"]))
            if isinstance(game_data.get("screenshots"), list):
                for s in game_data["screenshots"]:
                    if isinstance(s, dict) and s.get("path_full"):
                        media.append(force_https(s["path_full"]))
                    else:
                        media.append(force_https(str(s)))
            
            store_data = game_data.get("store_data", {})
            if isinstance(store_data, dict):
                movies = store_data.get("movies", [])
                for movie in movies:
                    webm_max = movie.get("webm", {}).get("max")
                    mp4_max = movie.get("mp4", {}).get("max")
                    if webm_max:
                        media.append(force_https(webm_max))
                    elif mp4_max:
                        media.append(force_https(mp4_max))
                    else:
                        thumb = movie.get("thumbnail")
                        if thumb:
                            media.append(force_https(thumb))
            
            # Extract AI summary
            summary_obj = summaries_dict.get(appid, {})
            ai_summary = summary_obj.get("ai_summary", "")
            
            # Extract genres
            genres = []
            if "store_data" in game_data and isinstance(game_data["store_data"], dict):
                 genre_list = game_data["store_data"].get("genres", [])
                 genres = [g.get("description") for g in genre_list if g.get("description")]
            
            # Extract year
            release_date_str = game_data.get("release_date", "")
            year = "Unknown"
            if release_date_str:
                 try: year = release_date_str.split(",")[-1].strip()
                 except: pass
            
            # Extract platforms
            platforms = game_data.get("store_data", {}).get("platforms", {})
            
            # Extract price
            is_free = game_data.get("store_data", {}).get("is_free", False)
            price = 0.0
            if not is_free:
                 price_overview = game_data.get("store_data", {}).get("price_overview", {})
                 if price_overview: price = price_overview.get("final", 0) / 100.0
            
            # Apply Filters
            if search_params["genre"] != "All" and search_params["genre"] not in genres: continue
            if search_params["year"] != "All" and year != search_params["year"]: continue
            if search_params["platform"] != "All":
                platform_key = search_params["platform"].lower()
                if not platforms.get(platform_key, False): continue
            if search_params["price"] == "Free" and not is_free: continue
            if search_params["price"] == "Paid" and is_free: continue
            
            # If filters pass, store the result
            results_dict[appid] = {
                "appid": appid,
                "name": game_data.get("name", "Unknown"),
                "media": media,
                "genres": genres,
                "release_year": year,
                "platforms": platforms,
                "is_free": is_free,
                "price": price,
                "pos_percent": pos_percent,
                "total_reviews": total_reviews,
                "ai_summary": ai_summary
            }
        
        # Check if the search is still valid
        if regular_search_status["session_id"] != session_id:
            print(f"Search session {session_id} was replaced. Terminating.")
            return None, None
        
        # Create the final list, respecting the processing order
        final_results = [results_dict[appid] for appid in processing_order_appids if appid in results_dict]
        
        # Apply final explicit sorting ONLY if the user chose something other than "Relevance"
        if search_params["sort_by"] != "Relevance":
            regular_search_status["current_step"] = f"Sorting results by {search_params['sort_by']}"
            
            if search_params["sort_by"] == "Name (A-Z)":
                final_results.sort(key=lambda x: x["name"])
            elif search_params["sort_by"] == "Release Date (Newest)":
                final_results.sort(key=lambda x: int(x["release_year"]) if x["release_year"].isdigit() else 0, reverse=True)
            elif search_params["sort_by"] == "Release Date (Oldest)":
                final_results.sort(key=lambda x: int(x["release_year"]) if x["release_year"].isdigit() else float('inf'))
            elif search_params["sort_by"] == "Price (Low to High)":
                final_results.sort(key=lambda x: x["price"])
            elif search_params["sort_by"] == "Price (High to Low)":
                final_results.sort(key=lambda x: x["price"], reverse=True)
            elif search_params["sort_by"] == "Review Count (High to Low)":
                final_results.sort(key=lambda x: x["total_reviews"], reverse=True)
            elif search_params["sort_by"] == "Positive Review % (High to Low)":
                final_results.sort(key=lambda x: x["pos_percent"], reverse=True)
        
        # Limit the final results based on the user's selection
        if search_params["result_limit"] and search_params["result_limit"] < len(final_results):
            final_results = final_results[:search_params["result_limit"]]
        
        # Mark the search as completed
        regular_search_status["progress"] = 100
        regular_search_status["current_step"] = "Search completed successfully"
        regular_search_status["completed"] = True
        regular_search_status["results"] = final_results  # Store the results for later retrieval
        
        print(f"==== {search_type.upper()} SEARCH COMPLETED FOR: '{original_query}' (Session: {session_id}) ====")
        print(f"Found {len(final_results)} results")
        
        return final_results, regular_search_status["optimization_explanation"]
        
    except Exception as e:
        print(f"Unexpected error in search background task: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Update status to show the error
        regular_search_status.update({
            "error": f"An unexpected error occurred: {str(e)}",
            "progress": 100,
            "current_step": "Error occurred",
            "completed": True
        })
        
        return [], ""

if __name__ == "__main__":
    # Make sure debug is True for development logging
    app.run(host='0.0.0.0', debug=True, threaded=True)
