import os
import json
import pickle
import logging
import sys

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Cache file for the index map
INDEX_CACHE_FILE = "data/index_map.pkl"

def build_steam_data_index(file_path: str) -> dict:
    """Builds an index map (appid -> file offset) for the large JSONL file.
       Uses a cache file to avoid re–scanning the file if it hasn't changed.
    """
    if os.path.exists(INDEX_CACHE_FILE):
        data_mtime = os.path.getmtime(file_path)
        cache_mtime = os.path.getmtime(INDEX_CACHE_FILE)
        if cache_mtime >= data_mtime:
            logging.info("Loading index map from cache...")
            with open(INDEX_CACHE_FILE, "rb") as f:
                return pickle.load(f)
    logging.info("Building index map from data file...")
    index_map = {}
    with open(file_path, "r", encoding="utf-8") as f:
        offset = 0
        line = f.readline()
        while line:
            try:
                data = json.loads(line)
                appid = data.get("appid")
                if appid is not None:
                    index_map[int(appid)] = offset
            except Exception as e:
                logging.warning(f"Error parsing line at offset {offset}: {e}")
            offset = f.tell()
            line = f.readline()
    with open(INDEX_CACHE_FILE, "wb") as f:
        pickle.dump(index_map, f)
    logging.info("Index map built and cached with %d entries.", len(index_map))
    return index_map

def load_summaries(file_path: str) -> dict:
    """Loads the AI summaries file fully into memory."""
    summaries_dict = {}
    if not os.path.exists(file_path):
        logging.warning(f"Summaries file not found: {file_path}")
        print(f"WARNING: Summaries file not found at {file_path}")
        return summaries_dict
    
    # Add file size check for debugging
    file_size = os.path.getsize(file_path)
    print(f"Found summaries file at {file_path} (Size: {file_size/1024:.2f} KB)")
    
    loaded_count = 0
    error_count = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                appid = obj.get("appid")
                if appid is not None:
                    summaries_dict[int(appid)] = obj
                    loaded_count += 1
                else:
                    error_count += 1
                    logging.warning(f"Missing appid in summary at line {line_num}")
            except Exception as e:
                error_count += 1
                logging.warning(f"Error parsing summary at line {line_num}: {e}")
    
    print(f"Loaded {loaded_count} summaries, encountered {error_count} errors")
    if loaded_count > 0:
        # Print a sample of loaded appids for verification
        sample_keys = list(summaries_dict.keys())[:5]
        print(f"Sample appids in summaries: {sample_keys}")
    else:
        print("WARNING: No summaries were loaded. Re-ranking will not work!")
        
    logging.info(f"Loaded {len(summaries_dict)} summaries.")
    return summaries_dict

def get_game_data_by_appid(appid: int, file_path: str, index_map: dict) -> dict:
    """Random-access lookup of game data from the large JSONL file using the pre-built index."""
    offset = index_map.get(appid)
    if offset is None:
        logger.warning(f"AppID {appid} not found in index map (total entries: {len(index_map)})")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
            return json.loads(line)
    except Exception as e:
        logger.error(f"Failed to load game data for appid {appid}: {e}")
        return None
