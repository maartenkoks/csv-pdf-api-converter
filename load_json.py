import pandas as pd
import json
import logging

def load_json(file_path):
    """
    Load a JSON file (array of objects or NDJSON) into a pandas DataFrame.
    Returns DataFrame on success, None on failure.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                # Standard JSON array
                data = json.load(f)
                if not isinstance(data, list):
                    logging.error('JSON root is not a list.')
                    return None
                return pd.DataFrame(data)
            else:
                # NDJSON (newline-delimited JSON)
                lines = f.readlines()
                records = [json.loads(line) for line in lines if line.strip()]
                return pd.DataFrame(records)
    except Exception as e:
        logging.error(f'Failed to load JSON: {e}')
        return None
