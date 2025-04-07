# robust_csv.py (Verbeterd: Eerst sniffen, dan frequentie)
import csv
import chardet
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def robust_csv_cleaner(file_path):
    """
    Reads and cleans an arbitrary CSV file more robustly.
    Improved delimiter detection (Sniffer > Frequency).
    Uses C engine with error handling.
    """
    logger.info(f"Starting robust cleaning for: {file_path}")

    # 1. Detect Encoding
    try:
        with open(file_path, 'rb') as f:
            sample_size = min(os.path.getsize(file_path), 65536)
            raw_data = f.read(sample_size)
            if not raw_data: logger.warning(f"File is empty: {file_path}"); return pd.DataFrame()
        detection_result = chardet.detect(raw_data)
        encoding = detection_result['encoding'] if detection_result['encoding'] else 'utf-8'
        confidence = detection_result['confidence'] if detection_result['confidence'] else 0.0
        logger.info(f"Detected encoding: {encoding} (Confidence: {confidence:.2f})")
        if confidence < 0.7: logger.warning(f"Low encoding confidence ({confidence:.2f}).")
    except FileNotFoundError: logger.error(f"File not found: {file_path}"); return None
    except Exception as e: logger.error(f"Error detecting encoding: {e}", exc_info=True); encoding = 'utf-8'; logger.warning("Falling back to utf-8.")

    # 2. Sniff/Detect Delimiter (Verbeterde Logica)
    delimiter = '|' # Start met een default
    try:
        sample_text = raw_data[:8192].decode(encoding, errors='replace') # Iets grotere sample voor sniffen
        if sample_text:
            # --- EERST PROBEREN TE SNIFFEN ---
            try:
                # Sniff met de meest waarschijnlijke delimiters
                dialect = csv.Sniffer().sniff(sample_text, delimiters=[',', ';', '|', '\t'])
                delimiter = dialect.delimiter
                logger.info(f"Detected delimiter via sniffing: '{delimiter}'")
            except csv.Error as sniff_err:
                logger.warning(f"CSV Sniffer failed: {sniff_err}. Falling back to frequency count.")
                # --- FALLBACK: FREQUENTIE TELLEN ---
                counts = {
                    ',': sample_text.count(','),
                    ';': sample_text.count(';'),
                    '|': sample_text.count('|'),
                    '\t': sample_text.count('\t')
                }
                # Vind de delimiter met de hoogste count
                likely_delimiter = max(counts, key=counts.get)
                if counts[likely_delimiter] > 0:
                    delimiter = likely_delimiter
                    logger.info(f"Using most frequent delimiter as fallback: '{delimiter}' (Count: {counts[delimiter]})")
                else:
                    logger.warning(f"No common delimiters found frequently after failed sniff. Using default '{delimiter}'.")
        else:
            logger.warning(f"Empty sample for delimiter sniffing. Using default '{delimiter}'.")
    except Exception as e:
        logger.error(f"Unexpected error during delimiter detection: {e}", exc_info=True)
        logger.warning(f"Falling back to default delimiter: '{delimiter}'.")

    # 3. Read CSV with Pandas (C-engine, encoding_errors, quote settings)
    df = None
    try:
        read_params = {
            'filepath_or_buffer': file_path, 'encoding': encoding, 'sep': delimiter,
            'dtype': str, 'on_bad_lines': 'skip', 'skipinitialspace': True,
            'quotechar': None, 'quoting': csv.QUOTE_NONE, 'encoding_errors': 'replace'
        }
        logger.info(f"Reading CSV with pandas using parameters: { {k:v for k,v in read_params.items() if k != 'filepath_or_buffer'} }")
        df = pd.read_csv(**read_params)
        file_size = os.path.getsize(file_path)
        if df.empty and file_size > 0:
             logger.warning(f"Pandas read_csv resulted in empty DataFrame for non-empty file {file_path} (Size: {file_size} bytes). Check parameters (sep='{delimiter}', quoting=None) or file content.")
    except FileNotFoundError: logger.error(f"File not found during pandas read: {file_path}"); return None
    except pd.errors.EmptyDataError: logger.warning(f"CSV file is empty: {file_path}"); return pd.DataFrame()
    except ValueError as ve:
        if 'quoting' in str(ve).lower() or 'quotechar' in str(ve).lower():
             logger.error(f"Pandas read_csv failed likely due to quoting conflict (quoting=NONE with actual quotes?): {ve}", exc_info=False)
             logger.warning("Attempting fallback read with default quoting...")
             try:
                  fallback_params = {
                       'filepath_or_buffer': file_path, 'encoding': encoding, 'sep': delimiter, 'dtype': str,
                       'on_bad_lines': 'skip', 'skipinitialspace': True, 'encoding_errors': 'replace'
                  }
                  logger.info(f"Reading CSV with fallback parameters: { {k:v for k,v in fallback_params.items() if k != 'filepath_or_buffer'} }")
                  df = pd.read_csv(**fallback_params)
                  logger.info(f"Fallback read successful with default quoting. Shape: {df.shape}")
                  # Als de fallback slaagt, MOETEN we de kolomnamen opnieuw cleanen,
                  # want die kunnen nu wel dubbele quotes bevatten die eerder genegeerd werden.
                  if df is not None and not df.empty:
                      logger.info("Re-cleaning column names after fallback read.")
                      try:
                          cleaned_columns = [str(col).strip().strip('"').strip() for col in df.columns]
                          df.columns = cleaned_columns
                      except Exception as col_e:
                          logger.error(f"Error re-cleaning column names after fallback: {col_e}")

             except Exception as fallback_e:
                  logger.error(f"Fallback CSV read also failed: {fallback_e}", exc_info=True)
                  return None # Geef op na fallback
        else: logger.error(f"Pandas read_csv failed with ValueError: {ve}", exc_info=True); return None
    except Exception as e: logger.error(f"Pandas failed to read CSV file {file_path} (encoding='{encoding}', sep='{delimiter}'): {e}", exc_info=True); return None

    if df is None: return None # Check na mogelijke fallback failure

    # 4. Clean Column Names (Alleen nodig als fallback *niet* is gebruikt, anders hierboven al gedaan)
    # We doen het voor de zekerheid altijd, tenzij de fallback specifiek is uitgevoerd.
    if 'fallback_params' not in locals(): # Check of fallback is uitgevoerd
        try:
            cleaned_columns = []; original_columns = list(df.columns)
            needs_cleaning = False
            for col in original_columns:
                cleaned_col = str(col).strip().strip('"').strip()
                if cleaned_col != str(col):
                    needs_cleaning = True
                cleaned_columns.append(cleaned_col)

            if needs_cleaning:
                logger.info(f"Cleaning column names: {original_columns} -> {cleaned_columns}")
                df.columns = cleaned_columns
            else:
                logger.info("Column names appear to be clean.")
        except Exception as e:
             logger.error(f"Error cleaning column names: {e}", exc_info=True)

    # 5. Clean Cell Contents
    def strip_value(value): return value.strip() if isinstance(value, str) else value
    logger.info("Stripping leading/trailing whitespace from all cell values...")
    try:
        # Gebruik .map op het hele dataframe indien mogelijk (pandas >= 1.1?)
        # Anders per kolom
        if hasattr(df, 'map') and callable(df.map):
             df = df.map(strip_value)
        else:
             for col in df.columns:
                  df[col] = df[col].map(strip_value, na_action='ignore')

    except Exception as e: logger.error(f"Error stripping whitespace from cells: {e}", exc_info=True)

    logger.info(f"Finished cleaning for: {file_path}. Final shape: {df.shape}")
    return df