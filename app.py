import os
import subprocess
import time
import pandas as pd
import requests
import json
import re
import logging
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, session, send_from_directory, flash
)
from werkzeug.utils import secure_filename
from robust_csv import robust_csv_cleaner
from load_json import load_json
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, AuthenticationError

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "verander_mij_in_productie_met_iets_willekeurigs_!")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp_uploads'
app.config['DATA_FOLDER'] = 'data'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = app.logger

# Maak mappen aan als ze nog niet bestaan
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

# --- Globale variabelen ---
DATAFRAME_STORAGE = None
SELECTED_COLS = []
COL_DESCRIPTIONS = {}
MULTI_SEARCH_PARAM_NAME = None
MULTI_SEARCH_COLS = []
COL_TYPES_CACHE = {}
COL_SAMPLES_CACHE = {}
NGROK_PROCESS = None
NGROK_URL = None

# --- OpenAI Client Initialisatie (optioneel) ---
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key or not openai_api_key.startswith("sk-"):
    logger.warning("OPENAI_API_KEY not found/invalid. LLM features disabled.")
    client = None
else:
    try:
        client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Failed to init OpenAI client: {e}", exc_info=True)
        client = None

# --- LLM Functie (optioneel) ---
def generate_column_descriptions_with_llm(context, columns, samples, max_sample_length=150):
    if not client:
        logger.error("OpenAI client unavailable.")
        return {}
    prompt_lines = [
        f"You are an AI data consultant analyzing a CSV dataset.",
        f"User context: '{context if context else 'None provided'}'",
        f"\nColumns & sample values (samples truncated to {max_sample_length} chars):"
    ]
    total_token_estimate = len(context) // 4
    for col in columns:
        unique_non_null = [s for s in samples.get(col, []) if pd.notna(s)][:5]
        sample_vals = [str(s).strip() for s in unique_non_null if str(s).strip()]
        truncated_samples = [
            s[:max_sample_length] + "..." if len(s) > max_sample_length else s
            for s in sample_vals
        ]
        sample_vals_str = ", ".join(f'"{v}"' for v in truncated_samples) if truncated_samples else "[empty/no distinct values]"
        line = f"- Column '{col}': Samples: {sample_vals_str}"
        prompt_lines.append(line)
        total_token_estimate += len(col) // 4 + len(sample_vals_str) // 4
    prompt_lines.extend([
        "\nBased on column name, samples, and context, provide a concise, one-sentence description for EACH column for API filtering.",
        'Return ONLY a valid JSON object like: {"col1": "Desc 1", "col2": "Desc 2", ...}',
        "Ensure output is *only* the JSON object."
    ])
    full_prompt = "\n".join(prompt_lines)
    logger.debug(f"LLM prompt (~{total_token_estimate} tokens): {full_prompt[:1000]}...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate JSON column descriptions."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        raw_content = response.choices[0].message.content.strip()
        logger.debug(f"Raw LLM response: {raw_content}")
        try:
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = raw_content.strip().strip('```json').strip('`').strip()
            descriptions_dict = json.loads(json_str)
            if not isinstance(descriptions_dict, dict):
                raise ValueError("LLM response not dict.")
            final_descriptions = {k: v for k, v in descriptions_dict.items() if v and isinstance(v, str)}
            missing_cols = [col for col in columns if col not in final_descriptions]
            if missing_cols:
                logger.warning(f"LLM missing descriptions for: {missing_cols}")
            logger.info(f"Parsed LLM descriptions for {len(final_descriptions)} columns.")
            return final_descriptions
        except Exception as json_err:
            logger.error(f"Failed parse/validate JSON: {json_err}\nContent: {raw_content}")
            return {}
    except Exception as e:
        logger.error(f"OpenAI call error: {e}", exc_info=True)
        return {}

# --- Route: Index ---
@app.route("/")
def index():
    global DATAFRAME_STORAGE, SELECTED_COLS, COL_DESCRIPTIONS, MULTI_SEARCH_PARAM_NAME, MULTI_SEARCH_COLS, COL_TYPES_CACHE, COL_SAMPLES_CACHE, NGROK_URL
    DATAFRAME_STORAGE = None
    SELECTED_COLS = []
    COL_DESCRIPTIONS = {}
    MULTI_SEARCH_PARAM_NAME = None
    MULTI_SEARCH_COLS = []
    COL_TYPES_CACHE = {}
    COL_SAMPLES_CACHE = {}
    NGROK_URL = None
    session.clear()
    return render_template("index.html")

# --- Hulpfunctie: Create Slug ---
def create_slug(filename):
    base_name = os.path.splitext(filename)[0].replace('temp_', '')
    slug = re.sub(r'\W+', '_', base_name).strip('_').lower()
    return slug or "data"

# --- Functie: analyze_columns_for_display ---
def analyze_columns_for_display(df):
    global COL_TYPES_CACHE, COL_SAMPLES_CACHE
    if df is None or df.empty:
        COL_TYPES_CACHE = {}
        COL_SAMPLES_CACHE = {}
        return

    COL_TYPES_CACHE = {}
    COL_SAMPLES_CACHE = {}
    for col in df.columns:
        try:
            unique_non_null = df[col].dropna().unique()
            COL_SAMPLES_CACHE[col] = [str(v) for v in unique_non_null[:5]]
        except Exception as e:
            logger.warning(f"Unique vals err '{col}': {e}")
            COL_SAMPLES_CACHE[col] = []
            unique_non_null = pd.Series([])

        sample_serie = df[col].dropna().head(1000)
        if sample_serie.empty:
            COL_TYPES_CACHE[col] = "empty"
            continue

        # Check boolean
        possible_bool_map = {
            "true": True, "1": True, "yes": True, "ja": True, "t": True,
            "false": False, "0": False, "no": False, "nee": False, "f": False
        }
        unique_strs_lower = {
            str(x).strip().lower()
            for x in unique_non_null if pd.notna(x)
        }
        if unique_strs_lower and unique_strs_lower.issubset(possible_bool_map.keys()):
            if len({possible_bool_map[s] for s in unique_strs_lower}) <= 2:
                COL_TYPES_CACHE[col] = "boolean"
                continue

        # Check numeriek
        try:
            numeric_series = pd.to_numeric(sample_serie, errors="coerce")
            if numeric_series.notna().all():
                is_close = ((numeric_series - numeric_series.round()).abs() < 1e-9).all()
                if is_close:
                    if sample_serie.astype(str).str.match(r"^-?\d+$").all():
                        COL_TYPES_CACHE[col] = "integer"
                    else:
                        COL_TYPES_CACHE[col] = "number"
                    continue
                else:
                    COL_TYPES_CACHE[col] = "number"
                    continue
            else:
                COL_TYPES_CACHE[col] = "string"
                continue
        except:
            COL_TYPES_CACHE[col] = "string"
            continue

        if col not in COL_TYPES_CACHE:
            COL_TYPES_CACHE[col] = "string"

# --- Functie: Guess Description ---
def guess_description(col_name):
    try:
        if not isinstance(col_name, str):
            col_name = str(col_name)
        original_col_name = col_name
        c_lower = col_name.lower().strip()
        if not c_lower:
            return "Filter by this column (empty name)."

        id_pattern = r"(^id_|_id$|identifier|\Bid\B|_nr$|^nr_)"
        if re.search(id_pattern, c_lower):
            base_name = re.sub(id_pattern, "", original_col_name, flags=re.IGNORECASE).strip().strip("_- .")
            return (
                f"Filter by unique ID for '{base_name}'."
                if base_name
                else f"Filter by unique ID ('{original_col_name}')."
            )

        if "city" in c_lower or "stad" in c_lower or "plaats" in c_lower or "gemeente" in c_lower:
            return f"Filter by city/municipality ('{original_col_name}')."
        if "zip" in c_lower or "postcode" in c_lower or "postal" in c_lower:
            return f"Filter by postal code ('{original_col_name}')."
        if "country" in c_lower or "land" in c_lower:
            return f"Filter by country ('{original_col_name}')."
        if "region" in c_lower or "provincie" in c_lower or "state" in c_lower or "gewest" in c_lower:
            return f"Filter by region/province/state ('{original_col_name}')."
        if "mail" in c_lower:
            return f"Filter by email ('{original_col_name}')."
        if "phone" in c_lower or "telefoon" in c_lower or "gsm" in c_lower:
            return f"Filter by phone number ('{original_col_name}')."
        if "url" in c_lower or "website" in c_lower or "link" in c_lower:
            return f"Filter by URL/website ('{original_col_name}')."
        if (
            "date" in c_lower
            or "datum" in c_lower
            or "_dt" in c_lower
            or c_lower.endswith("day")
            or "time" in c_lower
            or "stamp" in c_lower
            or "jaar" in c_lower
            or "year" in c_lower
            or "maand" in c_lower
            or "month" in c_lower
        ):
            return f"Filter by date/time ('{original_col_name}')."
        if (
            "wage" in c_lower
            or "loon" in c_lower
            or "salaris" in c_lower
            or "salary" in c_lower
            or "amount" in c_lower
            or "price" in c_lower
            or "value" in c_lower
            or "waarde" in c_lower
            or "bedrag" in c_lower
            or "cost" in c_lower
        ):
            return f"Filter by amount/price/wage ('{original_col_name}')."
        if "nummer" in c_lower or "number" in c_lower or "count" in c_lower or "aantal" in c_lower:
            return f"Filter by number/count ('{original_col_name}')."
        if "min" in c_lower:
            return f"Filter by minimum value ('{original_col_name}')."
        if "max" in c_lower:
            return f"Filter by maximum value ('{original_col_name}')."
        if "status" in c_lower or ("state" in c_lower and "status" in c_lower):
            return f"Filter by status/state ('{original_col_name}')."
        if "type" in c_lower or "soort" in c_lower:
            return f"Filter by type ('{original_col_name}')."
        if "lang" in c_lower or "taal" in c_lower:
            return f"Filter by language ('{original_col_name}')."
        if "code" in c_lower:
            return f"Filter by code ('{original_col_name}')."
        if "category" in c_lower or "categorie" in c_lower:
            return f"Filter by category ('{original_col_name}')."
        if "title" in c_lower or "titel" in c_lower:
            return f"Filter by title ('{original_col_name}')."
        if "desc" in c_lower or "omschr" in c_lower or "text" in c_lower or "tekst" in c_lower:
            return f"Filter by description/text ('{original_col_name}')."
        if "name" in c_lower or "naam" in c_lower:
            return f"Filter by name ('{original_col_name}')."
        if c_lower == "y" or c_lower == "n" or "flag" in c_lower or c_lower.startswith("is_") or c_lower.startswith("has_") or "publish" in c_lower:
            return f"Filter by boolean flag ('{original_col_name}')."
        return f"Filter by value in '{original_col_name}'."
    except Exception as e:
        logger.error(f"Guess desc error '{col_name}': {e}")
        return f"Filter by value in '{col_name}'."

# --- Hulpfunctie: Parse Integer Parameter ---
def parse_int_param(param_value, default, min_val, max_val):
    """Parses an integer parameter, clamps it, and returns default on error."""
    if param_value is None:
        return default
    try:
        val = int(param_value)
        return max(min_val, min(val, max_val))
    except (ValueError, TypeError):
        return default

# --- Route: Upload (CSV, PDF, JSON) ---
@app.route("/upload", methods=["POST"])
def upload_file():
    global DATAFRAME_STORAGE, SELECTED_COLS, COL_DESCRIPTIONS, MULTI_SEARCH_PARAM_NAME, MULTI_SEARCH_COLS, COL_TYPES_CACHE, COL_SAMPLES_CACHE, NGROK_URL

    file_type = request.form.get("file_type", "csv").lower()
    uploaded_file = request.files.get("file")
    dataset_context = request.form.get("dataset_context", "").strip()

    if not uploaded_file or uploaded_file.filename == "":
        logger.warning("Upload: No file selected.")
        return redirect(url_for("index"))

    filename = secure_filename(uploaded_file.filename)
    session["uploaded_filename"] = filename
    session["file_type"] = file_type

    # --- CSV-verwerking ---
    if file_type == "csv":
        session["dataset_context"] = dataset_context
        session["current_dataset_slug"] = create_slug(filename)
        temp_path = os.path.join(app.config["TEMP_FOLDER"], f"temp_{filename}")
        try:
            uploaded_file.save(temp_path)
            logger.info(f"CSV saved temporarily: {temp_path}")
            df = robust_csv_cleaner(temp_path)
            if df is None:
                logger.error(f"CSV cleaner failed: {temp_path}.")
                return redirect(url_for("index"))
            if df.empty and os.path.getsize(temp_path) > 0:
                logger.warning(f"CSV cleaner empty DF: {temp_path}.")

            DATAFRAME_STORAGE = df
            COL_TYPES_CACHE = {}
            COL_SAMPLES_CACHE = {}
            SELECTED_COLS = []
            MULTI_SEARCH_PARAM_NAME = None
            MULTI_SEARCH_COLS = []
            COL_DESCRIPTIONS = {}
            analyze_columns_for_display(DATAFRAME_STORAGE)
            logger.info("Column analysis complete.")

            if not DATAFRAME_STORAGE.empty:
                COL_DESCRIPTIONS = generate_column_descriptions_with_llm(
                    context=dataset_context,
                    columns=list(DATAFRAME_STORAGE.columns),
                    samples=COL_SAMPLES_CACHE,
                )
                logger.info(f"LLM descriptions: {len(COL_DESCRIPTIONS)} generated.")
            else:
                logger.info("Skipping LLM for empty DF.")

            # Verwijder eventueel input.json
            json_path = os.path.join(app.config["DATA_FOLDER"], "input.json")
            if os.path.exists(json_path):
                try:
                    os.remove(json_path)
                    logger.info(f"Removed old JSON file at {json_path}")
                except Exception as e_rem:
                    logger.warning(f"Error removing old JSON: {e_rem}")

            return redirect(url_for("analyze"))
        except Exception as e:
            logger.error(f"Error processing CSV '{filename}': {e}", exc_info=True)
            return redirect(url_for("index"))
        finally:
            if "temp_path" in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"Removed temp CSV: {temp_path}")
                except OSError as e_rem:
                    logger.error(f"Error removing temp CSV {temp_path}: {e_rem}")

    # --- PDF-verwerking ---
    elif file_type == "pdf":
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            # Reset oude DataFrame
            DATAFRAME_STORAGE = None
            SELECTED_COLS = []
            COL_DESCRIPTIONS = {}
            MULTI_SEARCH_PARAM_NAME = None
            MULTI_SEARCH_COLS = []
            COL_TYPES_CACHE = {}
            COL_SAMPLES_CACHE = {}
            session.pop("current_dataset_slug", None)

            uploaded_file.save(pdf_path)
            logger.info(f"PDF saved directly: {pdf_path}")

            # Verwijder eventueel input.json
            json_path = os.path.join(app.config["DATA_FOLDER"], "input.json")
            if os.path.exists(json_path):
                try:
                    os.remove(json_path)
                    logger.info(f"Removed old JSON file at {json_path}")
                except Exception as e_rem:
                    logger.warning(f"Error removing old JSON: {e_rem}")

            return redirect(url_for("generate_pdf"))
        except Exception as e:
            logger.error(f"Error saving PDF '{filename}': {e}", exc_info=True)
            return redirect(url_for("index"))

    # --- JSON-verwerking ---
    elif file_type == "json":
        # Sla JSON bestand op
        json_dir = os.path.join(app.config["DATA_FOLDER"])
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, "input.json")
        uploaded_file.save(json_path)
        logger.info(f"JSON saved: {json_path}")

        # Laad JSON in DataFrame
        df = load_json(json_path)
        if df is None:
            logger.error(f"JSON load failed: {json_path}")
            # Verwijder ongeldig JSON-bestand
            try:
                os.remove(json_path)
            except Exception:
                pass
            return redirect(url_for("index"))

        DATAFRAME_STORAGE = df
        COL_TYPES_CACHE = {}
        COL_SAMPLES_CACHE = {}
        SELECTED_COLS = list(df.columns)
        MULTI_SEARCH_PARAM_NAME = None
        MULTI_SEARCH_COLS = []
        COL_DESCRIPTIONS = {}
        analyze_columns_for_display(DATAFRAME_STORAGE)
        logger.info(f"JSON DataFrame loaded: {DATAFRAME_STORAGE.shape}")

        # Verwijder eventueel input.csv
        csv_path = os.path.join(json_dir, "input.csv")
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
                logger.info(f"Removed old CSV file at {csv_path}")
            except Exception as e_rem:
                logger.warning(f"Error removing old CSV: {e_rem}")

        return redirect(url_for("analyze"))

    else:
        logger.warning(f"Invalid file type: {file_type}")
        return redirect(url_for("index"))

# --- Route: Analyze (CSV & JSON) ---
@app.route("/analyze")
def analyze():
    global DATAFRAME_STORAGE, SELECTED_COLS, MULTI_SEARCH_PARAM_NAME, MULTI_SEARCH_COLS, COL_TYPES_CACHE, COL_SAMPLES_CACHE, COL_DESCRIPTIONS
    if DATAFRAME_STORAGE is None or session.get("file_type") not in ["csv", "json"]:
        return redirect(url_for("index"))

    if not COL_TYPES_CACHE or not COL_SAMPLES_CACHE:
        analyze_columns_for_display(DATAFRAME_STORAGE)

    all_columns = list(DATAFRAME_STORAGE.columns)
    full_descriptions = {
        col: COL_DESCRIPTIONS.get(col) or guess_description(col)
        for col in all_columns
    }
    logger.debug(f"Rendering analyze.html: {len(all_columns)} cols")
    return render_template(
        "analyze.html",
        all_columns=all_columns,
        col_types=COL_TYPES_CACHE,
        samples=COL_SAMPLES_CACHE,
        col_descriptions=full_descriptions,
        current_selected_cols=SELECTED_COLS,
        current_multi_param_name=MULTI_SEARCH_PARAM_NAME,
        current_multi_selected_cols=MULTI_SEARCH_COLS
    )

# --- Route: Select Columns & Multi-Search Config ---
@app.route("/select_columns_and_configure_multi", methods=["POST"])
def select_columns_and_configure_multi():
    global SELECTED_COLS, MULTI_SEARCH_PARAM_NAME, MULTI_SEARCH_COLS, DATAFRAME_STORAGE
    if DATAFRAME_STORAGE is None or session.get("file_type") not in ["csv", "json"]:
        return redirect(url_for("index"))

    SELECTED_COLS = request.form.getlist("selected_cols")
    logger.info(f"Selected cols: {SELECTED_COLS}")
    enable_multi = request.form.get("enable_multi_search")

    if enable_multi:
        param_name = request.form.get("multi_param_name", "query").strip()
        param_name_sanitized = re.sub(r"\W+", "", param_name) or "query"
        if param_name_sanitized in SELECTED_COLS:
            param_name_sanitized = f"{param_name_sanitized}_multi"
            logger.warning(f"Multi param rename: '{param_name_sanitized}'.")

        MULTI_SEARCH_PARAM_NAME = param_name_sanitized
        MULTI_SEARCH_COLS = request.form.getlist("multi_search_target_cols")
        all_df_columns = list(DATAFRAME_STORAGE.columns)
        valid_multi_cols = [col for col in MULTI_SEARCH_COLS if col in all_df_columns]
        if len(valid_multi_cols) != len(MULTI_SEARCH_COLS):
            logger.warning(f"Invalid multi cols ignored: {set(MULTI_SEARCH_COLS) - set(valid_multi_cols)}")
            MULTI_SEARCH_COLS = valid_multi_cols

        if not MULTI_SEARCH_COLS:
            logger.warning("Multi enabled, no valid cols. Disabling.")
            MULTI_SEARCH_PARAM_NAME = None
        else:
            logger.info(f"Multi configured: param='{MULTI_SEARCH_PARAM_NAME}', cols={MULTI_SEARCH_COLS}")
    else:
        MULTI_SEARCH_PARAM_NAME = None
        MULTI_SEARCH_COLS = []
        logger.info("Multi disabled.")

    all_df_columns = list(DATAFRAME_STORAGE.columns)
    valid_selected_cols = [col for col in SELECTED_COLS if col in all_df_columns]
    if len(valid_selected_cols) != len(SELECTED_COLS):
        logger.warning(f"Invalid selected cols ignored: {set(SELECTED_COLS) - set(valid_selected_cols)}")
        SELECTED_COLS = valid_selected_cols

    return redirect(url_for("generate_api"))

# --- Route: Generate API Info Page ---
@app.route("/generate")
def generate_api():
    global DATAFRAME_STORAGE, SELECTED_COLS, COL_DESCRIPTIONS, COL_TYPES_CACHE, COL_SAMPLES_CACHE, NGROK_URL, MULTI_SEARCH_PARAM_NAME, MULTI_SEARCH_COLS
    if DATAFRAME_STORAGE is None or session.get("file_type") not in ["csv", "json"]:
        return redirect(url_for("index"))

    final_descriptions = {
        col: COL_DESCRIPTIONS.get(col) or guess_description(col)
        for col in SELECTED_COLS
    }
    multi_col_details = {}
    if MULTI_SEARCH_PARAM_NAME and MULTI_SEARCH_COLS:
        for col in MULTI_SEARCH_COLS:
            multi_col_details[col] = {
                "description": COL_DESCRIPTIONS.get(col) or guess_description(col),
                "type": COL_TYPES_CACHE.get(col, "Unknown"),
                "samples": COL_SAMPLES_CACHE.get(col, []),
            }

    return render_template(
        "generate.html",
        selected_cols=SELECTED_COLS,
        descriptions=final_descriptions,
        col_types=COL_TYPES_CACHE,
        samples=COL_SAMPLES_CACHE,
        ngrok_url=NGROK_URL,
        multi_param_name=MULTI_SEARCH_PARAM_NAME,
        multi_search_cols=MULTI_SEARCH_COLS,
        multi_col_details=multi_col_details
    )

# --- API Endpoint: /api/data/<slug> ---
@app.route("/api/data/<string:dataset_slug>", methods=["GET"])
def api_data_endpoint(dataset_slug=None):
    global DATAFRAME_STORAGE, SELECTED_COLS, MULTI_SEARCH_PARAM_NAME, MULTI_SEARCH_COLS, COL_TYPES_CACHE

    if DATAFRAME_STORAGE is None:
        logger.warning("/api/data: No data loaded.")
        return jsonify({"error": "No data currently loaded."}), 400

    try:
        df_filtered = DATAFRAME_STORAGE.copy()
        original_columns = list(df_filtered.columns)
        query_params_used = {}

        allowed_params = set(SELECTED_COLS)
        if MULTI_SEARCH_PARAM_NAME:
            allowed_params.add(MULTI_SEARCH_PARAM_NAME)
        always_allowed_params = {"limit", "offset", "_sort", "_order"}

        invalid_params = [p for p in request.args if p not in allowed_params and p not in always_allowed_params]
        if invalid_params:
            return jsonify({"error": f"Invalid param(s): {', '.join(invalid_params)}"}), 400

        # Kolomfilters
        for col in SELECTED_COLS:
            if col in request.args:
                val = request.args.get(col)
                query_params_used[col] = val
                try:
                    df_filtered = df_filtered[
                        df_filtered[col].astype(str).str.contains(val, case=False, na=False, regex=False)
                    ]
                except Exception as filter_err:
                    logger.error(f"Filter err col '{col}': {filter_err}")
                    return jsonify({"error": f"Filter failed for '{col}'."}), 500
                if df_filtered.empty:
                    break

        # Multi-search
        if not df_filtered.empty and MULTI_SEARCH_PARAM_NAME and MULTI_SEARCH_PARAM_NAME in request.args:
            multi_query = request.args.get(MULTI_SEARCH_PARAM_NAME, "").strip()
            if multi_query and MULTI_SEARCH_COLS:
                keywords = [kw.strip() for kw in multi_query.lower().split() if kw.strip()]
                if keywords:
                    query_params_used[MULTI_SEARCH_PARAM_NAME] = multi_query
                    final_mask = pd.Series(True, index=df_filtered.index)
                    for kw in keywords:
                        keyword_mask = pd.Series(False, index=df_filtered.index)
                        for col in MULTI_SEARCH_COLS:
                            if col in df_filtered.columns:
                                try:
                                    keyword_mask |= df_filtered[col].astype(str).str.contains(kw, case=False, na=False, regex=False)
                                except Exception as multi_err:
                                    logger.warning(f"Multi err col '{col}': {multi_err}", exc_info=False)
                        final_mask &= keyword_mask
                        if not final_mask.any():
                            break
                    df_filtered = df_filtered[final_mask]

        # Sorteren
        sort_by = request.args.get("_sort")
        order = request.args.get("_order", "asc").lower()
        ascending = order == "asc"
        if sort_by and sort_by in df_filtered.columns:
            try:
                query_params_used["_sort"] = sort_by
                query_params_used["_order"] = order
                df_filtered = df_filtered.sort_values(
                    by=sort_by,
                    ascending=ascending,
                    key=pd.to_numeric if COL_TYPES_CACHE.get(sort_by) in ["number", "integer"] else None,
                    na_position="last"
                )
            except Exception as sort_err:
                logger.warning(f"Sort err '{sort_by}': {sort_err}")
                query_params_used.pop("_sort", None)
                query_params_used.pop("_order", None)

        # Paginatie
        limit = parse_int_param(request.args.get("limit"), default=50, min_val=1, max_val=1000)
        offset = parse_int_param(request.args.get("offset"), default=0, min_val=0, max_val=float("inf"))
        query_params_used["limit"] = limit
        query_params_used["offset"] = offset

        total_results_found = len(df_filtered)
        df_paginated = df_filtered.iloc[offset : offset + limit]
        results_returned = len(df_paginated)

        results_list = df_paginated.reindex(columns=original_columns).where(pd.notna(df_paginated), None).to_dict(orient="records")

        return jsonify({
            "status": "success",
            "query_parameters_used": query_params_used,
            "pagination": {
                "offset": offset,
                "limit": limit,
                "results_in_this_page": results_returned,
                "total_results_found": total_results_found
            },
            "results": results_list
        })
    except Exception as api_err:
        logger.error(f"API err /api/data: {api_err}", exc_info=True)
        return jsonify({"error": "Internal server error."}), 500

# --- Shortcut-route: /data (zelfde als /api/data) ---
@app.route("/data", methods=["GET"])
def data_shortcut():
    return api_data_endpoint(dataset_slug=None)

# --- Route: /schema ---
@app.route("/schema", methods=["GET"])
def schema():
    global SELECTED_COLS
    if DATAFRAME_STORAGE is None:
        return jsonify({"error": "No data currently loaded."}), 400
    return jsonify({"columns": SELECTED_COLS})

# --- Route: Start Ngrok (GET en POST) ---
@app.route("/start_ngrok", methods=["GET", "POST"])
def start_ngrok():
    global NGROK_PROCESS, NGROK_URL

    if request.method == "POST":
        auth_token = request.form.get("auth_token", "").strip()
        if not auth_token:
            session["ngrok_error"] = "Ngrok Authtoken cannot be empty."
            return redirect(url_for("start_ngrok"))

        ngrok_cmd = "ngrok.exe" if os.name == "nt" else "ngrok"

        # 1) Configureer het token
        try:
            logger.info("Configuring Ngrok authtoken...")
            result = subprocess.run(
                [ngrok_cmd, "config", "add-authtoken", auth_token],
                check=True,
                capture_output=True,
                text=True,
                timeout=15
            )
            logger.info(f"Ngrok authtoken config output: {result.stdout}")
        except FileNotFoundError:
            session["ngrok_error"] = "'ngrok' command not found. Check PATH."
            return redirect(url_for("start_ngrok"))
        except subprocess.CalledProcessError as e:
            error_message = f"Failed to set Ngrok authtoken: {e.stderr or e.stdout or e}"
            logger.error(error_message)
            session["ngrok_error"] = error_message
            return redirect(url_for("start_ngrok"))
        except subprocess.TimeoutExpired:
            session["ngrok_error"] = "Ngrok command timed out setting token."
            return redirect(url_for("start_ngrok"))
        except Exception as e:
            logger.error(f"Unexpected error configuring Ngrok token: {e}", exc_info=True)
            session["ngrok_error"] = f"Unexpected token error: {e}"
            return redirect(url_for("start_ngrok"))

        # 2) Indien al een tunnel draaide, stop die eerst netjes
        if NGROK_PROCESS and NGROK_PROCESS.poll() is None:
            try:
                logger.info("Terminating previous Ngrok process...")
                NGROK_PROCESS.terminate()
                NGROK_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Ngrok did not terminate gracefully, killing.")
                NGROK_PROCESS.kill()
            except Exception as e:
                logger.error(f"Error terminating previous Ngrok process: {e}")
            finally:
                NGROK_PROCESS = None
                NGROK_URL = None

        NGROK_URL = None
        time.sleep(0.5)  # even kort pauzeren zodat de poort 4040 herstart kan afhandelen

        # 3) Start de nieuwe ngrok-tunnel
        try:
            logger.info("Starting Ngrok HTTP tunnel on port 5000...")
            startupinfo = None
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            NGROK_PROCESS = subprocess.Popen(
                [ngrok_cmd, "http", "5000", "--log=stdout"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                startupinfo=startupinfo
            )
            logger.info(f"Started Ngrok process (PID: {NGROK_PROCESS.pid})")
            time.sleep(4)  # wachttijd om de tunnel op te zetten

            if NGROK_PROCESS.poll() is not None:
                # Ngrok is plots gestopt in plaats van succesvol te runnen
                stderr_output = NGROK_PROCESS.stderr.read()
                stdout_output = NGROK_PROCESS.stdout.read()
                error_msg = f"Ngrok terminated unexpectedly. Stderr:{stderr_output or 'N/A'} Stdout:{stdout_output or 'N/A'}"
                logger.error(error_msg)
                session["ngrok_error"] = "Ngrok failed. Check logs."
                NGROK_PROCESS = None
                return redirect(url_for("start_ngrok"))

            # 4) Haal de publieke URL op via de lokale API van ngrok
            try:
                logger.info("Fetching tunnel URL from Ngrok API...")
                resp = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=10)
                resp.raise_for_status()
                tunnels = resp.json().get("tunnels", [])
                logger.debug(f"Ngrok tunnels response: {tunnels}")
                https_tunnel = next(
                    (tunnel for tunnel in tunnels
                     if tunnel.get("proto") == "https"
                     and "localhost:5000" in tunnel.get("config", {}).get("addr", "")),
                    None
                )
                if https_tunnel and https_tunnel.get("public_url"):
                    NGROK_URL = https_tunnel["public_url"]
                    logger.info(f"Ngrok tunnel URL: {NGROK_URL}")
                    session.pop("ngrok_error", None)

                    # 5) Redirect naar de correcte “generate-pagina” op basis van bestandstype
                    file_type = session.get("file_type")
                    if file_type == "pdf":
                        return redirect(url_for("generate_pdf"))
                    elif file_type in ["csv", "json"]:
                        # zowel CSV als JSON leiden naar dezelfde generate_api-pagina
                        return redirect(url_for("generate_api"))
                    else:
                        logger.warning(f"Ngrok gestart, maar onbekend file_type '{file_type}'. Redirecting to index.")
                        return redirect(url_for("index"))
                else:
                    logger.error(f"HTTPS tunnel to localhost:5000 not found in {tunnels}")
                    session["ngrok_error"] = "Could not find correct Ngrok tunnel."
                    if NGROK_PROCESS and NGROK_PROCESS.poll() is None:
                        NGROK_PROCESS.terminate()
                    NGROK_PROCESS = None
                    NGROK_URL = None
                    return redirect(url_for("start_ngrok"))
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Failed to connect to Ngrok API: {req_err}")
                session["ngrok_error"] = "Failed get tunnel URL from Ngrok API."
                if NGROK_PROCESS and NGROK_PROCESS.poll() is None:
                    NGROK_PROCESS.terminate()
                NGROK_PROCESS = None
                NGROK_URL = None
                return redirect(url_for("start_ngrok"))

        except Exception as launch_e:
            logger.error(f"Error launching/managing Ngrok: {launch_e}", exc_info=True)
            session["ngrok_error"] = f"Ngrok start error: {launch_e}"
            if NGROK_PROCESS and NGROK_PROCESS.poll() is None:
                try:
                    NGROK_PROCESS.terminate()
                except Exception:
                    pass
            NGROK_PROCESS = None
            NGROK_URL = None
            return redirect(url_for("start_ngrok"))

    # Bij een GET-request of als er een fout in de POST optrad, render de setup-pagina
    ngrok_error = session.pop("ngrok_error", None)
    return render_template("start_ngrok.html", ngrok_error=ngrok_error)

# --- Route: Generate PDF Page ---
@app.route("/generate_pdf")
def generate_pdf():
    global NGROK_URL
    filename = session.get("uploaded_filename")
    file_type = session.get("file_type")
    if not filename or file_type != "pdf":
        return redirect(url_for("index"))
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(pdf_path):
        logger.error(f"PDF '{filename}' not found.")
        return redirect(url_for("index"))
    try:
        local_base = request.host_url.rstrip("/")
        pdf_rel = url_for("uploaded_file", filename=filename)
        local_url = f"{local_base}{pdf_rel}"
        public_url = f"{NGROK_URL}{pdf_rel}" if NGROK_URL else None
        logger.info(f"PDF URLs: Local={local_url}, Public={public_url}")
    except Exception as url_e:
        logger.error(f"PDF URL err: {url_e}")
        local_url = "#err"
        public_url = "#err" if NGROK_URL else None
    return render_template(
        "generate_pdf.html",
        local_pdf_url=local_url,
        public_pdf_url=public_url,
        filename=filename,
        ngrok_running=bool(NGROK_URL),
    )

# --- Route: Serve Uploaded Files ---
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        logger.warning(f"Unsafe path: {filename}")
        return "Not Found", 404
    logger.debug(f"Serving file: {safe_filename}")
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], safe_filename, as_attachment=False)
    except FileNotFoundError:
        logger.error(f"File not found: {safe_filename}")
        return "File not found", 404

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(debug=True, port=5000, host="0.0.0.0", use_reloader=True)
