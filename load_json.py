import pandas as pd
import json
import logging
import os

logger = logging.getLogger(__name__)


def load_json(file_path):
    """
    Laadt een JSON-bestand en flattent alle velden (inclusief meta-data bovenaan).
    - Als de root zelf een lijst is van dicts: flatten elk element als rij.
    - Als de root een dict is waarin één van de waarden een lijst van dicts is:
        * Neem die lijst als “records”.
        * Flatten elk record en voeg daarnaast alle andere velden uit de root (meta) toe in elke rij.
    - Als er geen lijst van dicts te vinden is: flatten de dict in één enkele rij.
    Returns: pandas.DataFrame of None bij een fout.
    """
    if not os.path.exists(file_path):
        logger.error(f"JSON file not found: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        return None

    def _flatten_dict(dct):
        """
        Flatten een dict volledig (nested) naar één platte dict met underscores in de keys.
        – Gebruikt pd.json_normalize onder de motorkap om alle geneste dicts weg te werken.
        """
        try:
            # json_normalize geeft een DataFrame met één rij; we halen daarvan de eerste rij als dict
            return pd.json_normalize(dct, sep='_').iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Flattening dict failed: {e}")
            return {}

    # 1) Als de root zelf een lijst is van dicts → flatten elk element en return DataFrame
    if isinstance(raw, list) and all(isinstance(el, dict) for el in raw):
        try:
            # pd.json_normalize op een lijst dicts flattent alle nested dicts in die lijst
            return pd.json_normalize(raw, sep='_')
        except Exception as e:
            logger.error(f"Flattening root-list failed: {e}")
            return None

    # 2) Als root een dict is waarin ten minste één van de waarden een lijst van dicts is
    if isinstance(raw, dict):
        lijst_key = None
        max_len = 0
        # Zoek op root-niveau naar de grootste lijst van dicts
        for key, val in raw.items():
            if isinstance(val, list) and val and all(isinstance(el, dict) for el in val):
                if len(val) > max_len:
                    lijst_key = key
                    max_len = len(val)

        # Als we een dergelijke lijst van dicts vonden, gebruik die als “records”
        if lijst_key:
            records = raw[lijst_key]
            # “Meta” is alles in raw behalve die lijst
            meta_dict = {k: v for k, v in raw.items() if k != lijst_key}
            flat_meta = _flatten_dict(meta_dict)

            try:
                # Flatten alle records (elk element in de lijst is een dict)
                df_records = pd.json_normalize(records, sep='_')
            except Exception as e:
                logger.error(f"Flattening records failed: {e}")
                return None

            # Bouw een DataFrame van de metadata, herhaald voor elk record
            df_meta = pd.DataFrame([flat_meta] * len(df_records))
            # Concat horizontaal: eerst meta-columns, dan record-columns
            df_combined = pd.concat(
                [df_meta.reset_index(drop=True), df_records.reset_index(drop=True)],
                axis=1
            )
            return df_combined

        # 3) Als we geen lijst van dicts op root-niveau vonden: flatten raw zelf als één rij
        flat_root = _flatten_dict(raw)
        try:
            return pd.DataFrame([flat_root])
        except Exception as e:
            logger.error(f"Creating DataFrame from flattened root failed: {e}")
            return None

    # 4) Als raw niet dict of lijst van dicts is, geen bruikbare structuur
    logger.error('Geen geschikte JSON-structuur gevonden om te flattenen.')
    return None
