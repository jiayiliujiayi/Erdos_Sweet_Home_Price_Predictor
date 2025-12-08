# src/query.py

from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import time
import pandas as pd
from homeharvest import scrape_property


def generate_date_chunks(
    start_date: str,
    end_date: str,
    chunk_days: int = 90,
) -> List[Tuple[str, str]]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    chunks = []
    current_start = start

    while current_start <= end:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end)
        chunks.append((current_start.isoformat(), current_end.isoformat()))
        current_start = current_end + timedelta(days=1)

    return chunks


def scrape_sales(
    location: str,
    listing_type: str,
    date_chunks: List[Tuple[str, str]],
    property_types: Optional[list] = None,
    foreclosure: bool = False,
    mls_only: bool = True,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> Tuple[pd.DataFrame, list]:
    """
    Scrape Redfin property data across multiple date chunks, with basic
    retry and error handling.

    Parameters
    ----------
    location : str
        Location string for Redfin / homeharvest (e.g. 'New Jersey').
    listing_type : str
        'sold', 'for_sale', 'for_rent', or 'pending'.
    date_chunks : list of (start, end) tuples
        Output from generate_date_chunks.
    property_types : list of str or None
        e.g., ['single_family', 'multi_family']. If None, all types.
    foreclosure : bool
        Include foreclosure listings.
    mls_only : bool
        If True, only fetch MLS listings.
    max_retries : int
        Maximum number of attempts per chunk before giving up.
    retry_sleep_seconds : float
        Seconds to sleep between retries.

    Returns
    -------
    (pd.DataFrame, list)
        - Concatenated DataFrame of all successfully scraped chunks.
        - List of failed chunks as tuples:
          (start_date, end_date, error_message).
    """
    all_dfs = []
    failed_chunks = []

    for i, (start_str, end_str) in enumerate(date_chunks, start=1):
        print(f"[{i}/{len(date_chunks)}] Scraping {location} from {start_str} to {end_str}...")

        df_chunk = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                df_chunk = scrape_property(
                    location=location,
                    listing_type=listing_type,
                    date_from=start_str,
                    date_to=end_str,
                    property_type=property_types,
                    foreclosure=foreclosure,
                    mls_only=mls_only,
                )
                break  # success, exit retry loop

            except Exception as e:
                last_error = e
                print(f"  Attempt {attempt}/{max_retries} failed with error: {repr(e)}")
                if attempt < max_retries:
                    time.sleep(retry_sleep_seconds)
                else:
                    print("  Giving up on this chunk.")
        
        if df_chunk is None:
            # All attempts failed for this chunk
            failed_chunks.append((start_str, end_str, repr(last_error)))
            continue

        print(f"  Retrieved {len(df_chunk)} properties")
        if len(df_chunk) > 0:
            all_dfs.append(df_chunk)

    if not all_dfs:
        print("No data returned for any chunk.")
        return pd.DataFrame(), failed_chunks

    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all, failed_chunks


def deduplicate_properties(df: pd.DataFrame) -> pd.DataFrame:
    id_candidates = ["property_id", "listing_id", "mls_id"]
    dedup_cols = [col for col in id_candidates if col in df.columns]

    if dedup_cols:
        print(f"Using {dedup_cols} for de-duplication.")
        return df.drop_duplicates(subset=dedup_cols)

    fallback_cols = [c for c in ["address", "city", "zip", "sold_date"] if c in df.columns]
    if fallback_cols:
        print(f"No unique ID found; using {fallback_cols} for de-duplication.")
        return df.drop_duplicates(subset=fallback_cols)

    print("No suitable columns found for de-duplication; returning original DataFrame.")
    return df
