"""Public-source fetchers and offline sample-data loading."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"
SAMPLE_FILES = {
    "prices": "prices.csv",
    "entities": "entities.csv",
    "filings": "filings.csv",
    "orders": "sebi_orders.csv",
}


def load_sample_data(data_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the complete demo dataset from CSV without making network calls."""
    root = Path(data_dir)
    return {name: pd.read_csv(root / filename, parse_dates=["date"] if name != "orders" else ["order_date"])
            for name, filename in SAMPLE_FILES.items()}


def fetch_csv(url: str, *, timeout: int = 20) -> pd.DataFrame:
    """Fetch a public CSV endpoint with browser-like headers (e.g. NSE archives)."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 JAAL research prototype", "Accept": "text/csv,*/*"})
    with urlopen(request, timeout=timeout) as response:
        return pd.read_csv(BytesIO(response.read()))


def fetch_nse_bhavcopy(url: str) -> pd.DataFrame:
    """Fetch an NSE bhavcopy CSV/ZIP URL supplied by the operator.

    NSE changes archive paths and may require a browser session. This explicit
    URL-based adapter avoids silently relying on an unstable endpoint.
    """
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 JAAL research prototype", "Accept": "text/csv,application/zip,*/*"})
    with urlopen(request, timeout=30) as response:
        payload = response.read()
    if url.lower().endswith(".zip"):
        return pd.read_csv(BytesIO(payload), compression="zip")
    return pd.read_csv(BytesIO(payload))


def fetch_nse_api(path: str, *, timeout: int = 20) -> dict:
    """Fetch a public NSE API JSON endpoint, such as corporate announcements."""
    request = Request(
        path,
        headers={"User-Agent": "Mozilla/5.0 JAAL research prototype", "Accept": "application/json,text/plain,*/*",
                 "Referer": "https://www.nseindia.com/"},
    )
    with urlopen(request, timeout=timeout) as response:
        import json

        return json.loads(response.read().decode("utf-8"))


def fetch_bse_csv(url: str) -> pd.DataFrame:
    """Fetch a public BSE CSV archive URL (deal disclosures/announcements)."""
    return fetch_csv(url)


def fetch_nse_bulk_block_deals(url: str) -> pd.DataFrame:
    """Fetch an NSE bulk/block deal export URL as a raw table."""
    return fetch_csv(url)


def fetch_nse_shareholding(url: str) -> pd.DataFrame:
    """Fetch an NSE shareholding-pattern export URL as a raw table."""
    return fetch_csv(url)


def fetch_nse_insider_disclosures(url: str) -> pd.DataFrame:
    """Fetch an NSE insider-trading disclosure export URL as a raw table."""
    return fetch_csv(url)


def fetch_nse_announcements(url: str) -> dict:
    """Fetch the JSON response from an NSE corporate-announcements API URL."""
    return fetch_nse_api(url)


def fetch_sebi_orders(url: str) -> pd.DataFrame:
    """Fetch a machine-readable SEBI orders export for historical labels."""
    return fetch_csv(url)