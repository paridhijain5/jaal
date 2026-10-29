"""Generate deterministic synthetic demo data with planted risk patterns."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "sample"


def generate_sample_data(output_dir: Path | str = ROOT) -> None:
    rng = np.random.default_rng(8142)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    symbols = [f"JAAL{i:02d}" for i in range(1, 13)]
    companies = ["Aster Components", "Bharat Circuits", "Cobalt Textiles", "Dakshin Foods", "Eastern Alloys", "Finpeak Systems", "Greenline Pharma", "Harbor Cables", "Indigo Packaging", "Juniper Motors", "Kaveri Energy", "Lotus Bearings"]
    days = pd.bdate_range("2024-01-02", "2025-12-30")
    rings = {"JAAL02": ["Meridian Advisors", "R. Mehta"], "JAAL05": ["Meridian Advisors", "R. Mehta"],
             "JAAL08": ["Meridian Advisors", "R. Mehta"], "JAAL10": ["Bluecrest Holdings", "S. Kapoor"],
             "JAAL11": ["Bluecrest Holdings", "S. Kapoor"]}
    anomaly_days = {symbol: [days[145], days[245], days[425]] for symbol in ["JAAL02", "JAAL05", "JAAL08"]}
    anomaly_days.update({symbol: [days[180], days[460]] for symbol in ["JAAL10", "JAAL11"]})
    price_rows = []
    for index, (symbol, company) in enumerate(zip(symbols, companies)):
        close = 72 + 9 * index
        for day in days:
            event = day in anomaly_days.get(symbol, [])
            ret = rng.normal(0.00025, 0.016)
            if event:
                ret = float(rng.choice([0.065, 0.085, -0.052]))
            opening = close * (1 + rng.normal(0, 0.006))
            close = max(8, close * (1 + ret))
            base_volume = int(rng.lognormal(11.0, 0.38))
            volume = int(base_volume * rng.uniform(4.5, 7.0)) if event else base_volume
            delivery = float(np.clip(rng.normal(36, 8) + (28 if event else 0), 8, 94))
            spread = abs(rng.normal(0.012, 0.006)) + (0.035 if event else 0)
            price_rows.append({"date": day, "symbol": symbol, "company": company,
                               "open": round(opening, 2), "high": round(max(opening, close) * (1 + spread), 2),
                               "low": round(min(opening, close) * (1 - spread), 2), "close": round(close, 2),
                               "volume": volume, "delivery_pct": round(delivery, 2)})
    prices = pd.DataFrame(price_rows)

    entity_rows = []
    base_entities = [f"Local Director {i:02d}" for i in range(20)]
    for i, symbol in enumerate(symbols):
        entity_rows.append({"date": days[0], "symbol": symbol, "entity": base_entities[i % len(base_entities)],
                            "entity_type": "Director", "relationship": "director", "source": "shareholding"})
    for symbol, linked in rings.items():
        for entity in linked:
            entity_rows.append({"date": days[35], "symbol": symbol, "entity": entity,
                                "entity_type": "Common director" if "." in entity else "Connected entity",
                                "relationship": "director" if "." in entity else "bulk_deal_counterparty",
                                "source": "bulk_block_deals"})
    entities = pd.DataFrame(entity_rows)

    filing_types = ["auditor_resignation", "pledge_increase", "delayed_results", "preferential_allotment", "insider_sale", "bulk_deal", "corporate_announcement"]
    filing_rows = [{"date": days[65], "symbol": symbol, "filing_type": "corporate_announcement",
                    "description": "Quarterly business update", "severity": 0.05, "entity": "Company"}
                   for symbol in symbols]
    for symbol in ["JAAL02", "JAAL05", "JAAL08"]:
        for offset, filing_type in zip([139, 141, 143], ["pledge_increase", "preferential_allotment", "insider_sale"]):
            filing_rows.append({"date": days[offset], "symbol": symbol, "filing_type": filing_type,
                                "description": f"Synthetic {filing_type.replace('_', ' ')} disclosure", "severity": 0.2,
                                "entity": rings[symbol][0]})
    for symbol in ["JAAL10", "JAAL11"]:
        filing_rows.append({"date": days[175], "symbol": symbol, "filing_type": "auditor_resignation",
                            "description": "Synthetic auditor resignation disclosure", "severity": 0.34,
                            "entity": rings[symbol][0]})
    for symbol in symbols:
        if symbol not in anomaly_days:
            for _ in range(3):
                filing_rows.append({"date": days[int(rng.integers(80, 400))], "symbol": symbol,
                                    "filing_type": str(rng.choice(filing_types)), "description": "Routine disclosure",
                                    "severity": 0.05, "entity": "Company"})
    filings = pd.DataFrame(filing_rows).sort_values("date")
    orders = pd.DataFrame([
        {"order_date": days[157], "symbol": "JAAL02", "case_id": "SEBI-SYN-0241", "order_type": "Market manipulation",
         "summary": "Coordinated trading and connected-account activity; synthetic demo case", "penalised": 1},
        {"order_date": days[257], "symbol": "JAAL05", "case_id": "SEBI-SYN-0318", "order_type": "Price manipulation",
         "summary": "Artificial price/volume build-up and misleading disclosures; synthetic demo case", "penalised": 1},
        {"order_date": days[192], "symbol": "JAAL10", "case_id": "SEBI-SYN-0280", "order_type": "Circular trading",
         "summary": "Repeated connected counterparties and abnormal delivery; synthetic demo case", "penalised": 1},
        {"order_date": days[442], "symbol": "JAAL08", "case_id": "SEBI-SYN-0425", "order_type": "Connected trading",
         "summary": "Synthetic holdout pattern involving repeat counterparties and abnormal turnover", "penalised": 1},
        {"order_date": days[469], "symbol": "JAAL11", "case_id": "SEBI-SYN-0460", "order_type": "Disclosure concerns",
         "summary": "Synthetic holdout pattern with an abnormal volume and price episode", "penalised": 1},
    ])
    prices.to_csv(out / "prices.csv", index=False, date_format="%Y-%m-%d")
    entities.to_csv(out / "entities.csv", index=False, date_format="%Y-%m-%d")
    filings.to_csv(out / "filings.csv", index=False, date_format="%Y-%m-%d")
    orders.to_csv(out / "sebi_orders.csv", index=False, date_format="%Y-%m-%d")


if __name__ == "__main__":
    generate_sample_data()