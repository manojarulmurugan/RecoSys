"""
Build demo/catalog.json from BigQuery item_features.
Run once: python scripts/build_catalog.py
Outputs a compact JSON lookup: { item_id: {c1, c2, brand, price} }
"""
import json
import os
from collections import Counter
from google.cloud import bigquery

PROJECT = "recosys-489001"

# Items used in the demo preset sessions — always include these
PRESET_ITEMS = {"1000544", "1000894", "1000978", "1001588", "1001605"}

QUERY_TOP = """
SELECT
  CAST(product_id AS STRING)           AS item_id,
  COALESCE(cat_l1, 'unknown')          AS c1,
  COALESCE(cat_l2, 'unknown')          AS c2,
  TRIM(LOWER(COALESCE(brand, '')))     AS brand,
  ROUND(CAST(avg_price AS FLOAT64), 2) AS price,
  CAST(item_total_confidence AS INT64) AS interactions
FROM `recosys-489001.recosys.item_features`
ORDER BY interactions DESC
LIMIT 2000
"""

QUERY_PRESETS = """
SELECT
  CAST(product_id AS STRING)           AS item_id,
  COALESCE(cat_l1, 'unknown')          AS c1,
  COALESCE(cat_l2, 'unknown')          AS c2,
  TRIM(LOWER(COALESCE(brand, '')))     AS brand,
  ROUND(CAST(avg_price AS FLOAT64), 2) AS price,
  CAST(item_total_confidence AS INT64) AS interactions
FROM `recosys-489001.recosys.item_features`
WHERE CAST(product_id AS STRING) IN ({placeholders})
"""

# Abbreviations that should not be title-cased
_SPECIAL = {
    "lg": "LG", "hp": "HP", "aeg": "AEG", "tcl": "TCL",
    "bmw": "BMW", "kia": "KIA", "cpu": "CPU", "ssd": "SSD",
    "led": "LED", "rgb": "RGB", "usb": "USB", "hdmi": "HDMI",
    "tv": "TV", "jbl": "JBL", "msn": "MSN",
}


def fmt_brand(raw: str) -> str:
    if not raw:
        return "Unknown"
    return " ".join(_SPECIAL.get(w, w.title()) for w in raw.split())


def main() -> None:
    client = bigquery.Client(project=PROJECT)

    print("Querying top 2000 items …")
    catalog: dict = {}
    for row in client.query(QUERY_TOP).result():
        catalog[row.item_id] = {
            "c1":    row.c1,
            "c2":    row.c2,
            "brand": fmt_brand(row.brand),
            "price": float(row.price) if row.price else 0.0,
        }
    print(f"  {len(catalog)} items loaded")

    missing = PRESET_ITEMS - set(catalog)
    if missing:
        print(f"Fetching {len(missing)} preset items separately …")
        ph = ", ".join(f"'{i}'" for i in missing)
        for row in client.query(QUERY_PRESETS.format(placeholders=ph)).result():
            catalog[row.item_id] = {
                "c1":    row.c1,
                "c2":    row.c2,
                "brand": fmt_brand(row.brand),
                "price": float(row.price) if row.price else 0.0,
            }

    out = os.path.join(os.path.dirname(__file__), "..", "demo", "catalog.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, separators=(",", ":"), ensure_ascii=False)

    print(f"\nWrote {len(catalog)} items to demo/catalog.json")
    cats = Counter(v["c1"] for v in catalog.values())
    print("\nCategory distribution:")
    for cat, n in cats.most_common():
        print(f"  {cat:20s} {n:4d} items")


if __name__ == "__main__":
    main()
