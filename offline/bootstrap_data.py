"""Create offline seed files if missing (CSV, JSON)."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data"


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)


def write_scheme_index() -> None:
    path = ROOT / "scheme_index.json"
    if path.exists():
        return
    ensure_dirs()
    schemes = [
        {
            "id": "pm_kisan",
            "name": "PM-KISAN",
            "eligibility": "Small and marginal farmers with cultivable land; Aadhaar linked bank account.",
            "benefits": "Rs 6000/year in 3 installments",
            "how_to_apply": "Visit nearest CSC or PM-KISAN portal; Aadhaar eKYC.",
            "keywords": ["pm-kisan", "6000", "farmer income", "installment"],
        },
        {
            "id": "pmfb",
            "name": "Pradhan Mantri Fasal Bima Yojana",
            "eligibility": "Loanee and non-loanee farmers growing notified crops in notified areas.",
            "benefits": "Affordable crop insurance against natural calamities.",
            "how_to_apply": "Through banks, CSC, or insurer portal within cutoff dates.",
            "keywords": ["insurance", "fasal bima", "crop loss"],
        },
        {
            "id": "kcc",
            "name": "Kisan Credit Card",
            "eligibility": "Farmers engaged in agriculture / allied activities.",
            "benefits": "Short-term credit for cultivation and post-harvest needs.",
            "how_to_apply": "Approach scheduled commercial banks / cooperative banks with land records.",
            "keywords": ["loan", "kcc", "credit"],
        },
    ]
    extra_names = [
        ("PMKSY", "Per Drop More Crop — drip/sprinkler subsidy", ["irrigation", "water use efficiency"]),
        ("Soil Health Card", "Soil testing and recommendations", ["soil", "testing", "fertilizer"]),
        ("eNAM", "National Agriculture Market online trading", ["mandi", "price", "online"]),
        ("NRLM", "Rural livelihoods for women SHGs", ["women", "self help"]),
        ("MIDH", "Mission for Integrated Development of Horticulture", ["horticulture", "orchard"]),
        ("NMSA", "National Mission on Sustainable Agriculture", ["organic", "sustainable"]),
        ("PMFBY_RICE", "PMFBY coverage for rice in notified districts", ["rice", "insurance"]),
        ("AGRI_INFRA_FUND", "Financing facility for post-harvest infrastructure", ["storage", "cold chain"]),
        ("FPO", "Formation & Promotion of Farmer Producer Organizations", ["collective", "market"]),
        ("NAMPHASE2", "Unified national market reforms", ["market", "reform"]),
    ]
    for i, (name, desc, kw) in enumerate(extra_names):
        schemes.append(
            {
                "id": f"scheme_{i}",
                "name": name,
                "eligibility": "Varies by state notification; check state agriculture portal.",
                "benefits": desc,
                "how_to_apply": "Contact district agriculture office or CSC.",
                "keywords": list(kw),
            }
        )
    path.write_text(json.dumps(schemes, ensure_ascii=False, indent=2), encoding="utf-8")


def write_crop_calendar() -> None:
    path = ROOT / "crop_calendar.json"
    if path.exists():
        return
    ensure_dirs()
    crops = {
        "wheat": {
            "soils": ["loamy", "clay"],
            "seasons": ["rabi"],
            "water_need": 2,
            "state_bonus": {"Punjab": 1.15, "Uttar Pradesh": 1.1},
        },
        "rice": {
            "soils": ["clay", "loamy"],
            "seasons": ["kharif"],
            "water_need": 3,
            "state_bonus": {"Punjab": 1.0, "Andhra Pradesh": 1.2},
        },
        "cotton": {
            "soils": ["black", "clay"],
            "seasons": ["kharif"],
            "water_need": 2,
            "state_bonus": {"Maharashtra": 1.15},
        },
        "mustard": {
            "soils": ["loamy", "sandy"],
            "seasons": ["rabi"],
            "water_need": 2,
            "state_bonus": {"Punjab": 1.1},
        },
        "maize": {
            "soils": ["loamy"],
            "seasons": ["kharif", "rabi"],
            "water_need": 2,
            "state_bonus": {"Uttar Pradesh": 1.05},
        },
    }
    path.write_text(json.dumps({"crops": crops}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_mandi_csv(rows: int = 500) -> None:
    path = ROOT / "mandi_prices.csv"
    if path.exists():
        return
    ensure_dirs()
    random.seed(42)
    crops = ["wheat", "rice", "maize", "cotton", "soybean", "onion", "potato", "tomato"]
    districts = [
        ("Ludhiana", "Punjab"),
        ("Karnal", "Haryana"),
        ("Nagpur", "Maharashtra"),
        ("Guntur", "Andhra Pradesh"),
        ("Coimbatore", "Tamil Nadu"),
    ]
    mandis = ["Main Yard", "Sub Yard A", "eNAM Hub"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["crop", "mandi", "district", "state", "date", "price_inr", "unit"],
        )
        w.writeheader()
        for i in range(rows):
            c = random.choice(crops)
            d, st = random.choice(districts)
            price = round(1500 + random.random() * 2500, 2)
            w.writerow(
                {
                    "crop": c,
                    "mandi": random.choice(mandis),
                    "district": d,
                    "state": st,
                    "date": f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                    "price_inr": price,
                    "unit": "quintal",
                }
            )


def write_weather_history_json() -> None:
    path = ROOT / "weather_history.json"
    if path.exists():
        return
    ensure_dirs()
    districts = [
        "Ludhiana",
        "Karnal",
        "Nagpur",
        "Guntur",
        "Coimbatore",
        "Pune",
        "Lucknow",
        "Patna",
    ]
    rows = []
    for d in districts:
        for m in range(1, 13):
            rows.append(
                {
                    "district": d,
                    "month": m,
                    "avg_temp_c": 18 + (m % 6) * 2.5,
                    "avg_rain_mm": 20 + (m * 7) % 120,
                }
            )
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def bootstrap_all() -> None:
    ensure_dirs()
    write_scheme_index()
    write_crop_calendar()
    write_mandi_csv(500)
    write_weather_history_json()


if __name__ == "__main__":
    bootstrap_all()
