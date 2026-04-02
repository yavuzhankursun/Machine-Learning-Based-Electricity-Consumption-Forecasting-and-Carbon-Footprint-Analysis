"""
Climate temperature data collection script.
Approach 1: World Bank Climate API (CRU TAS)
Approach 2: OWID co2-data (temperature_change_from_co2 column)
Approach 3: World Bank Development Indicators (climate-related)
Uses only stdlib (urllib, json, csv).
"""

import urllib.request
import json
import csv
import os
import time
import sys
import io

OUTPUT_DIR = r"C:\Users\LENOVO\Desktop\Tez Yeni\data\collected"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "climate_temperature_data.csv")
FALLBACK_FILE = os.path.join(OUTPUT_DIR, "climate_related_indicators.csv")

COUNTRIES = {
    "TUR": "Turkey", "USA": "United States", "GBR": "United Kingdom",
    "DEU": "Germany", "FRA": "France", "ITA": "Italy", "ESP": "Spain",
    "RUS": "Russia", "CHN": "China", "JPN": "Japan", "IND": "India",
    "BRA": "Brazil", "MEX": "Mexico", "CAN": "Canada", "AUS": "Australia",
    "ARG": "Argentina", "ZAF": "South Africa", "EGY": "Egypt",
    "NGA": "Nigeria", "KEN": "Kenya", "SAU": "Saudi Arabia",
    "IRN": "Iran", "IDN": "Indonesia", "KOR": "South Korea",
    "THA": "Thailand", "VNM": "Vietnam", "MYS": "Malaysia",
    "PHL": "Philippines", "PAK": "Pakistan", "BGD": "Bangladesh",
    "POL": "Poland", "UKR": "Ukraine", "NLD": "Netherlands",
    "BEL": "Belgium", "SWE": "Sweden", "NOR": "Norway",
    "FIN": "Finland", "DNK": "Denmark", "CHE": "Switzerland",
    "AUT": "Austria", "GRC": "Greece", "PRT": "Portugal",
    "CZE": "Czech Republic", "ROU": "Romania", "HUN": "Hungary",
    "COL": "Colombia", "CHL": "Chile", "PER": "Peru",
    "VEN": "Venezuela", "IRQ": "Iraq", "MAR": "Morocco",
    "TUN": "Tunisia", "GHA": "Ghana", "ETH": "Ethiopia",
    "TZA": "Tanzania", "NZL": "New Zealand", "ISR": "Israel",
    "JOR": "Jordan", "LBN": "Lebanon", "ARE": "United Arab Emirates",
    "QAT": "Qatar", "KWT": "Kuwait", "SGP": "Singapore",
}

YEAR_MIN = 2000
YEAR_MAX = 2023


def fetch_url(url, timeout=30):
    """URL'den veri cek, JSON veya text olarak dondur"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def approach1():
    """World Bank Climate Data API"""
    print("=" * 60)
    print("APPROACH 1: World Bank Climate API (CRU TAS)")
    print("=" * 60)

    # Test
    test_url = "http://climatedataapi.worldbank.org/climateweb/rest/v1/country/cru/tas/year/TUR"
    try:
        raw = fetch_url(test_url, timeout=15)
        test_data = json.loads(raw)
        if not test_data:
            print("  Test failed: empty data")
            return None
        print(f"  Test OK! Sample: {test_data[0]}")
    except Exception as e:
        print(f"  API unreachable: {e}")
        return None

    all_data = []
    ok = 0
    fail = 0

    for iso3, name in COUNTRIES.items():
        url = f"http://climatedataapi.worldbank.org/climateweb/rest/v1/country/cru/tas/year/{iso3}"
        try:
            raw = fetch_url(url, timeout=15)
            data = json.loads(raw)
            count = 0
            if data:
                for entry in data:
                    yr = entry.get("year")
                    tmp = entry.get("data")
                    if yr and tmp and YEAR_MIN <= yr <= YEAR_MAX:
                        all_data.append({
                            "Country": name,
                            "Country_Code": iso3,
                            "Year": int(yr),
                            "Avg_Temperature": round(float(tmp), 2),
                        })
                        count += 1
            ok += 1
            print(f"  [OK] {name} ({iso3}): {count} years")
            time.sleep(0.3)
        except Exception as e:
            fail += 1
            print(f"  [FAIL] {name} ({iso3}): {e}")

    print(f"\n  Result: {ok} ok, {fail} failed, {len(all_data)} records")
    if all_data:
        save_temperature_csv(all_data,
            "# Kaynak: World Bank Climate Data API - CRU TS (http://climatedataapi.worldbank.org)")
        return True
    return None


def approach2():
    """OWID co2-data.csv -- contains temperature_change_from_co2 per country"""
    print("\n" + "=" * 60)
    print("APPROACH 2: Our World in Data (co2-data)")
    print("=" * 60)

    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    print(f"  Fetching: {url}")

    try:
        raw = fetch_url(url, timeout=60)
    except Exception as e:
        print(f"  Download failed: {e}")
        return None

    reader = csv.DictReader(io.StringIO(raw))
    fields = reader.fieldnames
    print(f"  Columns (first 20): {fields[:20]}")

    # Find temperature-related columns
    temp_cols = [c for c in fields if "temp" in c.lower()]
    print(f"  Temperature columns: {temp_cols}")

    if not temp_cols:
        print("  No temperature column found")
        return None

    temp_col = temp_cols[0]
    all_data = []

    for row in reader:
        try:
            year = int(row.get("year", 0))
            country = row.get("country", "")
            iso = row.get("iso_code", "")
            val = row.get(temp_col, "")

            if (YEAR_MIN <= year <= YEAR_MAX and val and val.strip()
                    and iso and len(iso) == 3):
                all_data.append({
                    "Country": country,
                    "Country_Code": iso,
                    "Year": year,
                    "Avg_Temperature": round(float(val), 4),
                })
        except (ValueError, TypeError):
            continue

    if all_data:
        unique = len(set(r["Country_Code"] for r in all_data))
        print(f"  Found {len(all_data)} records, {unique} countries (col: {temp_col})")
        # NOTE: this is temperature *change* (anomaly), not absolute temperature
        save_temperature_csv(all_data,
            f"# Kaynak: Our World in Data CO2 Dataset - {temp_col} (https://github.com/owid/co2-data)")
        return True

    print("  No valid data extracted")
    return None


def approach3():
    """World Bank Development Indicators -- climate-related"""
    print("\n" + "=" * 60)
    print("APPROACH 3: World Bank Indicators (climate-related)")
    print("=" * 60)

    indicators = {
        "EN.ATM.CO2E.KT": "CO2_Emissions_kt",
        "EN.ATM.CO2E.PC": "CO2_Per_Capita",
        "AG.LND.AGRI.ZS": "Agricultural_Land_Pct",
        "EG.USE.PCAP.KG.OE": "Energy_Use_Per_Capita",
        "EN.ATM.METH.KT.CE": "Methane_Emissions_kt",
        "EN.ATM.NOXE.KT.CE": "Nitrous_Oxide_kt",
    }

    all_rows = {}

    for code, col_name in indicators.items():
        print(f"\n  Fetching: {col_name} ({code})")
        page = 1
        total = 0

        while True:
            url = (f"https://api.worldbank.org/v2/country/all/indicator/{code}"
                   f"?date={YEAR_MIN}:{YEAR_MAX}&format=json&per_page=1000&page={page}")
            try:
                raw = fetch_url(url, timeout=30)
                data = json.loads(raw)

                if not isinstance(data, list) or len(data) < 2 or not data[1]:
                    break

                meta = data[0]
                for rec in data[1]:
                    val = rec.get("value")
                    if val is None:
                        continue
                    c_name = rec.get("country", {}).get("value", "")
                    c_code = rec.get("countryiso3code", "")
                    yr = rec.get("date", "")
                    if not c_code or len(c_code) != 3 or not yr:
                        continue
                    try:
                        yr = int(yr)
                    except ValueError:
                        continue

                    key = (c_name, c_code, yr)
                    if key not in all_rows:
                        all_rows[key] = {}
                    all_rows[key][col_name] = round(float(val), 4)
                    total += 1

                if page >= meta.get("pages", 1):
                    break
                page += 1
                time.sleep(0.3)
            except Exception as e:
                print(f"    Error: {e}")
                break

        print(f"    {total} records fetched")
        time.sleep(0.5)

    if not all_rows:
        return None

    ind_names = list(indicators.values())
    csv_rows = []
    for (c_name, c_code, yr), vals in sorted(all_rows.items(), key=lambda x: (x[0][0], x[0][2])):
        row = {"Country": c_name, "Country_Code": c_code, "Year": yr}
        for n in ind_names:
            row[n] = vals.get(n, "")
        csv_rows.append(row)

    fieldnames = ["Country", "Country_Code", "Year"] + ind_names
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(FALLBACK_FILE, "w", newline="", encoding="utf-8") as f:
        f.write("# Kaynak: World Bank Development Indicators API (https://api.worldbank.org/v2/)\n")
        f.write(f"# Gostergeler: {', '.join(f'{k}={v}' for k,v in indicators.items())}\n")
        f.write(f"# Yil araligi: {YEAR_MIN}-{YEAR_MAX}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    unique = len(set(r["Country_Code"] for r in csv_rows))
    print(f"\n  SUCCESS! {len(csv_rows)} records, {unique} countries")
    print(f"  File: {FALLBACK_FILE}")
    return True


def save_temperature_csv(data, metadata):
    data.sort(key=lambda x: (x["Country"], x["Year"]))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        f.write(metadata + "\n")
        writer = csv.DictWriter(f, fieldnames=["Country", "Country_Code", "Year", "Avg_Temperature"])
        writer.writeheader()
        writer.writerows(data)

    unique = len(set(r["Country_Code"] for r in data))
    print(f"\n  SUCCESS! {len(data)} records, {unique} countries")
    print(f"  File: {OUTPUT_FILE}")


def main():
    print("Climate Temperature Data Collection Script")
    print(f"Year range: {YEAR_MIN}-{YEAR_MAX}")
    print(f"Target countries: {len(COUNTRIES)}\n")

    # Approach 1
    if approach1():
        print("\n>>> APPROACH 1 SUCCEEDED <<<")
        return

    # Approach 2
    if approach2():
        print("\n>>> APPROACH 2 SUCCEEDED <<<")
        return

    # Approach 3
    if approach3():
        print("\n>>> APPROACH 3 SUCCEEDED (climate-related indicators) <<<")
        return

    print("\n>>> ALL APPROACHES FAILED <<<")
    sys.exit(1)


if __name__ == "__main__":
    main()
