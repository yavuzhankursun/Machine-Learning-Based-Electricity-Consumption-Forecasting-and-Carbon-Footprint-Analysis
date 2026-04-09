import urllib.request
import json
import csv
import os
import time

OUTPUT_PATH = r"C:/Users/LENOVO/Desktop/Tez Yeni/data/collected/world_bank_indicators.csv"

INDICATORS = {
    "SP.POP.TOTL": "Population",
    "EN.POP.DNST": "Population_Density",
    "SP.URB.TOTL.IN.ZS": "Urbanization_Rate",
    "NY.GDP.PCAP.CD": "GDP_Per_Capita",
    "NY.GDP.MKTP.KD.ZG": "GDP_Growth",
}

BASE_URL = "http://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000&date=2000:2023"


def fetch_indicator(indicator_code, column_name):
    """Fetch a single indicator from the World Bank API, handling pagination."""
    url = BASE_URL.format(indicator=indicator_code)
    print(f"Fetching {column_name} ({indicator_code}) ...")

    all_records = []
    page = 1

    while True:
        page_url = url + f"&page={page}"
        retries = 3
        for attempt in range(retries):
            try:
                req = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    raw = response.read().decode("utf-8")
                    data = json.loads(raw)
                break
            except Exception as e:
                print(f"  Attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(3)
                else:
                    print(f"  ERROR: Could not fetch {indicator_code} page {page}. Skipping.")
                    return all_records

        if not isinstance(data, list) or len(data) < 2 or data[1] is None:
            print(f"  No data returned for {indicator_code} page {page}.")
            break

        metadata = data[0]
        records = data[1]

        for record in records:
            country_name = record.get("country", {}).get("value", "")
            country_code = record.get("countryiso3code", "") or record.get("country", {}).get("id", "")
            year = record.get("date", "")
            value = record.get("value")

            all_records.append({
                "Country": country_name,
                "Country_Code": country_code,
                "Year": year,
                "indicator": column_name,
                "value": value,
            })

        total_pages = metadata.get("pages", 1)
        print(f"  Page {page}/{total_pages} - got {len(records)} records")
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.5)

    print(f"  Total records for {column_name}: {len(all_records)}")
    return all_records


def main():
    # Fetch all indicators
    all_data = {}  # key: (Country, Country_Code, Year) -> dict of indicator values

    for indicator_code, column_name in INDICATORS.items():
        records = fetch_indicator(indicator_code, column_name)
        for rec in records:
            key = (rec["Country"], rec["Country_Code"], rec["Year"])
            if key not in all_data:
                all_data[key] = {
                    "Country": rec["Country"],
                    "Country_Code": rec["Country_Code"],
                    "Year": rec["Year"],
                }
            all_data[key][rec["indicator"]] = rec["value"]
        time.sleep(1)

    # Convert to sorted list
    rows = sorted(all_data.values(), key=lambda x: (x.get("Country", ""), x.get("Year", "")))

    print(f"\nTotal merged rows: {len(rows)}")

    # Write CSV
    columns = ["Country", "Country_Code", "Year", "Population", "Population_Density",
               "Urbanization_Rate", "GDP_Per_Capita", "GDP_Growth"]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        # Write metadata/source line
        f.write("# Kaynak: World Bank - World Development Indicators (WDI) - https://data.worldbank.org\n")

        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to: {OUTPUT_PATH}")

    # Quick summary
    countries = set(r.get("Country_Code") for r in rows if r.get("Country_Code"))
    years = set(r.get("Year") for r in rows if r.get("Year"))
    print(f"Countries/regions: {len(countries)}")
    print(f"Years: {min(years)} - {max(years)}")
    print(f"Total rows: {len(rows)}")

    # Show first few rows
    print("\nFirst 5 rows:")
    for r in rows[:5]:
        print({k: r.get(k) for k in columns})


if __name__ == "__main__":
    main()
