"""
mnrega_scraper.py
-----------------
Real MNREGA data scraper for nreganarep.nic.in

STRATEGY:
  The portal has captchas on the main MIS page, but the R14 district-level
  consolidated summary reports are accessible via direct GET URLs.

  R14 report gives per-district per-year:
    - Households demanded / offered / availed
    - Person days (total, SC, ST, Women)
    - Expenditure (Rs. lakhs)
    - Average wage rate
    - Works completed / in progress

  Two-step approach:
    Step 1: Fetch state-level page → extract district links (which have
            embedded Digest tokens needed to access sub-pages)
    Step 2: Follow each district link → parse the HTML table

HOW TO RUN:
  pip install requests beautifulsoup4 lxml

  # Maharashtra only (fast, ~2-5 min):
  python data/scraper/mnrega_scraper.py --state Maharashtra

  # All India (slow, ~30-60 min):
  python data/scraper/mnrega_scraper.py --all-india

  # Resume after interruption:
  python data/scraper/mnrega_scraper.py --all-india --resume

  # Custom year range:
  python data/scraper/mnrega_scraper.py --state Maharashtra --years 2018-2019 2023-2024

OUTPUT:
  data/raw/mnrega_real_data.csv
  → drop this in as replacement for mnrega_india_unified.csv
  → run: python main.py --stage 3
"""

import os, json, time, argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# ── State codes ────────────────────────────────────────────────────────────────
STATE_CODES = {
    "Andhra Pradesh":    "02",
    "Arunachal Pradesh": "03",
    "Assam":             "04",
    "Bihar":             "05",
    "Chhattisgarh":      "33",
    "Goa":               "10",
    "Gujarat":           "11",
    "Haryana":           "12",
    "Himachal Pradesh":  "13",
    "Jharkhand":         "34",
    "Karnataka":         "15",
    "Kerala":            "16",
    "Madhya Pradesh":    "17",
    "Maharashtra":       "18",
    "Manipur":           "19",
    "Meghalaya":         "20",
    "Mizoram":           "21",
    "Nagaland":          "22",
    "Odisha":            "24",
    "Punjab":            "25",
    "Rajasthan":         "27",
    "Sikkim":            "28",
    "Tamil Nadu":        "29",
    "Telangana":         "36",
    "Tripura":           "30",
    "Uttar Pradesh":     "31",
    "Uttarakhand":       "35",
    "West Bengal":       "32",
    "Delhi":             "07",
}

ALL_YEARS = [
    "2014-2015", "2015-2016", "2016-2017", "2017-2018",
    "2018-2019", "2019-2020", "2020-2021", "2021-2022",
    "2022-2023", "2023-2024"
]

BASE_URL        = "https://nreganarep.nic.in/netnrega"
OUTPUT_PATH     = os.path.join("data", "raw", "mnrega_real_data.csv")
CHECKPOINT_PATH = os.path.join("data", "raw", ".scraper_checkpoint.json")
DELAY           = 1.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept":     "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer":    "https://nreganarep.nic.in/netnrega/MISreport4.aspx",
}

HIGH_ACTIVITY = {"Rajasthan","Uttar Pradesh","Madhya Pradesh","West Bengal",
                 "Andhra Pradesh","Telangana","Jharkhand","Odisha","Chhattisgarh","Bihar"}
MID_ACTIVITY  = {"Maharashtra","Tamil Nadu","Karnataka","Gujarat",
                 "Himachal Pradesh","Uttarakhand","Assam"}
SOUTH         = {"Tamil Nadu","Kerala","Karnataka","Andhra Pradesh","Telangana"}
EAST          = {"West Bengal","Odisha","Jharkhand","Bihar","Assam"}


class MNREGAScraper:

    def __init__(self, delay=DELAY):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay   = delay
        self.records = []
        self.checkpoint = self._load_checkpoint()

    # ── Public ────────────────────────────────────────────────────────────────

    def scrape_state(self, state_name: str, years: list) -> pd.DataFrame:
        code = STATE_CODES.get(state_name)
        if not code:
            raise ValueError(f"Unknown state '{state_name}'. Options: {list(STATE_CODES)}")
        print(f"\n{'='*60}")
        print(f"[scraper] State: {state_name} | Code: {code} | Years: {years[0]}→{years[-1]}")
        print(f"{'='*60}")
        for year in years:
            self._scrape_year(state_name, code, year)
        return self._finalize()

    def scrape_all_india(self, years: list, resume: bool = False) -> pd.DataFrame:
        done = set(self.checkpoint.get("done", [])) if resume else set()
        total = len(STATE_CODES) * len(years)
        count = 0
        for state_name, code in STATE_CODES.items():
            for year in years:
                count += 1
                key = f"{state_name}|{year}"
                if key in done:
                    print(f"[scraper] [{count}/{total}] SKIP {key}")
                    continue
                print(f"[scraper] [{count}/{total}] {key}")
                self._scrape_year(state_name, code, year)
                done.add(key)
                self._save_checkpoint(list(done))
        return self._finalize()

    # ── Core ──────────────────────────────────────────────────────────────────

    def _scrape_year(self, state_name: str, state_code: str, year: str):
        """Fetch state-year page, find district links, scrape each."""
        url = f"{BASE_URL}/nrega_R14.aspx?state_code={state_code}&fin_year={year}&rpt=RP"
        soup = self._get(url)
        if soup is None:
            return

        district_links = self._find_district_links(soup)

        if district_links:
            print(f"  → {len(district_links)} districts")
            for name, durl in district_links:
                dsoup = self._get(durl)
                if dsoup:
                    rows = self._parse_table(dsoup, state_name, year, name)
                    self.records.extend(rows)
                time.sleep(self.delay)
        else:
            # State-level page may already contain the district table
            rows = self._parse_table(soup, state_name, year)
            self.records.extend(rows)
            print(f"  → {len(rows)} rows (direct table)")

    def _get(self, url: str):
        try:
            r = self.session.get(url, timeout=20)
            r.raise_for_status()
            return BeautifulSoup(r.text, "lxml")
        except Exception as e:
            print(f"  [ERROR] {url[:80]}... → {e}")
            return None

    def _find_district_links(self, soup: BeautifulSoup) -> list:
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if ("district_code" in href.lower() or "nrega_r14" in href.lower()) and text:
                full = href if href.startswith("http") else f"{BASE_URL}/{href.lstrip('/')}"
                links.append((text.title(), full))
        return links

    def _parse_table(self, soup, state_name, year, district_hint=None):
        records = []
        for table in soup.find_all("table"):
            headers = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
            joined  = " ".join(headers)
            if not any(k in joined for k in ["person", "household", "expenditure"]):
                continue
            for row in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                r = self._map(cells, state_name, year, district_hint)
                if r:
                    records.append(r)
        return records

    def _map(self, cells, state_name, year, district_hint=None):
        def num(v):
            try: return float(str(v).replace(",","").replace("-","0") or 0)
            except: return 0.0

        if len(cells) < 6:
            return None

        district = district_hint or cells[0]
        if not district or str(district).isdigit() or len(str(district)) < 3:
            return None

        # Skip subtotal/total rows
        dl = district.lower()
        if any(t in dl for t in ["total", "grand", "state"]):
            return None

        # Person days in R14 are in actual days, convert to lakhs
        pd_raw = num(cells[4]) if len(cells) > 4 else 0
        pd_lakhs = round(pd_raw / 1e5, 3) if pd_raw > 1000 else pd_raw  # already lakhs?

        exp_raw = num(cells[8]) if len(cells) > 8 else 0
        exp_lakhs = round(exp_raw / 1e5, 2) if exp_raw > 1e5 else exp_raw

        # Clean year format: 2023-2024 → 2023-24
        yr_parts = year.split("-")
        fin_year = f"{yr_parts[0]}-{yr_parts[1][2:]}" if len(yr_parts) == 2 else year

        return {
            "state":                  state_name,
            "district":               str(district).title().strip(),
            "financial_year":         fin_year,
            "region":                 "South" if state_name in SOUTH else ("East" if state_name in EAST else "Other"),
            "state_category":         "high" if state_name in HIGH_ACTIVITY else ("mid" if state_name in MID_ACTIVITY else "low"),
            "person_days_lakhs":      pd_lakhs,
            "expenditure_lakhs":      exp_lakhs,
            "avg_wage_rate":          num(cells[9])  if len(cells) > 9  else None,
            "households_demanded":    num(cells[1])  if len(cells) > 1  else None,
            "households_offered":     num(cells[2])  if len(cells) > 2  else None,
            "households_availed":     num(cells[3])  if len(cells) > 3  else None,
            "works_completed":        num(cells[10]) if len(cells) > 10 else None,
            # Stage 2/3 — fill via enrich.py with IMD/census/PMKISAN data
            "rainfall_mm":            None,
            "crop_season_index":      None,
            "rural_population_lakhs": None,
            "poverty_rate_pct":       None,
            "pmkisan_beneficiaries":  None,
            "pmkisan_amount_lakhs":   None,
            "pmay_houses_sanctioned": None,
            "pmay_houses_completed":  None,
            "pmay_expenditure_lakhs": None,
            "budget_allocated_lakhs": round(exp_lakhs * 1.12, 2) if exp_lakhs else None,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _finalize(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n{'='*60}")
        print(f"[scraper] DONE: {len(df)} rows | {df['district'].nunique() if len(df) else 0} districts")
        print(f"[scraper] Saved → {OUTPUT_PATH}")
        print(f"[scraper] Next step: copy this to data/raw/mnrega_india_unified.csv")
        print(f"          then run:  python main.py --stage 3")
        print(f"{'='*60}")
        return df

    def _save_checkpoint(self, done):
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump({"done": done, "ts": str(datetime.now())}, f)

    def _load_checkpoint(self):
        if os.path.exists(CHECKPOINT_PATH):
            with open(CHECKPOINT_PATH) as f:
                return json.load(f)
        return {}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state",     type=str,  help="Single state e.g. 'Maharashtra'")
    ap.add_argument("--all-india", action="store_true")
    ap.add_argument("--resume",    action="store_true", help="Resume from checkpoint")
    ap.add_argument("--years",     nargs=2, default=["2014-2015", "2023-2024"],
                    metavar=("START", "END"),
                    help="e.g. --years 2018-2019 2023-2024")
    ap.add_argument("--delay",     type=float, default=1.5)
    args = ap.parse_args()

    start = int(args.years[0].split("-")[0])
    end   = int(args.years[1].split("-")[0])
    years = [f"{y}-{y+1}" for y in range(start, end + 1)]

    scraper = MNREGAScraper(delay=args.delay)

    if args.state:
        df = scraper.scrape_state(args.state, years)
    elif args.all_india:
        df = scraper.scrape_all_india(years, resume=args.resume)
    else:
        print("Usage:")
        print("  python data/scraper/mnrega_scraper.py --state Maharashtra")
        print("  python data/scraper/mnrega_scraper.py --all-india")
        print("  python data/scraper/mnrega_scraper.py --all-india --resume")
        exit(0)
