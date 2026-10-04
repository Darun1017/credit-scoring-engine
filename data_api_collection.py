"""
IntelliCredit - API Data Collection (unbuffered output)
"""
import requests
import pandas as pd
import os
import sys
import time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "api")
os.makedirs(OUT, exist_ok=True)

# Force unbuffered output
def log(msg):
    print(msg, flush=True)

# ===== 1. WORLD BANK API =====
log("=" * 50)
log("WORLD BANK API (India Macro Indicators)")
log("=" * 50)

indicators = {
    "FP.CPI.TOTL.ZG": "cpi_inflation_rate",
    "FR.INR.RINR": "real_interest_rate",
    "SL.UEM.TOTL.ZS": "unemployment_rate",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_rate",
    "FI.RES.TOTL.CD": "total_reserves_usd",
    "GC.DOD.TOTL.GD.ZS": "govt_debt_pct_gdp",
}

all_data = {}
for code, col in indicators.items():
    try:
        url = f"https://api.worldbank.org/v2/country/IND/indicator/{code}?date=2015:2024&format=json&per_page=50"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if len(data) > 1 and data[1]:
                for item in data[1]:
                    y = int(item["date"])
                    if y not in all_data:
                        all_data[y] = {"year": y}
                    all_data[y][col] = item["value"]
                log(f"  OK: {col} ({len(data[1])} records)")
            else:
                log(f"  --: {col} (no data)")
        else:
            log(f"  !!: {col} (HTTP {r.status_code})")
    except Exception as e:
        log(f"  !!: {col} ({e})")

if all_data:
    wb_df = pd.DataFrame(sorted(all_data.values(), key=lambda x: x["year"]))
    wb_path = os.path.join(OUT, "worldbank_india_macro.csv")
    wb_df.to_csv(wb_path, index=False)
    log(f"\n  Saved: {wb_path}")
    log(f"  Rows: {len(wb_df)}, Cols: {len(wb_df.columns)}")
    log(f"\n{wb_df.to_string(index=False)}\n")

# ===== 2. FRED API =====
log("=" * 50)
log("FRED API (US Macro Indicators)")
log("=" * 50)

FRED_KEY = os.environ.get("FRED_API_KEY", "ad3c23da2dec34d480c92a2af3911a3b")
if not FRED_KEY:
    log("  SKIPPED - no FRED_API_KEY environment variable set")
    log("  Register free: https://fred.stlouisfed.org/docs/api/api_key.html")
else:
    # Monthly series
    monthly_series = {
        "CPIAUCSL": "cpi_index",
        "CPILFESL": "core_cpi_index",
        "FEDFUNDS": "fed_funds_rate",
        "DGS10": "treasury_10yr_yield",
        "MORTGAGE30US": "mortgage_30yr_rate",
        "UNRATE": "unemployment_rate",
        "CIVPART": "labor_force_participation_rate",
        "TOTALSL": "total_consumer_credit_billions",
        "REVOLSL": "revolving_consumer_credit_billions",
        "NONREVSL": "nonrevolving_credit_billions",
        "PSAVERT": "personal_saving_rate_pct",
        "UMCSENT": "consumer_sentiment_index",
    }
    
    # Quarterly delinquency & debt burden series
    quarterly_series = {
        "DRCCLACBS": "credit_card_delinquency_rate",
        "DRCLACBS": "consumer_loan_delinquency_rate",
        "DRSFRMACBS": "mortgage_delinquency_rate",
        "TDSP": "debt_service_ratio_pct_disposable_income",
    }

    frames = []
    
    # 1. Fetch monthly series
    for sid, col in monthly_series.items():
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={sid}&api_key={FRED_KEY}&observation_start=2010-01-01"
                f"&frequency=m&aggregation_method=avg&file_type=json"
            )
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                obs = r.json().get("observations", [])
                if obs:
                    temp = pd.DataFrame(obs)[["date", "value"]]
                    temp.columns = ["date", col]
                    temp[col] = pd.to_numeric(temp[col].replace(".", None), errors="coerce")
                    frames.append(temp)
                    log(f"  OK: {col} ({len(obs)} monthly observations)")
        except Exception as e:
            log(f"  !!: {col} error ({e})")

    # 2. Fetch quarterly series without frequency restriction
    for sid, col in quarterly_series.items():
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={sid}&api_key={FRED_KEY}&observation_start=2010-01-01"
                f"&file_type=json"
            )
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                obs = r.json().get("observations", [])
                if obs:
                    temp = pd.DataFrame(obs)[["date", "value"]]
                    temp.columns = ["date", col]
                    temp[col] = pd.to_numeric(temp[col].replace(".", None), errors="coerce")
                    frames.append(temp)
                    log(f"  OK: {col} ({len(obs)} quarterly observations)")
        except Exception as e:
            log(f"  !!: {col} error ({e})")

    if frames:
        fred_df = frames[0]
        for f in frames[1:]:
            fred_df = fred_df.merge(f, on="date", how="outer")
        fred_df = fred_df.sort_values("date").reset_index(drop=True)
        
        # Forward fill quarterly values across monthly dates for continuous time-series
        fred_df = fred_df.ffill()
        
        fred_path = os.path.join(OUT, "fred_macro_expanded.csv")
        fred_df.to_csv(fred_path, index=False)
        
        # Also try default path if not locked
        try:
            fred_df.to_csv(os.path.join(OUT, "fred_macro.csv"), index=False)
        except PermissionError:
            pass
            
        log(f"\n  Saved: {fred_path}")
        log(f"  Rows: {len(fred_df)}, Total Indicators: {len(fred_df.columns) - 1}")
        log(f"\nSample Preview (Most Recent Observations):")
        log(fred_df.tail(6)[["date", "cpi_index", "fed_funds_rate", "unemployment_rate", "credit_card_delinquency_rate"]].to_string(index=False))

# ===== 3. CFPB API =====
log("")
log("=" * 50)
log("CFPB API (Consumer Financial Complaints)")
log("=" * 50)

products = ["Payday+loan", "Consumer+Loan", "Debt+collection"]
all_records = []

for product in products:
    pname = product.replace("+", " ")
    log(f"  Fetching: {pname}...")
    try:
        url = (
            f"https://www.consumerfinance.gov/data-research/consumer-complaints/"
            f"search/api/v1/?size=50&product={product}&has_narrative=true"
        )
        r = requests.get(url, timeout=45)
        if r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            for h in hits:
                s = h["_source"]
                all_records.append({
                    "complaint_id": s.get("complaint_id"),
                    "date_received": s.get("date_received"),
                    "product": s.get("product"),
                    "sub_product": s.get("sub_product"),
                    "issue": s.get("issue"),
                    "complaint_narrative": s.get("complaint_what_happened", ""),
                    "company": s.get("company"),
                    "state": s.get("state"),
                    "company_response": s.get("company_response"),
                })
            log(f"    -> Got {len(hits)} records")
        else:
            log(f"    -> HTTP {r.status_code}")
    except requests.exceptions.ReadTimeout:
        log(f"    -> TIMEOUT (CFPB servers slow, try later)")
    except Exception as e:
        log(f"    -> ERROR: {e}")
    time.sleep(1)

if all_records:
    cfpb_df = pd.DataFrame(all_records)
    cfpb_path = os.path.join(OUT, "cfpb_complaints.csv")
    cfpb_df.to_csv(cfpb_path, index=False)
    log(f"\n  Saved: {cfpb_path}")
    log(f"  Total records: {len(cfpb_df)}")
    with_narr = cfpb_df[cfpb_df["complaint_narrative"].str.len() > 10]
    log(f"  With narratives: {len(with_narr)}")
    if len(with_narr) > 0:
        sample = with_narr.iloc[0]["complaint_narrative"][:200]
        log(f"\n  Sample narrative:\n  {sample}...")
else:
    log("  No CFPB data collected (servers may be unreachable)")

# ===== SUMMARY =====
log("")
log("=" * 50)
log("COLLECTION COMPLETE")
log("=" * 50)
log(f"Output directory: {os.path.abspath(OUT)}")

for f in os.listdir(OUT):
    fp = os.path.join(OUT, f)
    size = os.path.getsize(fp)
    log(f"  {f}: {size / 1024:.1f} KB")
