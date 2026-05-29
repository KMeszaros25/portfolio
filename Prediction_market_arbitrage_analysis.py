import os
import time
import json
import requests
import pandas as pd
from datetime import datetime

# =====================================================================
# 1. Directory & Pagination Fetcher
# =====================================================================
class MarketDirectoryFetcher:
    def __init__(self):
        self.kalshi_api = "https://external-api.kalshi.com/trade-api/v2"
        self.poly_gamma = "https://gamma-api.polymarket.com"

    def get_filtered_kalshi_markets(self, keywords: list[str], series_tickers: list[str] = None) -> pd.DataFrame:
        print("Fetching targeted Kalshi market directories...")
        print(" -> Collecting LIVE, Future-Expiring, Tightly Contested crypto markets...")
        
        url = f"{self.kalshi_api}/markets"
        data_rows = []
        current_ts = int(time.time())
        
        if series_tickers:
            for series in series_tickers:
                cursor = None
                pages = 0
                valid_count = 0
                
                while True:
                    # FIX 1: Restored the series_ticker parameter to bypass the 30,000+ sports markets!
                    params = {
                        "limit": 100, 
                        "status": "open", 
                        "series_ticker": series
                    }
                    if cursor:
                        params["cursor"] = cursor
                        
                    try:
                        response = requests.get(url, params=params, timeout=10)
                        if response.status_code != 200:
                            break
                            
                        data = response.json()
                        markets = data.get("markets", [])
                        
                        if not markets:
                            break
                            
                        for m in markets:
                            ticker = m.get("ticker", "")
                            
                            # Ban Bracket (-B) markets. We only want Target (-T) markets to match Polymarket
                            if "-B" in ticker:
                                continue
                                
                            title = m.get("title", "")
                            subtitle = m.get("subtitle", "")
                            full_title = f"{title} [{subtitle}]" if subtitle else title

                            with open('k_debug_markets.txt', "a", encoding="utf-8") as f:
                                f.write(full_title + "\n")
                            
                            close_time_str = m.get("close_time")
                            is_future = True
                            if close_time_str:
                                try:
                                    close_ts = int(pd.to_datetime(close_time_str).timestamp())
                                    if close_ts < current_ts:
                                        is_future = False
                                except Exception:
                                    pass
                            
                            if not is_future:
                                continue
                            
                            yes_bid_dollars = float(m.get("yes_bid_dollars", 0))
                            yes_bid_cents = int(yes_bid_dollars * 100)
                            
                            is_tightly_contested = (1 <= yes_bid_cents <= 99)
                            
                            if not is_tightly_contested:
                                continue
                                
                            if any(k.lower() in full_title.lower() for k in keywords):
                                data_rows.append({
                                    "kalshi_ticker": ticker, 
                                    "kalshi_title": full_title
                                })
                                valid_count += 1
                                
                        cursor = data.get("cursor")
                        
                        if not cursor or valid_count >= 250:
                            break 
                        time.sleep(0.1)
                        
                        pages += 1
                        if pages >= 150: 
                            break
                            
                    except Exception as e:
                        print(f"Error fetching Kalshi: {e}")
                        break
                        
        print(f"     [*] Successfully extracted {len(data_rows)} target markets across {len(series_tickers)} specific Kalshi series.")
        return pd.DataFrame(data_rows)

    def get_filtered_polymarket_events(self, keywords: list[str]) -> pd.DataFrame:
        print("Paginating through Polymarket event directory...")
        print(" -> Extracting INDIVIDUAL STRIKES for highly liquid LIVE events...")
        
        url = f"{self.poly_gamma}/events"
        data_rows = []
        offset = 0
        limit = 100
        max_retries = 3
        valid_count = 0
        current_ts = int(time.time())
        
        while True:
            params = {
                "limit": limit, 
                "active": "true", 
                "closed": "false", 
                "offset": offset,
                "order": "endDate",
                "ascending": "true" 
            }
            events = []
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code != 200:
                        break
                    
                    events = response.json()
                    break 
                except requests.exceptions.RequestException:
                    time.sleep(2)
            else:
                break

            if not events:
                break
                
            for event in events:
                event_title = event.get("title", "")
                
                end_date_str = event.get("endDate")
                if end_date_str:
                    try:
                        end_ts = int(pd.to_datetime(end_date_str).timestamp())
                        if end_ts < current_ts:
                            continue 
                    except:
                        pass
                
                if any(k.lower() in event_title.lower() for k in keywords):
                    for market in event.get("markets", []):
                        volume_usd = float(market.get("volume", 0))
                        
                        if volume_usd >= 10:
                            market_group_title = market.get("groupItemTitle", "")
                            market_question = market.get("question", "")
                            
                            if market_group_title and market_group_title not in ["Yes", "No"]:
                                full_poly_title = f"{event_title} [{market_group_title}]"
                            elif market_question and market_question != event_title:
                                full_poly_title = f"{event_title} [{market_question}]"
                            else:
                                full_poly_title = event_title
                            
                            clob_token_ids = market.get("clobTokenIds", [])
                            if isinstance(clob_token_ids, str):
                                try:
                                    clob_token_ids = json.loads(clob_token_ids.replace("'", '"'))
                                except:
                                    clob_token_ids = []
                                    
                            market_id = clob_token_ids[0] if isinstance(clob_token_ids, list) and len(clob_token_ids) > 0 else market.get("conditionId")
                            
                            if market_id:
                                data_rows.append({
                                    "poly_market_id": market_id,
                                    "poly_title": full_poly_title
                                })
                                valid_count += 1
            
            offset += limit
            
            if valid_count >= 500 or offset >= 9900 or len(events) < limit:
                break
            time.sleep(0.2)

        df = pd.DataFrame(data_rows)
        if not df.empty:
            df = df.drop_duplicates(subset=['poly_market_id']).reset_index(drop=True)
        return df


# =====================================================================
# 2. LLM Semantic Matcher
# =====================================================================
class LLMMarketMatcher:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")

    def find_matches(self, kalshi_df: pd.DataFrame, poly_df: pd.DataFrame) -> pd.DataFrame:
        if kalshi_df.empty or poly_df.empty:
            return pd.DataFrame()

        k_titles = kalshi_df['kalshi_title'].drop_duplicates().tolist()[:500]
        all_p_titles = poly_df['poly_title'].drop_duplicates().tolist()
        all_matches = []
        
        print(f"\nBatch Processing {len(all_p_titles)} Polymarket STRIKES against {len(k_titles)} Kalshi STRIKES...")

        schema = {
            "type": "OBJECT",
            "properties": {
                "matches": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "kalshi_title": {"type": "STRING"},
                            "poly_title": {"type": "STRING"},
                            "confidence_score": {"type": "NUMBER"},
                            "match_reasoning": {"type": "STRING"}
                        },
                        "required": ["kalshi_title", "poly_title", "confidence_score", "match_reasoning"]
                    }
                }
            },
            "required": ["matches"]
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={self.api_key}"
        chunk_size = 100
        
        for i in range(0, len(all_p_titles), chunk_size):
            p_titles_chunk = all_p_titles[i:i+chunk_size]
            prompt = f"""
            You are an expert quantitative analyst building a statistical arbitrage engine.
            Match Kalshi markets with Polymarket markets that share the SAME UNDERLYING ASSET, EXACT SAME TIMEFRAME, AND EXACT SAME STRIKE PRICE.
            
            CRITICAL TIMEFRAME & STRIKE RULE: 
            The markets MUST cover the exact same date and specific time window. 
            Furthermore, you must perfectly align the strike prices found in the brackets []. 
            Do NOT pair a $85,000 Kalshi bracket with a $86,000 Polymarket bracket. They must match perfectly!

            For example: Kalshi's "Bitcoin price range on May 18? [$85,000]" and Polymarket's "Bitcoin price on May 18? [$85,000]" SHOULD be paired.

            Kalshi Titles:
            {k_titles}

            Polymarket Titles:
            {p_titles_chunk}

            Only return pairs where you have high confidence (>0.90).
            """

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema, "temperature": 0.1}
            }

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=150)
                    if response.status_code in [429, 503]:
                        time.sleep(65)
                        continue
                    if response.status_code != 200:
                        break
                        
                    result_dict = response.json()
                    text_response = result_dict['candidates'][0]['content']['parts'][0]['text']
                    chunk_matches = json.loads(text_response).get('matches', [])
                    if chunk_matches:
                        print(f"     [+] Found {len(chunk_matches)} exact strike match(es) in this chunk!")
                        all_matches.extend(chunk_matches)
                    break 
                except Exception:
                    time.sleep(2)
            time.sleep(12)

        matched_df = pd.DataFrame(all_matches)
        if matched_df.empty: return matched_df
        matched_df = matched_df.merge(kalshi_df[['kalshi_title', 'kalshi_ticker']], on='kalshi_title', how='left')
        matched_df = matched_df.merge(poly_df[['poly_title', 'poly_market_id']], on='poly_title', how='left')
        return matched_df.drop_duplicates(subset=['kalshi_ticker', 'poly_market_id']).reset_index(drop=True)


# =====================================================================
# 3. Live L2 Order Book Arbitrage Scanner
# =====================================================================
class LiveOrderBookArbScanner:
    def __init__(self):
        self.kalshi_api = "https://external-api.kalshi.com/trade-api/v2/markets"
        self.poly_clob = "https://clob.polymarket.com/book"

    def get_kalshi_l2(self, ticker: str):
        url = f"{self.kalshi_api}/{ticker}/orderbook"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                
                ob = data.get('orderbook_fp') or data.get('orderbook', {})
                
                if 'yes_dollars' in ob or 'no_dollars' in ob:
                    yes_orders = ob.get('yes_dollars', [])
                    no_orders = ob.get('no_dollars', [])
                    
                    top_bid = max([float(o[0]) for o in yes_orders]) if yes_orders else 0.0
                    top_ask = (1.0 - max([float(o[0]) for o in no_orders])) if no_orders else 1.0
                    
                else:
                    yes_orders = ob.get('yes', [])
                    no_orders = ob.get('no', [])
                    
                    top_bid = max([float(o[0])/100.0 for o in yes_orders]) if yes_orders else 0.0
                    top_ask = (1.0 - max([float(o[0])/100.0 for o in no_orders])) if no_orders else 1.0
                    
                return top_bid, top_ask
        except Exception:
            pass
        return 0.0, 1.0

    def get_poly_l2(self, token_id: str):
        clean_token_id = str(token_id).strip()
        url = f"{self.poly_clob}?token_id={clean_token_id}"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                
                top_bid = max([float(b['price']) for b in bids]) if bids else 0.0
                top_ask = min([float(a['price']) for a in asks]) if asks else 1.0
                return top_bid, top_ask
        except Exception:
            pass
        return 0.0, 1.0

    def get_l2_spreads(self, kalshi_ticker: str, poly_market_id: str):
        k_bid, k_ask = self.get_kalshi_l2(kalshi_ticker)
        p_bid, p_ask = self.get_poly_l2(poly_market_id)
        
        spread_A = p_bid - k_ask  # Buy Kalshi YES, Sell Poly YES
        spread_B = k_bid - p_ask  # Buy Poly YES, Sell Kalshi YES
        
        return k_bid, k_ask, p_bid, p_ask, spread_A, spread_B


# =====================================================================
# 4. Main Execution Orchestrator
# =====================================================================
if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    YOUR_API_KEY = "" 
    
    USE_LOCAL_CSVS = False 
    
    try:
        with open('API_KEY.txt', 'r') as file:
            YOUR_API_KEY = file.read().strip()
    except FileNotFoundError:
        pass
    
    TARGET_KEYWORDS = ["Bitcoin", "BTC", "Ethereum", "ETH"]
    
    # FIX 2: Added ALL Kalshi timeframe identifiers: 
    # Daily (KXBTC), Hourly (KXBTCD), and 15-Minute (KXBTC15M) 
    TARGET_KALSHI_SERIES = ["KXBTC", "KXBTCD", "KXBTC15M", "KXETH", "KXETHD", "KXETH15M"]
    # ---------------------

    api_key = YOUR_API_KEY if YOUR_API_KEY else os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: No API key found. Please ensure 'API_KEY.txt' contains your key.")
        exit(1)
        
    os.environ["GEMINI_API_KEY"] = api_key
    print(f"Starting LIVE Arbitrage Engine pipeline for keywords: {TARGET_KEYWORDS}\n")
            
    if not USE_LOCAL_CSVS and os.path.exists("debug_matches.csv"):
        try:
            os.remove("debug_matches.csv")
        except Exception:
            pass
            
    if USE_LOCAL_CSVS and os.path.exists("debug_kalshi.csv") and os.path.exists("debug_poly.csv"):
        print("\n--- LOADING DIRECTORIES FROM LOCAL CSVS ---")
        kalshi_filtered_df = pd.read_csv("debug_kalshi.csv")
        poly_filtered_df = pd.read_csv("debug_poly.csv")
        poly_filtered_df['poly_market_id'] = poly_filtered_df['poly_market_id'].apply(
            lambda x: json.loads(x.replace("'", '"'))[0] if isinstance(x, str) and x.startswith('[') else x
        )
    else:
        dir_fetcher = MarketDirectoryFetcher()
        kalshi_filtered_df = dir_fetcher.get_filtered_kalshi_markets(TARGET_KEYWORDS, series_tickers=TARGET_KALSHI_SERIES)
        poly_filtered_df = dir_fetcher.get_filtered_polymarket_events(TARGET_KEYWORDS)
        kalshi_filtered_df.to_csv("debug_kalshi.csv", index=False)
        poly_filtered_df.to_csv("debug_poly.csv", index=False)

    if USE_LOCAL_CSVS and os.path.exists("debug_matches.csv"):
        print("\n--- LOADING MATCHES FROM LOCAL CSV ---")
        matches_df = pd.read_csv("debug_matches.csv")
        if 'poly_market_id' in matches_df.columns:
            matches_df = matches_df.drop(columns=['poly_market_id'])
        matches_df = matches_df.merge(poly_filtered_df[['poly_title', 'poly_market_id']], on='poly_title', how='inner')
    else:
        matcher = LLMMarketMatcher()
        matches_df = matcher.find_matches(kalshi_filtered_df, poly_filtered_df)
        if not matches_df.empty:
            matches_df.to_csv("debug_matches.csv", index=False)

    if not matches_df.empty:
        print(f"\nInitializing Continuous L2 Order Book Logger for {len(matches_df)} pairs...")
        scanner = LiveOrderBookArbScanner()
        
        log_file = "live_spread_log.csv"
        
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("timestamp,kalshi_ticker,poly_id,k_bid,k_ask,p_bid,p_ask,spread_buy_k_sell_p,spread_buy_p_sell_k,arb_opportunity\n")
        
        print(f"-> Logging all ticks to: {log_file}")
        print("-> Press Ctrl+C to stop continuous monitoring.\n")
        
        try:
            sweep_count = 0
            while True:
                sweep_count += 1
                print(f"--- Sweeping Order Books (Sweep #{sweep_count}) ---")
                
                arb_count = 0
                for index, match in matches_df.iterrows():
                    k_ticker = match['kalshi_ticker']
                    p_id = match['poly_market_id']
                    
                    k_bid, k_ask, p_bid, p_ask, spread_A, spread_B = scanner.get_l2_spreads(k_ticker, p_id)
                    
                    arb_found = False
                    if spread_A > 0.001 or spread_B > 0.001:
                        arb_found = True
                        arb_count += 1
                        print(f" [$$$] ARB ALERT! \n       Kalshi: {match['kalshi_title']} (Ask: ${k_ask:.3f}) \n       Poly:   {match['poly_title']} (Bid: ${p_bid:.3f}) \n       Spread A: {spread_A:.3f} | Spread B: {spread_B:.3f}\n")
                    else:
                        print(f" [{k_ticker}] K: ${k_bid:.3f}/${k_ask:.3f} | P: ${p_bid:.3f}/${p_ask:.3f} -> Spreads: {spread_A:.3f} / {spread_B:.3f}")
                        
                    ts = datetime.now().isoformat()
                    with open(log_file, "a") as f:
                        f.write(f"{ts},{k_ticker},{p_id},{k_bid},{k_ask},{p_bid},{p_ask},{spread_A:.4f},{spread_B:.4f},{arb_found}\n")
                    
                    time.sleep(0.5)
                
                print(f"Sweep #{sweep_count} complete. Waiting 5s before next sweep...\n")
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n[!] Continuous logging stopped by user.")
    else:
        print("\nPipeline finished. No matches found.")