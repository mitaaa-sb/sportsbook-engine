import os
import random
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson
import plotly.graph_objects as go

# ====================================================================================
# 0. PAGE CONFIG
# ====================================================================================
st.set_page_config(
    page_title="Quantitative Sportsbook Odds Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================================================
# 1. SECURE KEY LOADING & CONFIGURATION
# ====================================================================================
def get_secret(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


API_FOOTBALL_KEY = get_secret("API_FOOTBALL_KEY")
FOOTBALL_DATA_KEY = get_secret("FOOTBALL_DATA_KEY")
THE_ODDS_API_KEY = get_secret("THE_ODDS_API_KEY")

API_FOOTBALL_HOST = "v3.football.api-sports.io"
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

DEMO_MODE = not any([API_FOOTBALL_KEY, FOOTBALL_DATA_KEY, THE_ODDS_API_KEY])

LEAGUES = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Serie A": "SA",
    "Bundesliga": "BL1",
    "Ligue 1": "FL1",
}

# API-Football Competition IDs for filtering stats STRICTLY to League matches
API_FOOTBALL_LEAGUE_IDS = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
}

HIST_LEAGUE_MAP = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
}

HIST_SECONDARY_LEAGUE_MAP = {
    "E0": "E1",    
    "SP1": "SP2",  
    "I1": "I2",    
    "D1": "D2",    
    "F1": "F2",    
}

SECOND_TIER_DISPLAY_NAMES = {
    "E1": "Championship",
    "SP2": "Segunda División",
    "I2": "Serie B",
    "D2": "2. Bundesliga",
    "F2": "Ligue 2",
}

LEAGUE_DISCIPLINE_STATS = {
    "Premier League": {"avg_fouls": 21.5, "avg_cards": 4.1},
    "La Liga": {"avg_fouls": 25.5, "avg_cards": 5.2},
    "Serie A": {"avg_fouls": 24.0, "avg_cards": 4.6},
    "Bundesliga": {"avg_fouls": 22.5, "avg_cards": 4.2},
    "Ligue 1": {"avg_fouls": 23.5, "avg_cards": 4.0},
}

PROMOTION_ATTACK_DISCOUNT = 0.55
PROMOTION_DEFENSE_PENALTY = 1.48

_STADIUM_CITIES = {
    "Premier League": ("London", 51.5072, -0.1276),
    "La Liga": ("Madrid", 40.4168, -3.7038),
    "Serie A": ("Milan", 45.4642, 9.1900),
    "Bundesliga": ("Munich", 48.1351, 11.5820),
    "Ligue 1": ("Paris", 48.8566, 2.3522),
}

ODDS_API_SPORT_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
}

LEAGUE_TEAM_POOLS = {
    "Premier League": ["Arsenal", "Man City", "Liverpool", "Chelsea", "Aston Villa", "Tottenham"],
    "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Real Sociedad", "Sevilla", "Villarreal"],
    "Serie A": ["Inter", "Juventus", "AC Milan", "Napoli", "Roma", "Atalanta"],
    "Bundesliga": ["Bayern Munich", "Dortmund", "RB Leipzig", "Leverkusen", "Union Berlin", "Freiburg"],
    "Ligue 1": ["PSG", "Monaco", "Marseille", "Lyon", "Lille", "Nice"],
}

# ====================================================================================
# UNIFIED TEAM NAME STANDARDIZATION ENGINE
# ====================================================================================
HIST_TEAM_ALIASES = {
    "manchester city": "man city", "man city fc": "man city", "manchester city fc": "man city",
    "manchester united": "man united", "man utd": "man united", "manchester united fc": "man united",
    "newcastle united": "newcastle", "newcastle united fc": "newcastle",
    "tottenham hotspur": "tottenham", "tottenham hotspur fc": "tottenham",
    "wolverhampton wanderers": "wolves", "wolverhampton": "wolves", "wolves fc": "wolves",
    "nottingham forest": "nott'm forest", "nottm forest": "nott'm forest", "nottingham forest fc": "nott'm forest",
    "west ham united": "west ham", "west ham united fc": "west ham",
    "brighton & hove albion": "brighton", "brighton and hove albion": "brighton", "brighton fc": "brighton",
    "leicester city": "leicester", "leeds united": "leeds",
    "afc bournemouth": "bournemouth", "bournemouth fc": "bournemouth",
    "atletico madrid": "ath madrid", "atlético madrid": "ath madrid", "atlético de madrid": "ath madrid",
    "athletic club": "ath bilbao", "athletic bilbao": "ath bilbao",
    "real betis": "betis", "celta vigo": "celta", "deportivo alaves": "alaves", "rayo vallecano": "vallecano",
    "ac milan": "milan", "internazionale": "inter", "inter milan": "inter", "hellas verona": "verona",
    "borussia dortmund": "dortmund", "borussia monchengladbach": "m'gladbach", "borussia mönchengladbach": "m'gladbach",
    "bayer leverkusen": "leverkusen", "eintracht frankfurt": "ein frankfurt",
    "1899 hoffenheim": "hoffenheim", "hoffenheim": "hoffenheim",
    "1. fc koln": "fc koln", "1. fc köln": "fc koln", "koln": "fc koln",
    "paris saint germain": "paris sg", "paris saint-germain": "paris sg", "psg": "paris sg",
    "saint-etienne": "st etienne", "as saint-etienne": "st etienne",
    "olympique marseille": "marseille", "olympique lyonnais": "lyon",
    "coventry city": "coventry",
}

def normalize_team_name(name: str) -> str:
    if not name:
        return ""
    n = name.strip()
    suffixes = [" Football Club", " FC", " CF", " AFC", " SC", " SV"]
    prefixes = ["AFC ", "FC ", "CF ", "SC ", "SV "]
    for p in prefixes:
        if n.startswith(p):
            n = n[len(p):].strip()
    for s in suffixes:
        if n.endswith(s):
            n = n[:-len(s)].strip()
            
    clean_key = n.lower().replace("-", " ").replace("'", "").strip()
    return HIST_TEAM_ALIASES.get(clean_key, n).strip()

def api_search_name(name: str) -> str:
    if not name:
        return ""
    n = name.strip()
    for p in ["AFC ", "FC ", "CF ", "SC ", "SV "]:
        if n.startswith(p):
            n = n[len(p):].strip()
    for s in [" Football Club", " FC", " CF", " AFC", " SC", " SV"]:
        if n.endswith(s):
            n = n[: -len(s)].strip()
    return n

def is_team_match(name1: str, name2: str) -> bool:
    if not name1 or not name2:
        return False
    norm1 = normalize_team_name(name1).casefold()
    norm2 = normalize_team_name(name2).casefold()
    
    if norm1 == norm2:
        return True
    if len(norm1) > 3 and len(norm2) > 3:
        if norm1 in norm2 or norm2 in norm1:
            return True
    return False

def _current_season_year() -> int:
    now = dt.datetime.now()
    return now.year if now.month >= 7 else now.year - 1

def _recent_hist_season_codes(n: int = 3) -> list:
    start_year = _current_season_year()
    return [f"{str(y)[-2:]}{str(y + 1)[-2:]}" for y in range(start_year - n + 1, start_year + 1)]

# ====================================================================================
# 2. HISTORICAL DATA ENGINE (MULTI-SEASON ARCHIVE)
# ====================================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_historical_league_data(league_code: str = "E0") -> pd.DataFrame:
    seasons = _recent_hist_season_codes(3)
    dfs = []
    for s in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{s}/{league_code}.csv"
        try:
            df = pd.read_csv(url)
            cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'B365H', 'B365D', 'B365A']
            available_cols = [c for c in cols if c in df.columns]
            df_clean = df[available_cols].dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
            dfs.append(df_clean)
        except Exception:
            continue
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined['Date'] = pd.to_datetime(combined['Date'], dayfirst=True, errors='coerce')
        return combined.sort_values('Date', ascending=False)
    return pd.DataFrame()

def _match_hist_team_rows(hist_df: pd.DataFrame, column: str, team_name: str) -> pd.DataFrame:
    if hist_df.empty or column not in hist_df.columns:
        return hist_df.iloc[0:0]
    target_norm = normalize_team_name(team_name).casefold()
    col = hist_df[column].astype(str)
    exact = hist_df[col.apply(lambda c: normalize_team_name(c).casefold() == target_norm)]
    if not exact.empty:
        return exact
    candidates = col.unique()
    hits = [c for c in candidates if is_team_match(c, team_name)]
    if len(hits) == 1:
        return hist_df[col == hits[0]]
    return hist_df.iloc[0:0]

def _team_hist_side_stats(team_name: str, side: str, primary_df: pd.DataFrame, secondary_df: Optional[pd.DataFrame], league_home_avg_g: float, league_away_avg_g: float) -> dict:
    team_col = 'HomeTeam' if side == 'home' else 'AwayTeam'
    for_col = 'FTHG' if side == 'home' else 'FTAG'
    against_col = 'FTAG' if side == 'home' else 'FTHG'
    
    primary_avg_for = league_home_avg_g if for_col == 'FTHG' else league_away_avg_g
    primary_avg_against = league_home_avg_g if against_col == 'FTHG' else league_away_avg_g

    matches = _match_hist_team_rows(primary_df, team_col, team_name).head(38)
    if not matches.empty:
        gf, ga = float(matches[for_col].mean()), float(matches[against_col].mean())
        return {
            "goals_for": gf, "goals_against": ga,
            "attack": round(gf / primary_avg_for, 3), "defense": round(ga / primary_avg_against, 3),
            "tier": "top-flight", "n": len(matches),
            "raw_attack": None, "raw_defense": None, "raw_goals_for": None, "raw_goals_against": None,
        }

    if secondary_df is not None and not secondary_df.empty:
        sec_matches = _match_hist_team_rows(secondary_df, team_col, team_name).head(38)
        if not sec_matches.empty:
            sec_avg_for = max(secondary_df[for_col].mean(), 1.0)
            sec_avg_against = max(secondary_df[against_col].mean(), 1.0)
            raw_gf, raw_ga = float(sec_matches[for_col].mean()), float(sec_matches[against_col].mean())
            adj_gf = raw_gf * PROMOTION_ATTACK_DISCOUNT
            adj_ga = raw_ga * PROMOTION_DEFENSE_PENALTY
            return {
                "goals_for": adj_gf, "goals_against": adj_ga,
                "attack": round(adj_gf / primary_avg_for, 3), "defense": round(adj_ga / primary_avg_against, 3),
                "tier": "second-tier (promotion-adjusted)", "n": len(sec_matches),
                "raw_attack": round(raw_gf / sec_avg_for, 3), "raw_defense": round(raw_ga / sec_avg_against, 3),
                "raw_goals_for": round(raw_gf, 3), "raw_goals_against": round(raw_ga, 3),
            }

    return {"goals_for": None, "goals_against": None, "attack": 1.0, "defense": 1.0,
            "tier": "no match — league average", "n": 0,
            "raw_attack": None, "raw_defense": None, "raw_goals_for": None, "raw_goals_against": None}

def calculate_historical_lambdas(home_team: str, away_team: str, hist_df: pd.DataFrame, secondary_df: Optional[pd.DataFrame] = None) -> Tuple[float, float, dict, dict, float, float]:
    empty_keys = {"attack": 1.0, "defense": 1.0, "raw_attack": None, "raw_defense": None, "raw_goals_for": None, "raw_goals_against": None, "goals_for": None, "goals_against": None}
    if hist_df.empty:
        empty_info = {**empty_keys, "tier": "no historical data", "n": 0}
        return 1.55, 1.20, empty_info, empty_info, 1.55, 1.20

    league_home_avg_g = max(hist_df['FTHG'].mean(), 1.0)
    league_away_avg_g = max(hist_df['FTAG'].mean(), 1.0)

    home_info = _team_hist_side_stats(home_team, 'home', hist_df, secondary_df, league_home_avg_g, league_away_avg_g)
    away_info = _team_hist_side_stats(away_team, 'away', hist_df, secondary_df, league_home_avg_g, league_away_avg_g)

    lam_home = league_home_avg_g * home_info["attack"] * away_info["defense"]
    lam_away = league_away_avg_g * away_info["attack"] * home_info["defense"]

    return (round(float(lam_home), 3), round(float(lam_away), 3), home_info, away_info, round(float(league_home_avg_g), 3), round(float(league_away_avg_g), 3))

def fetch_team_recent_xg_and_sos(team_name: str, hist_df: pd.DataFrame, n_matches: int = 5) -> dict:
    default_res = {"gf": 1.5, "xgf": 1.5, "ga": 1.2, "xga": 1.2, "opp_def": 1.00, "opp_att": 1.00, "season_gf": 1.5, "season_ga": 1.5}
    if hist_df.empty: return default_res

    league_h_g = max(hist_df['FTHG'].mean(), 1.0)
    league_a_g = max(hist_df['FTAG'].mean(), 1.0)

    # 1. LONG-TERM SEASON AVERAGES (Crucial fix for regression math)
    home_all = _match_hist_team_rows(hist_df, 'HomeTeam', team_name).head(38)
    away_all = _match_hist_team_rows(hist_df, 'AwayTeam', team_name).head(38)
    
    gf_all, ga_all = [], []
    if not home_all.empty:
        gf_all.extend(home_all['FTHG'].tolist())
        ga_all.extend(home_all['FTAG'].tolist())
    if not away_all.empty:
        gf_all.extend(away_all['FTAG'].tolist())
        ga_all.extend(away_all['FTHG'].tolist())
        
    season_gf = round(float(np.mean(gf_all)), 2) if gf_all else league_h_g
    season_ga = round(float(np.mean(ga_all)), 2) if ga_all else league_a_g

    # 2. RECENT FORM STATS
    home_m = _match_hist_team_rows(hist_df, 'HomeTeam', team_name)
    away_m = _match_hist_team_rows(hist_df, 'AwayTeam', team_name)
    combined_matches = pd.concat([home_m, away_m]).sort_values('Date', ascending=False).head(n_matches)
    
    if combined_matches.empty: 
        default_res["season_gf"] = season_gf
        default_res["season_ga"] = season_ga
        return default_res

    gf_list, ga_list, xgf_list, xga_list, opp_def_list, opp_att_list = [], [], [], [], [], []

    for _, row in combined_matches.iterrows():
        is_home = is_team_match(str(row['HomeTeam']), team_name)
        opp_name = row['AwayTeam'] if is_home else row['HomeTeam']

        gf = row['FTHG'] if is_home else row['FTAG']
        ga = row['FTAG'] if is_home else row['FTHG']
        gf_list.append(gf)
        ga_list.append(ga)

        hs = row['HS'] if not pd.isna(row.get('HS')) else 10.0
        as_ = row['AS'] if not pd.isna(row.get('AS')) else 10.0
        hst = row['HST'] if not pd.isna(row.get('HST')) else 3.5
        ast = row['AST'] if not pd.isna(row.get('AST')) else 3.5

        xg_home = 0.32 * hst + 0.03 * max(0, hs - hst)
        xg_away = 0.32 * ast + 0.03 * max(0, as_ - ast)

        xgf_list.append(xg_home if is_home else xg_away)
        xga_list.append(xg_away if is_home else xg_home)

        opp_home_rows = _match_hist_team_rows(hist_df, 'HomeTeam', opp_name).head(38)
        opp_away_rows = _match_hist_team_rows(hist_df, 'AwayTeam', opp_name).head(38)
        
        opp_gf_h = opp_home_rows['FTHG'].mean() if not opp_home_rows.empty else league_h_g
        opp_ga_h = opp_home_rows['FTAG'].mean() if not opp_home_rows.empty else league_a_g
        opp_gf_a = opp_away_rows['FTAG'].mean() if not opp_away_rows.empty else league_a_g
        opp_ga_a = opp_away_rows['FTHG'].mean() if not opp_away_rows.empty else league_h_g

        opp_att = ((opp_gf_h + opp_gf_a) / 2.0) / ((league_h_g + league_a_g) / 2.0)
        opp_def = ((opp_ga_h + opp_ga_a) / 2.0) / ((league_h_g + league_a_g) / 2.0)

        opp_att_list.append(opp_att)
        opp_def_list.append(opp_def)

    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    if len(gf_list) < 5:
        weights = weights[:len(gf_list)] / weights[:len(gf_list)].sum()

    return {
        "gf": round(float(np.average(gf_list, weights=weights)), 2),
        "xgf": round(float(np.average(xgf_list, weights=weights)), 2),
        "ga": round(float(np.average(ga_list, weights=weights)), 2),
        "xga": round(float(np.average(xga_list, weights=weights)), 2),
        "opp_def": round(float(np.average(opp_def_list, weights=weights)), 2),
        "opp_att": round(float(np.average(opp_att_list, weights=weights)), 2),
        "season_gf": season_gf,
        "season_ga": season_ga
    }

# ====================================================================================
# 3. MOCK DATA GENERATORS (DEMO FALLBACKS)
# ====================================================================================
def _mock_fixtures(league: str) -> pd.DataFrame:
    random.seed(hash(league) % 1000)
    teams = LEAGUE_TEAM_POOLS.get(league, LEAGUE_TEAM_POOLS["Premier League"]).copy()
    fixtures = []
    base_date = dt.datetime.now() + dt.timedelta(days=2)
    random.shuffle(teams)
    for i in range(0, len(teams) - 1, 2):
        fixtures.append({
            "fixture_id": f"{league[:2].upper()}{i}",
            "home_team": teams[i],
            "away_team": teams[i + 1],
            "kickoff": (base_date + dt.timedelta(hours=3 * i)).replace(minute=0, second=0, microsecond=0),
        })
    return pd.DataFrame(fixtures)

def _mock_season_stats(league: str) -> pd.DataFrame:
    random.seed(hash(league) % 2000)
    teams = LEAGUE_TEAM_POOLS.get(league, LEAGUE_TEAM_POOLS["Premier League"])
    data = []
    for t in teams:
        data.append({
            "team": t,
            "home_xG_for": round(random.uniform(1.3, 2.3), 2),
            "home_xG_against": round(random.uniform(0.7, 1.5), 2),
            "away_xG_for": round(random.uniform(1.0, 1.9), 2),
            "away_xG_against": round(random.uniform(0.9, 1.8), 2),
        })
    return pd.DataFrame(data)

def _mock_squad(team: str, seed_offset: int = 0, n_games: int = 5) -> pd.DataFrame:
    random.seed((hash(team) + seed_offset) % 10000)
    positions = ["GK", "DF", "DF", "DF", "DF", "MF", "MF", "MF", "FW", "FW", "FW"]
    names_pool = ["Silva", "Rodrigues", "Kovac", "Muller", "Dubois", "Novak", "Andersen", "Fernandez", "Costa", "Brandt", "Diaz"]
    rows = []
    for i, pos in enumerate(positions):
        name = f"{random.choice(names_pool)} {chr(65 + i)}"
        minutes = random.randint(600, 2500)
        xg90 = round(max(0, np.random.normal(0.38 if pos == "FW" else 0.12 if pos == "MF" else 0.02, 0.10)), 3)
        xa90 = round(max(0, np.random.normal(0.22 if pos in ("FW", "MF") else 0.03, 0.08)), 3)
        games_rated = min(n_games, max(0, n_games - random.choice([0, 0, 0, 1, 2, n_games])))
        rating = round(np.random.normal(7.0, 0.5), 2) if games_rated > 0 else 6.8
        key_passes90 = round(max(0, np.random.normal(1.5 if pos == "MF" else 0.6, 0.5)), 2)
        rows.append({
            "player": name, "position": pos, "minutes": minutes,
            "xG90": xg90, "xA90": xa90, "key_passes90": key_passes90,
            "avg_rating": rating, "games_rated": games_rated, "status": "Active",
        })
    return pd.DataFrame(rows)

def _mock_form(team: str) -> pd.DataFrame:
    random.seed(hash(team) % 5000)
    rows = []
    for m in range(5, 0, -1):
        rows.append({
            "matches_ago": m,
            "goals_for": np.random.poisson(1.5),
            "goals_against": np.random.poisson(1.1),
            "xG_for": round(max(0.2, np.random.normal(1.5, 0.4)), 2),
            "xG_against": round(max(0.2, np.random.normal(1.1, 0.3)), 2),
        })
    return pd.DataFrame(rows)

def _mock_odds(home: str, away: str) -> dict:
    random.seed(hash(home + away) % 9999)
    home_p = random.uniform(0.35, 0.55)
    draw_p = random.uniform(0.22, 0.28)
    away_p = 1.0 - home_p - draw_p
    margin = 1.055
    return {
        "1X2": {
            "home": round(1 / (home_p * margin), 2),
            "draw": round(1 / (draw_p * margin), 2),
            "away": round(1 / (away_p * margin), 2),
        },
        "over_2_5": round(1 / (0.52 * margin), 2),
        "under_2_5": round(1 / (0.48 * margin), 2),
        "btts_yes": round(1 / (0.53 * margin), 2),
        "btts_no": round(1 / (0.47 * margin), 2),
        "source": "Mock Data Engine"
    }

# ====================================================================================
# 4. LIVE DATA LAYER
# ====================================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather(lat: float, lon: float, kickoff: dt.datetime) -> dict:
    try:
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "forecast_days": 7, "timezone": "auto",
        }
        r = requests.get(OPEN_METEO_BASE, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        hourly_times = data["hourly"]["time"]
        target = kickoff.strftime("%Y-%m-%dT%H:00")
        idx = hourly_times.index(target) if target in hourly_times else 0
        return {
            "temperature_c": data["hourly"]["temperature_2m"][idx],
            "precipitation_mm": data["hourly"]["precipitation"][idx],
            "wind_speed_kmh": data["hourly"]["wind_speed_10m"][idx],
            "source": "Open-Meteo API",
        }
    except Exception:
        return {"temperature_c": 14.5, "precipitation_mm": 0.0, "wind_speed_kmh": 12.0, "source": "Mock Weather"}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fixtures(league_name: str) -> pd.DataFrame:
    league_code = LEAGUES.get(league_name)
    if FOOTBALL_DATA_KEY:
        try:
            headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
            r = requests.get(f"{FOOTBALL_DATA_BASE}/competitions/{league_code}/matches", headers=headers, params={"status": "SCHEDULED"}, timeout=6)
            r.raise_for_status()
            matches = r.json().get("matches", [])[:8]
            rows = [{
                "fixture_id": m["id"], "home_team": m["homeTeam"]["name"], "away_team": m["awayTeam"]["name"],
                "kickoff": dt.datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")),
            } for m in matches]
            if rows:
                return pd.DataFrame(rows)
        except Exception:
            pass
    return _mock_fixtures(league_name)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_team_season_stats(league_name: str) -> pd.DataFrame:
    league_code = LEAGUES.get(league_name)
    if FOOTBALL_DATA_KEY:
        try:
            headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
            r = requests.get(
                f"{FOOTBALL_DATA_BASE}/competitions/{league_code}/standings",
                headers=headers, timeout=8,
            )
            r.raise_for_status()
            standings = r.json().get("standings", [])
            home_table = next((s["table"] for s in standings if s.get("type") == "HOME"), None)
            away_table = next((s["table"] for s in standings if s.get("type") == "AWAY"), None)

            if home_table and away_table:
                home_lookup = {row["team"]["name"]: row for row in home_table}
                away_lookup = {row["team"]["name"]: row for row in away_table}
                rows = []
                for name in set(home_lookup) & set(away_lookup):
                    h, a = home_lookup[name], away_lookup[name]
                    if not h.get("playedGames") or not a.get("playedGames"):
                        continue
                    rows.append({
                        "team": name,
                        "home_xG_for": round(h["goalsFor"] / h["playedGames"], 2),
                        "home_xG_against": round(h["goalsAgainst"] / h["playedGames"], 2),
                        "away_xG_for": round(a["goalsFor"] / a["playedGames"], 2),
                        "away_xG_against": round(a["goalsAgainst"] / a["playedGames"], 2),
                    })
                if rows:
                    return pd.DataFrame(rows)
        except Exception:
            pass
    return _mock_season_stats(league_name)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_player_recent_ratings(team_id: int, league_name: str = "Premier League", n_games: int = 5) -> dict:
    if not API_FOOTBALL_KEY:
        return {}
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    league_id = API_FOOTBALL_LEAGUE_IDS.get(league_name, 39)
    try:
        r_fx = requests.get(
            f"https://{API_FOOTBALL_HOST}/fixtures",
            headers=headers, 
            params={"team": team_id, "league": league_id, "last": n_games}, 
            timeout=8,
        )
        r_fx.raise_for_status()
        fixture_ids = [f["fixture"]["id"] for f in r_fx.json().get("response", [])]

        ratings: dict = {}
        for fid in fixture_ids:
            r_pl = requests.get(
                f"https://{API_FOOTBALL_HOST}/fixtures/players",
                headers=headers, params={"fixture": fid}, timeout=8,
            )
            r_pl.raise_for_status()
            for team_block in r_pl.json().get("response", []):
                if team_block.get("team", {}).get("id") != team_id:
                    continue
                for p in team_block.get("players", []):
                    name = p.get("player", {}).get("name")
                    stat = (p.get("statistics") or [{}])[0]
                    games = stat.get("games", {}) or {}
                    rating_raw = games.get("rating")
                    minutes = games.get("minutes")
                    if name and rating_raw and minutes:
                        ratings.setdefault(name, []).append(float(rating_raw))

        return {
            name: {"avg_rating": round(sum(vals) / len(vals), 2), "games_rated": len(vals)}
            for name, vals in ratings.items()
        }
    except Exception:
        return {}

def _parse_players_stats_page(response_items: list, pos_map: dict, recent_ratings: dict, fallback_xg90: dict, fallback_xa90: dict) -> list:
    rows = []
    for p in response_items:
        info = p.get("player", {})
        stat = (p.get("statistics") or [{}])[0]
        games = stat.get("games", {}) or {}
        goals = stat.get("goals", {}) or {}
        minutes = games.get("minutes") or 0
        pos = pos_map.get(games.get("position"), "MF")
        goals_total = goals.get("total") or 0
        assists_total = goals.get("assists") or 0
        name = info.get("name")

        season_rating_raw = games.get("rating")
        recent = recent_ratings.get(name)
        
        if recent and recent["games_rated"] > 0:
            rating = recent["avg_rating"]
            games_rated = recent["games_rated"]
        elif season_rating_raw:
            rating = round(float(season_rating_raw), 2)
            games_rated = 0
        else:
            rating = 6.8
            games_rated = 0

        if minutes and minutes > 0:
            xg90 = round(goals_total / minutes * 90, 3)
            xa90 = round(assists_total / minutes * 90, 3)
        else:
            xg90 = fallback_xg90.get(pos, 0.08)
            xa90 = fallback_xa90.get(pos, 0.05)

        rows.append({
            "player": name, "position": pos, "minutes": minutes,
            "xG90": xg90, "xA90": xa90,
            "key_passes90": round(assists_total / max(minutes, 1) * 90 * 1.8, 2),
            "avg_rating": rating, "games_rated": games_rated, "status": "Active",
        })
    return rows

def _fetch_players_stats_all_pages(team_id: int, season: int, league_id: int, headers: dict) -> list:
    items, page, total_pages = [], 1, 1
    while page <= total_pages and page <= 3:
        params = {"team": team_id, "season": season, "league": league_id, "page": page}
        r_stats = requests.get(
            f"https://{API_FOOTBALL_HOST}/players",
            headers=headers, params=params, timeout=8,
        )
        r_stats.raise_for_status()
        payload = r_stats.json()
        total_pages = payload.get("paging", {}).get("total", 1)
        items.extend(payload.get("response", []))
        page += 1
    return items

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_squad(team_name: str, league_name: str = "Premier League", seed_offset: int = 0, n_games: int = 5) -> pd.DataFrame:
    if API_FOOTBALL_KEY:
        try:
            headers = {"x-apisports-key": API_FOOTBALL_KEY}
            clean_search_name = api_search_name(team_name)
            r_team = requests.get(
                f"https://{API_FOOTBALL_HOST}/teams",
                headers=headers, params={"search": clean_search_name}, timeout=6,
            )
            r_team.raise_for_status()
            team_res = r_team.json().get("response", [])

            if team_res:
                team_id = team_res[0]["team"]["id"]
                season = _current_season_year()
                league_id = API_FOOTBALL_LEAGUE_IDS.get(league_name, 39)
                
                pos_map = {"Goalkeeper": "GK", "Defender": "DF", "Midfielder": "MF", "Attacker": "FW"}
                fallback_xg90 = {"FW": 0.30, "MF": 0.10, "DF": 0.02, "GK": 0.0}
                fallback_xa90 = {"FW": 0.15, "MF": 0.16, "DF": 0.03, "GK": 0.0}

                recent_ratings = fetch_player_recent_ratings(team_id, league_name=league_name, n_games=n_games)
                items = _fetch_players_stats_all_pages(team_id, season, league_id, headers)
                data_source = f"API-Football stats ({season} {league_name})"

                if not items:
                    items = _fetch_players_stats_all_pages(team_id, season - 1, league_id, headers)
                    data_source = f"API-Football stats ({season - 1} {league_name})"

                rows = _parse_players_stats_page(items, pos_map, recent_ratings, fallback_xg90, fallback_xa90)

                if not rows:
                    r_squad = requests.get(
                        f"https://{API_FOOTBALL_HOST}/players/squads",
                        headers=headers, params={"team": team_id}, timeout=8,
                    )
                    r_squad.raise_for_status()
                    squad_res = r_squad.json().get("response", [])
                    if squad_res and "players" in squad_res[0]:
                        data_source = f"API-Football roster ({league_name})"
                        for p in squad_res[0]["players"]:
                            pos = pos_map.get(p.get("position"), "MF")
                            name = p.get("name")
                            recent = recent_ratings.get(name)
                            rating, games_rated = (recent["avg_rating"], recent["games_rated"]) if recent else (6.8, 0)
                            rows.append({
                                "player": name, "position": pos, "minutes": 0,
                                "xG90": fallback_xg90.get(pos, 0.08), "xA90": fallback_xa90.get(pos, 0.05),
                                "key_passes90": 0.5 if pos == "MF" else 0.3,
                                "avg_rating": rating, "games_rated": games_rated, "status": "Active",
                            })

                if rows:
                    df = pd.DataFrame(rows)
                    df["data_source"] = data_source
                    return df
        except Exception:
            pass

    df = _mock_squad(team_name, seed_offset, n_games)
    df["data_source"] = f"Mock ({league_name} - API-Football unavailable)"
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_team_form(team_name: str, league_name: str = "Premier League") -> pd.DataFrame:
    league_code = LEAGUES.get(league_name)
    if FOOTBALL_DATA_KEY:
        try:
            headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
            r_std = requests.get(
                f"{FOOTBALL_DATA_BASE}/competitions/{league_code}/standings",
                headers=headers, timeout=8,
            )
            r_std.raise_for_status()
            total_table = next(
                (s["table"] for s in r_std.json().get("standings", []) if s.get("type") == "TOTAL"), []
            )
            team_id = None
            team_name_match = _find_unique_team_match([row["team"]["name"] for row in total_table], team_name)
            if team_name_match:
                team_id = next((row["team"]["id"] for row in total_table if row["team"]["name"] == team_name_match), None)

            if team_id:
                r_m = requests.get(
                    f"{FOOTBALL_DATA_BASE}/teams/{team_id}/matches",
                    headers=headers, params={"status": "FINISHED", "limit": 5}, timeout=8,
                )
                r_m.raise_for_status()
                matches = r_m.json().get("matches", [])[-5:]
                rows = []
                for idx, m in enumerate(matches):
                    is_home = is_team_match(m["homeTeam"]["name"], team_name)
                    gf = m["score"]["fullTime"]["home"] if is_home else m["score"]["fullTime"]["away"]
                    ga = m["score"]["fullTime"]["away"] if is_home else m["score"]["fullTime"]["home"]
                    if gf is None or ga is None:
                        continue
                    rows.append({
                        "matches_ago": len(matches) - idx,
                        "goals_for": gf, "goals_against": ga,
                        "xG_for": gf, "xG_against": ga,
                    })
                if len(rows) >= 3:
                    return pd.DataFrame(rows)
        except Exception:
            pass
    return _mock_form(team_name)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_odds(home_team: str, away_team: str, league_name: str = "Premier League") -> dict:
    sport_key = ODDS_API_SPORT_KEYS.get(league_name, "soccer_epl")
    if not THE_ODDS_API_KEY: 
        return _mock_odds(home_team, away_team)
        
    try:
        r = requests.get(f"{ODDS_API_BASE}/sports/{sport_key}/odds", params={"apiKey": THE_ODDS_API_KEY, "regions": "uk,eu", "markets": "h2h,totals,btts", "oddsFormat": "decimal"}, timeout=8)
        r.raise_for_status()
        events = r.json()
        for ev in events:
            if is_team_match(ev["home_team"], home_team) and is_team_match(ev["away_team"], away_team):
                h2h_prices, ou_prices, btts_prices = {"h": [], "d": [], "a": []}, {"o": [], "u": []}, {"y": [], "n": []}
                
                for book in ev["bookmakers"]:
                    for m in book["markets"]:
                        if m["key"] == "h2h":
                            for o in m["outcomes"]:
                                if is_team_match(o["name"], ev["home_team"]): h2h_prices["h"].append(o["price"])
                                elif o["name"] == "Draw": h2h_prices["d"].append(o["price"])
                                else: h2h_prices["a"].append(o["price"])
                        elif m["key"] == "totals":
                            for o in m["outcomes"]:
                                if o.get("point") != 2.5:
                                    continue
                                if o["name"] == "Over": ou_prices["o"].append(o["price"])
                                elif o["name"] == "Under": ou_prices["u"].append(o["price"])
                        elif m["key"] == "btts":
                            for o in m["outcomes"]:
                                if o["name"] == "Yes": btts_prices["y"].append(o["price"])
                                elif o["name"] == "No": btts_prices["n"].append(o["price"])

                if not h2h_prices["h"]: return _mock_odds(home_team, away_team)
                
                return {
                    "1X2": {
                        "home": round(np.mean(h2h_prices["h"]), 2),
                        "draw": round(np.mean(h2h_prices["d"]), 2),
                        "away": round(np.mean(h2h_prices["a"]), 2)
                    },
                    "over_2_5": round(np.mean(ou_prices["o"]), 2) if ou_prices["o"] else None,
                    "under_2_5": round(np.mean(ou_prices["u"]), 2) if ou_prices["u"] else None,
                    "btts_yes": round(np.mean(btts_prices["y"]), 2) if btts_prices["y"] else None,
                    "btts_no": round(np.mean(btts_prices["n"]), 2) if btts_prices["n"] else None,
                    "source": f"Consensus Average ({len(ev['bookmakers'])} Bookies)"
                }
    except Exception:
        pass
        
    return _mock_odds(home_team, away_team)

# ====================================================================================
# 5. QUANTITATIVE MODELING ENGINE
# ====================================================================================
def _find_unique_team_match(candidates, target_name: str):
    target_norm = normalize_team_name(target_name).casefold()
    exact = [c for c in candidates if normalize_team_name(c).casefold() == target_norm]
    if exact:
        return exact[0]
    hits = [c for c in candidates if is_team_match(c, target_name)]
    return hits[0] if len(hits) == 1 else None

def calculate_team_base_lambdas(
    home_team: str, away_team: str, team_stats: pd.DataFrame
) -> Tuple[float, float, dict, dict, float, float]:
    league_home_avg = max(team_stats['home_xG_for'].mean(), 1.0)
    league_away_avg = max(team_stats['away_xG_for'].mean(), 1.0)
    
    home_match = _find_unique_team_match(team_stats['team'].astype(str).tolist(), home_team)
    away_match = _find_unique_team_match(team_stats['team'].astype(str).tolist(), away_team)
    h_stat = team_stats.loc[team_stats['team'] == home_match] if home_match else team_stats.iloc[0:0]
    a_stat = team_stats.loc[team_stats['team'] == away_match] if away_match else team_stats.iloc[0:0]
    
    home_att = (h_stat['home_xG_for'].values[0] if not h_stat.empty else 1.55) / league_home_avg
    away_def = (a_stat['away_xG_against'].values[0] if not a_stat.empty else 1.20) / league_home_avg
    
    away_att = (a_stat['away_xG_for'].values[0] if not a_stat.empty else 1.20) / league_away_avg
    home_def = (h_stat['home_xG_against'].values[0] if not h_stat.empty else 1.55) / league_away_avg
    
    base_lam_home = league_home_avg * home_att * away_def
    base_lam_away = league_away_avg * away_att * home_def

    home_info = {
        "goals_for": float(h_stat['home_xG_for'].values[0]) if not h_stat.empty else None,
        "goals_against": float(h_stat['home_xG_against'].values[0]) if not h_stat.empty else None,
        "attack": round(float(home_att), 3), "defense": round(float(home_def), 3),
        "tier": "live standings" if not h_stat.empty else "no match — league average", "n": None,
        "raw_attack": None, "raw_defense": None, "raw_goals_for": None, "raw_goals_against": None,
    }
    away_info = {
        "goals_for": float(a_stat['away_xG_for'].values[0]) if not a_stat.empty else None,
        "goals_against": float(a_stat['away_xG_against'].values[0]) if not a_stat.empty else None,
        "attack": round(float(away_att), 3), "defense": round(float(away_def), 3),
        "tier": "live standings" if not a_stat.empty else "no match — league average", "n": None,
        "raw_attack": None, "raw_defense": None, "raw_goals_for": None, "raw_goals_against": None,
    }
    
    return (round(base_lam_home, 3), round(base_lam_away, 3), home_info, away_info,
            round(float(league_home_avg), 3), round(float(league_away_avg), 3))

def player_impact_score(squad: pd.DataFrame, active_mask: dict) -> Tuple[float, pd.DataFrame]:
    if squad.empty: return 1.0, squad
    squad = squad.copy()
    squad["status"] = squad["player"].map(lambda p: "Active" if active_mask.get(p, True) else "Injured/Out")
    squad["PIV"] = (squad["xG90"] + 0.85 * squad["xA90"]).round(3)

    total_piv = squad["PIV"].sum()
    active_piv = squad.loc[squad["status"] == "Active", "PIV"].sum()

    squad["absence_penalty_%"] = np.where(
        squad["status"] == "Injured/Out",
        (squad["PIV"] / max(total_piv, 1e-6) * 100).round(1), 0.0
    )

    availability_ratio = active_piv / max(total_piv, 1e-6)
    piv_multiplier = 0.60 + 0.40 * availability_ratio
    return round(piv_multiplier, 3), squad

def team_form_rating_0_100(form_df: pd.DataFrame) -> float:
    weights = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    df = form_df.sort_values("matches_ago", ascending=False).reset_index(drop=True)
    if len(df) < 5:
        weights = weights[-len(df):] / weights[-len(df):].sum()
    xg_diff = df["xG_for"] - df["xG_against"]
    score = float((xg_diff * weights).sum())
    return round(min(max(50 + score * 25, 0), 100), 1)

def weather_modifier(weather: dict) -> float:
    penalty = 1.0
    wind, precip = weather.get("wind_speed_kmh", 0), weather.get("precipitation_mm", 0)
    if wind > 35: penalty -= 0.10
    elif wind > 25: penalty -= 0.05
    if precip > 5: penalty -= 0.06
    elif precip > 2: penalty -= 0.03
    return round(max(penalty, 0.80), 3)

def rest_modifier(days_rest: int) -> float:
    if days_rest <= 2: return 0.93
    elif days_rest == 3: return 0.97
    return 1.00

@dataclass
class TeamModelInputs:
    name: str
    base_lambda: float
    piv_multiplier: float
    form_rating: float
    weather_mult: float
    rest_mult: float
    xg_att_reg_mult: float = 1.0
    xg_def_reg_mult_opponent: float = 1.0
    press_mult: float = 1.0
    travel_mult: float = 1.0
    squad_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def lambda_final(self) -> float:
        raw = self.base_lambda * self.piv_multiplier * self.weather_mult * self.rest_mult * self.xg_att_reg_mult * self.xg_def_reg_mult_opponent * self.press_mult * self.travel_mult
        combined_mult = (raw / self.base_lambda) if self.base_lambda > 0 else 1.0
        combined_mult = max(0.5, min(2.0, combined_mult))
        val = self.base_lambda * combined_mult
        return round(max(val, 0.05), 3)

def dixon_coles_tau(x: int, y: int, lam_home: float, lam_away: float, rho: float = -0.06) -> float:
    if x == 0 and y == 0: return 1 - lam_home * lam_away * rho
    elif x == 0 and y == 1: return 1 + lam_home * rho
    elif x == 1 and y == 0: return 1 + lam_away * rho
    elif x == 1 and y == 1: return 1 - rho
    return 1.0

def scoreline_matrix(lam_home: float, lam_away: float, max_goals: int = 9, rho: float = -0.06) -> np.ndarray:
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away) * dixon_coles_tau(i, j, lam_home, lam_away, rho)
            matrix[i, j] = max(p, 0)
    matrix /= matrix.sum()
    return matrix

# ====================================================================================
# 6. DEVIGGING & KELLY TRADING LAYER
# ====================================================================================
def fair_odds(prob: float) -> float:
    return round(1 / prob, 3) if prob > 1e-6 else float("inf")

def apply_margin(probs: dict, target_margin_pct: float) -> dict:
    margin_factor = 1 + (target_margin_pct / 100)
    return {k: round(fair_odds(v) / margin_factor, 2) for k, v in probs.items()}

def devig_proportional(home_o: float, draw_o: float, away_o: float) -> Optional[Tuple[float, float, float]]:
    if not all([home_o, draw_o, away_o]) or any(o <= 1.0 for o in [home_o, draw_o, away_o]):
        return None
    raw_h, raw_d, raw_a = 1 / home_o, 1 / draw_o, 1 / away_o
    overround = raw_h + raw_d + raw_a
    return round(raw_h / overround, 4), round(raw_d / overround, 4), round(raw_a / overround, 4)

def devig_two_way(odd_a: float, odd_b: float) -> Optional[Tuple[float, float]]:
    if not odd_a or not odd_b or odd_a <= 1.0 or odd_b <= 1.0:
        return None
    raw_a, raw_b = 1 / odd_a, 1 / odd_b
    overround = raw_a + raw_b
    return round(raw_a / overround, 4), round(raw_b / overround, 4)

def kelly_stake(model_prob: float, book_odds: float, fraction: float = 0.25) -> float:
    if not book_odds or book_odds <= 1.0 or model_prob <= 0:
        return 0.0
    b = book_odds - 1.0
    p = model_prob
    q = 1.0 - p
    f = (b * p - q) / b
    return round(max(0.0, f * fraction * 100), 2)

def derive_markets(matrix: np.ndarray) -> dict:
    max_goals = matrix.shape[0] - 1
    home_win = float(np.tril(matrix, -1).sum())
    away_win = float(np.triu(matrix, 1).sum())
    draw = float(np.trace(matrix))

    over_25 = sum(matrix[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j > 2.5)
    btts_yes = sum(matrix[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i > 0 and j > 0)

    return {
        "1X2": {"home": home_win, "draw": draw, "away": away_win},
        "over_2_5": over_25, "under_2_5": 1 - over_25,
        "btts_yes": btts_yes, "btts_no": 1 - btts_yes,
        "ah_home_-0.5": home_win, "ah_away_+0.5": draw + away_win,
    }

# ====================================================================================
# 7. STREAMLIT SIDEBAR & INPUTS
# ====================================================================================
st.sidebar.title("⚽ Sportsbook Odds Engine")
st.sidebar.caption("Quant Modeling & Devigged Edge Detection")

if DEMO_MODE:
    st.sidebar.warning("⚠️ **DEMO MODE**: Set API keys in `st.secrets` for live feeds.")

st.sidebar.subheader("API & Data Connections")
st.sidebar.markdown(f"{'🟢' if API_FOOTBALL_KEY else '🔴'} API-Football")
st.sidebar.markdown(f"{'🟢' if FOOTBALL_DATA_KEY else '🔴'} football-data.org")
st.sidebar.markdown(f"{'🟢' if THE_ODDS_API_KEY else '🔴'} The Odds API")
st.sidebar.markdown("🟢 Historical CSV Engine (Football-Data.co.uk)")
st.sidebar.markdown("🟢 Open-Meteo Weather")

if st.sidebar.button("🔄 Refresh Rosters & Live Stats", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("Cached for 15–30 min per source. Historical data cached for 24 hours.")

st.sidebar.divider()
league = st.sidebar.selectbox("Select League", list(LEAGUES.keys()))
fixtures_df = fetch_fixtures(league)
fixtures_df["label"] = fixtures_df["home_team"] + " vs " + fixtures_df["away_team"]
match_label = st.sidebar.selectbox("Select Match", fixtures_df["label"])
match_row = fixtures_df.loc[fixtures_df["label"] == match_label].iloc[0]

home_team, away_team, kickoff = match_row["home_team"], match_row["away_team"], match_row["kickoff"]

hist_code = HIST_LEAGUE_MAP.get(league, "E0")
hist_df = fetch_historical_league_data(hist_code)
secondary_code = HIST_SECONDARY_LEAGUE_MAP.get(hist_code)
secondary_hist_df = fetch_historical_league_data(secondary_code) if secondary_code else pd.DataFrame()

st.sidebar.divider()
st.sidebar.subheader("Trading Parameters")
target_margin = st.sidebar.slider("Target Model Margin (%)", 2.0, 8.0, 5.0, 0.5)
kelly_fraction = st.sidebar.slider("Kelly Fractional Sizing", 0.1, 1.0, 0.25, 0.05, help="0.25 = Quarter Kelly")

st.sidebar.subheader("Rest Differential (Days Rest)")
c_r1, c_r2 = st.sidebar.columns(2)
with c_r1:
    home_rest = st.number_input(f"{home_team}", min_value=1, max_value=14, value=7)
with c_r2:
    away_rest = st.number_input(f"{away_team}", min_value=1, max_value=14, value=7)

# --- ADVANCED MODIFIERS ---
st.sidebar.divider()
st.sidebar.subheader("🔬 Advanced Modifiers")

auto_home_sos = fetch_team_recent_xg_and_sos(home_team, hist_df, n_matches=5)
auto_away_sos = fetch_team_recent_xg_and_sos(away_team, hist_df, n_matches=5)

with st.sidebar.expander("1. xG Regression & Opponent Strength", expanded=False):
    st.caption("Auto-synced from historical db. Adjusts for finishing variance & SoS.")
    
    st.markdown(f"**{home_team} (Recent Form)**")
    c_h1, c_h2, c_h3 = st.columns(3)
    with c_h1:
        h_gf = st.number_input(f"{home_team} GF", value=float(auto_home_sos["gf"]), step=0.1, key="h_gf_in")
        h_xgf = st.number_input(f"{home_team} xGF", value=float(auto_home_sos["xgf"]), step=0.1, key="h_xgf_in")
    with c_h2:
        h_ga = st.number_input(f"{home_team} GA", value=float(auto_home_sos["ga"]), step=0.1, key="h_ga_in")
        h_xga = st.number_input(f"{home_team} xGA", value=float(auto_home_sos["xga"]), step=0.1, key="h_xga_in")
    with c_h3:
        h_opp_def = st.number_input("Opp Avg Def (β)", value=float(auto_home_sos["opp_def"]), step=0.05, key="h_opp_d")
        h_opp_att = st.number_input("Opp Avg Att (α)", value=float(auto_home_sos["opp_att"]), step=0.05, key="h_opp_a")

    st.markdown(f"**{away_team} (Recent Form)**")
    c_a1, c_a2, c_a3 = st.columns(3)
    with c_a1:
        a_gf = st.number_input(f"{away_team} GF", value=float(auto_away_sos["gf"]), step=0.1, key="a_gf_in")
        a_xgf = st.number_input(f"{away_team} xGF", value=float(auto_away_sos["xgf"]), step=0.1, key="a_xgf_in")
    with c_a2:
        a_ga = st.number_input(f"{away_team} GA", value=float(auto_away_sos["ga"]), step=0.1, key="a_ga_in")
        a_xga = st.number_input(f"{away_team} xGA", value=float(auto_away_sos["xga"]), step=0.1, key="a_xga_in")
    with c_a3:
        a_opp_def = st.number_input("Opp Avg Def (β)", value=float(auto_away_sos["opp_def"]), step=0.05, key="a_opp_d")
        a_opp_att = st.number_input("Opp Avg Att (α)", value=float(auto_away_sos["opp_att"]), step=0.05, key="a_opp_a")

    h_true_xgf = h_xgf / h_opp_def if h_opp_def > 0 else h_xgf
    h_true_xga = h_xga / h_opp_att if h_opp_att > 0 else h_xga
    a_true_xgf = a_xgf / a_opp_def if a_opp_def > 0 else a_xgf
    a_true_xga = a_xga / a_opp_att if a_opp_att > 0 else a_xga

    # 1. Calculate raw regressed expected goals
    h_regressed_xgf = (h_true_xgf * 0.70) + (h_gf * 0.30)
    h_regressed_xga = (h_true_xga * 0.70) + (h_ga * 0.30)
    a_regressed_xgf = (a_true_xgf * 0.70) + (a_gf * 0.30)
    a_regressed_xga = (a_true_xga * 0.70) + (a_ga * 0.30)

    # 2. Divide by the LONG-TERM season averages (Fixes the Unlucky but Terrible paradox)
    h_att_reg_mult = h_regressed_xgf / auto_home_sos["season_gf"] if auto_home_sos["season_gf"] > 0 else 1.0
    h_def_reg_mult = h_regressed_xga / auto_home_sos["season_ga"] if auto_home_sos["season_ga"] > 0 else 1.0
    a_att_reg_mult = a_regressed_xgf / auto_away_sos["season_gf"] if auto_away_sos["season_gf"] > 0 else 1.0
    a_def_reg_mult = a_regressed_xga / auto_away_sos["season_ga"] if auto_away_sos["season_ga"] > 0 else 1.0

    h_att_reg_mult = float(np.clip(h_att_reg_mult, 0.7, 1.3))
    h_def_reg_mult = float(np.clip(h_def_reg_mult, 0.7, 1.3))
    a_att_reg_mult = float(np.clip(a_att_reg_mult, 0.7, 1.3))
    a_def_reg_mult = float(np.clip(a_def_reg_mult, 0.7, 1.3))

    st.caption(f"**{home_team} Modifiers:** Attack **{h_att_reg_mult:.2f}x** | Defense **{h_def_reg_mult:.2f}x**")
    st.caption(f"**{away_team} Modifiers:** Attack **{a_att_reg_mult:.2f}x** | Defense **{a_def_reg_mult:.2f}x**")

with st.sidebar.expander("2. Tactical Pressing (PPDA & Tilt)", expanded=False):
    st.caption("High-pressing teams get an attack rating boost based on Field Tilt.")
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        h_ppda = st.number_input(f"{home_team} PPDA", value=12.0, step=0.5)
    with c_p2:
        a_ppda = st.number_input(f"{away_team} PPDA", value=12.0, step=0.5)
    
    tilt_diff = st.number_input("Field Tilt Diff (%)", value=0.0, step=1.0)
    press_edge_h = a_ppda / h_ppda if h_ppda > 0 else 1.0
    press_edge_a = h_ppda / a_ppda if a_ppda > 0 else 1.0
    
    h_press_mult = (1.0 + 0.05 * (press_edge_h - 1.0)) * (1.0 + tilt_diff/100 * 0.05)
    a_press_mult = (1.0 + 0.05 * (press_edge_a - 1.0)) * (1.0 - tilt_diff/100 * 0.05)
    h_press_mult = float(np.clip(h_press_mult, 0.85, 1.15))
    a_press_mult = float(np.clip(a_press_mult, 0.85, 1.15))
    st.caption(f"Multiplier: Home **{h_press_mult:.2f}x** | Away **{a_press_mult:.2f}x**")

with st.sidebar.expander("3. Travel Distance & Fatigue", expanded=False):
    st.caption("Long-distance travel decays away team expectations.")
    away_travel_km = st.number_input("Away Travel Distance (km)", value=0.0, step=100.0)
    away_travel_mult = max(0.90, 1.0 - (away_travel_km / 25000.0))
    st.caption(f"Away Travel Penalty: **{away_travel_mult:.2f}x**")

with st.sidebar.expander("4. Cards & Referee Strictness", expanded=False):
    st.caption("Foul rates impact the likelihood of red cards, scaling the Dixon-Coles ρ parameter.")
    league_defaults = LEAGUE_DISCIPLINE_STATS.get(league, {"avg_fouls": 22.0, "avg_cards": 4.5})
    
    c_rc1, c_rc2 = st.columns(2)
    with c_rc1:
        h_fouls = st.number_input("Home Avg Fouls", value=league_defaults["avg_fouls"]/2, step=0.5)
        ref_cards = st.number_input("Referee Avg Cards", value=league_defaults["avg_cards"], step=0.1)
    with c_rc2:
        a_fouls = st.number_input("Away Avg Fouls", value=league_defaults["avg_fouls"]/2, step=0.5)
        league_cards = st.number_input("League Avg Cards", value=league_defaults["avg_cards"], step=0.1)
        
    expected_cards = ref_cards * ((h_fouls + a_fouls) / league_defaults["avg_fouls"])
    ref_strictness = expected_cards / league_cards if league_cards > 0 else 1.0
    base_rho = -0.06
    adjusted_rho = base_rho * ref_strictness
    st.caption(f"Expected Match Cards: **{expected_cards:.1f}** | Adjusted Dixon-Coles ρ: **{adjusted_rho:.3f}**")

st.sidebar.divider()
st.sidebar.subheader("Player Availability")
rating_window = st.sidebar.slider(
    "Player Form Window (games)", 3, 5, 5,
    help="avg_rating below is each player's average rating over their last N matches."
)

home_squad_raw = fetch_squad(home_team, league_name=league, seed_offset=1, n_games=rating_window)
away_squad_raw = fetch_squad(away_team, league_name=league, seed_offset=2, n_games=rating_window)

with st.sidebar.expander(f"🏠 {home_team} Lineup"):
    home_active = {row["player"]: st.checkbox(f"{row['player']} ({row['position']})", value=True, key=f"h_{row['player']}") for _, row in home_squad_raw.iterrows()}

with st.sidebar.expander(f"🚗 {away_team} Lineup"):
    away_active = {row["player"]: st.checkbox(f"{row['player']} ({row['position']})", value=True, key=f"a_{row['player']}") for _, row in away_squad_raw.iterrows()}

# ====================================================================================
# 8. COMPUTATION & DATA PROCESSING
# ====================================================================================
city, lat, lon = _STADIUM_CITIES.get(league, ("London", 51.5072, -0.1276))
weather = fetch_weather(lat, lon, kickoff)
w_mult = weather_modifier(weather)

lambda_source = "mock"
home_tier_info = away_tier_info = {"tier": "n/a", "n": 0, "attack": 1.0, "defense": 1.0, "goals_for": None, "goals_against": None}
league_home_avg_used = league_away_avg_used = 1.55

if not hist_df.empty:
    lambda_source = f"historical CSV ({len(hist_df)} matches, {hist_code})"
    (base_lam_home, base_lam_away, home_tier_info, away_tier_info,
     league_home_avg_used, league_away_avg_used) = calculate_historical_lambdas(
        home_team, away_team, hist_df, secondary_hist_df
    )
else:
    season_stats = fetch_team_season_stats(league)
    (base_lam_home, base_lam_away, home_tier_info, away_tier_info,
     league_home_avg_used, league_away_avg_used) = calculate_team_base_lambdas(
        home_team, away_team, season_stats
    )
    lambda_source = "live standings" if FOOTBALL_DATA_KEY else "mock season stats"

st.sidebar.divider()
st.sidebar.subheader("⚙️ Team Rating Engine Controls")

rating_mode = st.sidebar.radio(
    "Rating Mode",
    options=["Automated (Data-Driven)", "Custom Manual Override"],
    index=0,
)

def _overall_rating(attack: float, defense: float) -> float:
    return round(min(max(50 + (attack - defense) * 25, 0), 100), 1)

st.sidebar.caption(f"{home_team}: {home_tier_info['tier']} · {away_team}: {away_tier_info['tier']}")

if rating_mode == "Automated (Data-Driven)":
    st.sidebar.success("🟢 Ratings auto-synced from data source")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(f"{home_team} α (Att)", home_tier_info["attack"])
        st.metric(f"{home_team} β (Def)", home_tier_info["defense"])
    with col2:
        st.metric(f"{away_team} α (Att)", away_tier_info["attack"])
        st.metric(f"{away_team} β (Def)", away_tier_info["defense"])

    home_attack_eff, home_defense_eff = home_tier_info["attack"], home_tier_info["defense"]
    away_attack_eff, away_defense_eff = away_tier_info["attack"], away_tier_info["defense"]
    ratings_overridden = False
else:
    st.sidebar.warning("✏️ Custom override active")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.markdown(f"**{home_team}**")
        home_attack_eff = st.number_input("Attack (α)", 0.10, 3.00, value=float(home_tier_info["attack"]), step=0.05, key="h_att_c")
        home_defense_eff = st.number_input("Defense (β)", 0.10, 3.00, value=float(home_tier_info["defense"]), step=0.05, key="h_def_c")
    with c2:
        st.markdown(f"**{away_team}**")
        away_attack_eff = st.number_input("Attack (α)", 0.10, 3.00, value=float(away_tier_info["attack"]), step=0.05, key="a_att_c")
        away_defense_eff = st.number_input("Defense (β)", 0.10, 3.00, value=float(away_tier_info["defense"]), step=0.05, key="a_def_c")
    ratings_overridden = True

if ratings_overridden:
    base_lam_home = round(league_home_avg_used * home_attack_eff * away_defense_eff, 3)
    base_lam_away = round(league_away_avg_used * away_attack_eff * home_defense_eff, 3)

home_form_df = fetch_team_form(home_team, league)
away_form_df = fetch_team_form(away_team, league)

home_piv_mult, home_squad = player_impact_score(home_squad_raw, home_active)
away_piv_mult, away_squad = player_impact_score(away_squad_raw, away_active)

home_model = TeamModelInputs(
    name=home_team, base_lambda=base_lam_home, piv_multiplier=home_piv_mult,
    form_rating=team_form_rating_0_100(home_form_df),
    weather_mult=w_mult, rest_mult=rest_modifier(home_rest), 
    xg_att_reg_mult=h_att_reg_mult, 
    xg_def_reg_mult_opponent=a_def_reg_mult,
    press_mult=h_press_mult, travel_mult=1.0, squad_table=home_squad,
)

away_model = TeamModelInputs(
    name=away_team, base_lambda=base_lam_away, piv_multiplier=away_piv_mult,
    form_rating=team_form_rating_0_100(away_form_df),
    weather_mult=w_mult, rest_mult=rest_modifier(away_rest), 
    xg_att_reg_mult=a_att_reg_mult, 
    xg_def_reg_mult_opponent=h_def_reg_mult,
    press_mult=a_press_mult, travel_mult=away_travel_mult, squad_table=away_squad,
)

lam_home, lam_away = home_model.lambda_final, away_model.lambda_final
matrix = scoreline_matrix(lam_home, lam_away, max_goals=9, rho=adjusted_rho)
model_probs = derive_markets(matrix)
market_odds = fetch_market_odds(home_team, away_team, league)

model_odds_1x2 = apply_margin(model_probs["1X2"], target_margin)
model_odds_totals = apply_margin({"over_2_5": model_probs["over_2_5"], "under_2_5": model_probs["under_2_5"]}, target_margin)
model_odds_btts = apply_margin({"btts_yes": model_probs["btts_yes"], "btts_no": model_probs["btts_no"]}, target_margin)

# ====================================================================================
# 9. DASHBOARD DISPLAY
# ====================================================================================
st.title(f"{home_team} vs {away_team}")
st.caption(f"{league} · Kickoff {kickoff.strftime('%A %d %B, %H:%M')} · Venue: {city}")

def _tier_warning(team_name: str, info: dict) -> Optional[str]:
    if info["tier"] == "second-tier (promotion-adjusted)":
        return (f"ℹ️ {team_name} has no top-flight matches in the scanned window (likely newly promoted) — "
                f"using their last {info['n']} second-tier matches with a promotion adjustment "
                f"(attack ×{PROMOTION_ATTACK_DISCOUNT}, defense ×{PROMOTION_DEFENSE_PENALTY}) instead of "
                f"the raw league average.")
    if info["tier"] == "no match — league average":
        return (f"⚠️ Couldn't find {team_name} in the top-flight OR second-tier data for the scanned window — "
                f"using the 1.0 league-average fallback. If this team should have data, check "
                f"`HIST_TEAM_ALIASES` for a naming mismatch.")
    return None

if "historical CSV" in lambda_source:
    for _team, _info in ((home_team, home_tier_info), (away_team, away_tier_info)):
        _msg = _tier_warning(_team, _info)
        if _msg:
            (st.warning if "⚠️" in _msg else st.info)(_msg)
    tier_msg = f" · {home_team}: {home_tier_info['tier']} ({home_tier_info['n']} matches) · {away_team}: {away_tier_info['tier']} ({away_tier_info['n']} matches)"
    st.caption(f"λ base source: **{lambda_source}**{tier_msg}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("λ Home (Final xG)", lam_home)
    st.caption(f"Base {base_lam_home} → PIV×{home_piv_mult:.2f} · Reg×{h_att_reg_mult:.2f} (Att) / {a_def_reg_mult:.2f} (Opp Def) · Press×{h_press_mult:.2f}")
with c2:
    st.metric("λ Away (Final xG)", lam_away)
    st.caption(f"Base {base_lam_away} → PIV×{away_piv_mult:.2f} · Reg×{a_att_reg_mult:.2f} (Att) / {h_def_reg_mult:.2f} (Opp Def) · Trvl×{away_travel_mult:.2f}")
with c3:
    st.metric(f"{home_team} Form", f"{home_model.form_rating}/100")
    st.metric(f"{away_team} Form", f"{away_model.form_rating}/100")
    st.caption("Informational only — recent-form effect on λ flows through the Reg× multipliers above, not this metric.")
with c4:
    st.metric("🌡️ Temp", f"{weather['temperature_c']}°C")
    st.metric("💨 Wind", f"{weather['wind_speed_kmh']} km/h")

with st.expander("📊 Rating Calculation Breakdown"):
    primary_league_name = league
    secondary_league_name = SECOND_TIER_DISPLAY_NAMES.get(secondary_code, secondary_code or "n/a")

    summary_rows = []
    for team_name, info in ((home_team, home_tier_info), (away_team, away_tier_info)):
        if info["tier"] == "second-tier (promotion-adjusted)":
            summary_rows.append({"Team": team_name, "League Context": f"{secondary_league_name} Baseline",
                                  "Attack Rating (α)": info["raw_attack"], "Defense Rating (β)": info["raw_defense"]})
            summary_rows.append({"Team": team_name, "League Context": f"{primary_league_name} Adjusted",
                                  "Attack Rating (α)": info["attack"], "Defense Rating (β)": info["defense"]})
        else:
            label = f"{primary_league_name} Baseline" if info["tier"] == "top-flight" else f"League Average ({info['tier']})"
            summary_rows.append({"Team": team_name, "League Context": label,
                                  "Attack Rating (α)": info["attack"], "Defense Rating (β)": info["defense"]})

    st.markdown("**Rating Summary Table**")
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    st.caption(
        f"Base λ formula: Home = {league_home_avg_used} × α_home × β_away ≈ {base_lam_home} · "
        f"Away = {league_away_avg_used} × α_away × β_home ≈ {base_lam_away}. "
        f"Final λ additionally applies PIV, form, weather, rest, xG-regression, press, and travel multipliers "
        f"(see captions above the heatmap)."
    )

st.divider()

st.subheader("👥 Player Form & Impact Penalties (Current League Matches)")
home_squad_source = home_squad["data_source"].iloc[0] if "data_source" in home_squad.columns and len(home_squad) else "unknown"
away_squad_source = away_squad["data_source"].iloc[0] if "data_source" in away_squad.columns and len(away_squad) else "unknown"
st.caption(f"Squad data source — {home_team}: **{home_squad_source}** · {away_team}: **{away_squad_source}**")

pc1, pc2 = st.columns(2)
display_cols = ["player", "position", "avg_rating", "games_rated", "xG90", "xA90", "status", "absence_penalty_%"]
with pc1:
    st.markdown(f"**{home_team} Lineup & Stats**")
    st.dataframe(home_squad[display_cols].sort_values("avg_rating", ascending=False, na_position="last"), hide_index=True)
with pc2:
    st.markdown(f"**{away_team} Lineup & Stats**")
    st.dataframe(away_squad[display_cols].sort_values("avg_rating", ascending=False, na_position="last"), hide_index=True)

st.divider()

st.subheader("💰 Odds Engine, Consensus Market & Kelly Staking")

book_1x2 = market_odds.get("1X2", {})
devig_1x2 = devig_proportional(book_1x2.get("home", 0), book_1x2.get("draw", 0), book_1x2.get("away", 0))
devig_ou = devig_two_way(market_odds.get("over_2_5"), market_odds.get("under_2_5"))
devig_btts = devig_two_way(market_odds.get("btts_yes"), market_odds.get("btts_no"))

def build_trade_row(market_label, model_prob, model_odd, book_odd, fair_mkt_prob):
    if not book_odd or book_odd <= 1.0:
        return {"Market": market_label, "Model Odds": model_odd, "Consensus Odds": "N/A", "Edge (pp)": 0, "Kelly Stake": "0.0%", "Signal": "N/A"}

    devigged = True
    if fair_mkt_prob is None:
        fair_mkt_prob = 1 / book_odd
        devigged = False

    edge_pp = round((model_prob - fair_mkt_prob) * 100, 2)
    stake_pct = kelly_stake(model_prob, book_odd, kelly_fraction)

    if not devigged: signal = "⚠️ NO DEVIG"
    else: signal = "🟢 VALUE" if edge_pp >= 2.0 and stake_pct > 0 else ("🔴 OVERPRICED" if edge_pp <= -2.0 else "⚪ FAIR")

    return {
        "Market": market_label,
        "Model Prob %": round(model_prob * 100, 1),
        "Model Odds": model_odd,
        "Consensus Odds": book_odd,
        "Fair Market Prob %": round(fair_mkt_prob * 100, 1),
        "Edge (pp)": edge_pp,
        "Kelly Stake %": f"{stake_pct}%",
        "Signal": signal
    }

fair_mkt_h, fair_mkt_d, fair_mkt_a = devig_1x2 if devig_1x2 else (None, None, None)
fair_ou_over, fair_ou_under = devig_ou if devig_ou else (None, None)
fair_btts_yes, fair_btts_no = devig_btts if devig_btts else (None, None)

trade_rows = [
    build_trade_row(f"1X2 — {home_team} Win", model_probs["1X2"]["home"], model_odds_1x2["home"], book_1x2.get("home"), fair_mkt_h),
    build_trade_row("1X2 — Draw", model_probs["1X2"]["draw"], model_odds_1x2["draw"], book_1x2.get("draw"), fair_mkt_d),
    build_trade_row(f"1X2 — {away_team} Win", model_probs["1X2"]["away"], model_odds_1x2["away"], book_1x2.get("away"), fair_mkt_a),
    build_trade_row("Over 2.5 Goals", model_probs["over_2_5"], model_odds_totals["over_2_5"], market_odds.get("over_2_5"), fair_ou_over),
    build_trade_row("Under 2.5 Goals", model_probs["under_2_5"], model_odds_totals["under_2_5"], market_odds.get("under_2_5"), fair_ou_under),
    build_trade_row("BTTS — Yes", model_probs["btts_yes"], model_odds_btts["btts_yes"], market_odds.get("btts_yes"), fair_btts_yes),
]

st.dataframe(pd.DataFrame(trade_rows), hide_index=True)
if market_odds.get("source"): st.caption(f"Odds Source: {market_odds['source']}")

st.divider()

st.subheader("🔥 Scoreline Probability Matrix Heatmap")
heat_labels = list(range(matrix.shape[0]))
fig = go.Figure(data=go.Heatmap(
    z=matrix * 100, x=[f"{away_team} {g}" for g in heat_labels], y=[f"{home_team} {g}" for g in heat_labels],
    colorscale="YlOrRd", text=np.round(matrix * 100, 1), texttemplate="%{text}%", textfont={"size": 11}, hoverongaps=False
))
fig.update_layout(
    title=f"Scoreline Probability Distribution (%) — Adjusted Dixon-Coles ρ: {adjusted_rho:.4f}",
    xaxis_title=f"Away Goals ({away_team})", yaxis_title=f"Home Goals ({home_team})", height=500, margin=dict(l=40, r=40, t=50, b=40)
)
st.plotly_chart(fig)

st.divider()
st.subheader("📥 Export Analysis")

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Odds & Edge Analysis (CSV)",
    data=convert_df_to_csv(pd.DataFrame(trade_rows)),
    file_name=f"{home_team}_vs_{away_team}_odds_analysis.csv",
    mime="text/csv",
)
