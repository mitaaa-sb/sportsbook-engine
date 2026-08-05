import os
import math
import random
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, Tuple

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

HIST_LEAGUE_MAP = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
}

# Second-tier equivalent for each top-flight league, on football-data.co.uk's
# own code scheme. Used ONLY as a fallback for a team with zero matches in
# the top-flight window — i.e. a newly promoted club (e.g. Coventry City
# going up to the Premier League for 2026/27 with no E0 history yet). This is
# not a naming-mismatch fix; it's a genuinely different data source.
HIST_SECONDARY_LEAGUE_MAP = {
    "E0": "E1",    # Premier League <- Championship
    "SP1": "SP2",  # La Liga <- Segunda División
    "I1": "I2",    # Serie A <- Serie B
    "D1": "D2",    # Bundesliga <- 2. Bundesliga
    "F1": "F2",    # Ligue 1 <- Ligue 2
}

# Display names for the second tier, used only in the Rating Calculation
# Breakdown UI so the transparency table reads "Championship Baseline"
# rather than the raw football-data.co.uk code "E1 Baseline".
SECOND_TIER_DISPLAY_NAMES = {
    "E1": "Championship",
    "SP2": "Segunda División",
    "I2": "Serie B",
    "D2": "2. Bundesliga",
    "F2": "Ligue 2",
}

# Heuristic adjustment for a newly promoted club: they tend to score less
# and concede more once they step up a division compared to their previous
# tier's numbers. NOT fitted to this dataset — a chosen prior, not a
# calibrated estimate. Switched (by request) from a milder 0.85/1.15 to this
# harsher 0.55/1.48 pair, matching the reference document's values. Both
# sets are equally unsourced; this one simply applies a bigger discount/
# penalty. If you want this properly calibrated rather than chosen, it
# should be backtested against actual promoted teams' first-season
# top-flight numbers vs. their prior second-tier numbers.
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


def _current_season_year() -> int:
    now = dt.datetime.now()
    return now.year if now.month >= 7 else now.year - 1


def _recent_hist_season_codes(n: int = 3) -> list:
    """
    Football-Data.co.uk season codes like '2324' for the 2023/24 season.
    Previously hardcoded to ["2324", "2425", "2526"], which will silently go
    stale every year (never picks up the new season, keeps re-requesting an
    ever-more-outdated 3-season window). This computes the trailing n
    seasons ending at the CURRENT season, so it stays correct automatically.
    """
    start_year = _current_season_year()
    return [f"{str(y)[-2:]}{str(y + 1)[-2:]}" for y in range(start_year - n + 1, start_year + 1)]


# ====================================================================================
# 2. HISTORICAL DATA ENGINE (MULTI-SEASON ARCHIVE)
# ====================================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_historical_league_data(league_code: str = "E0") -> pd.DataFrame:
    """
    Downloads multi-season match data, goals, and closing odds directly 
    from Football-Data.co.uk CSVs without requiring an API key.

    BUGFIX: the URL previously pointed at "mmz4235", which is not a real path
    on football-data.co.uk (verified against their live site) — every request
    404'd, was swallowed by the except/continue below, and this function
    always silently returned an empty DataFrame. That means the multi-season
    historical engine has never actually run; the app has been falling
    through to the single-season standings fallback the whole time with no
    indication anything failed. The correct path segment is "mmz4281".
    """
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


# Known football-data.co.uk short-name quirks that don't match the official
# names returned by football-data.org / API-Football (e.g. "Manchester City
# FC" vs "Man City", "Atletico Madrid" vs "Ath Madrid"). Extend this table if
# other mismatches surface — particularly for leagues/seasons not checked here.
HIST_TEAM_ALIASES = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "West Ham United": "West Ham",
    "West Bromwich Albion": "West Brom",
    "Brighton & Hove Albion": "Brighton",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "AFC Bournemouth": "Bournemouth",
    "Atletico Madrid": "Ath Madrid",
    "Atlético Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Athletic Bilbao": "Ath Bilbao",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "Deportivo Alaves": "Alaves",
    "Rayo Vallecano": "Vallecano",
    "AC Milan": "Milan",
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "Hellas Verona": "Verona",
    "Borussia Dortmund": "Dortmund",
    "Borussia Monchengladbach": "M'gladbach",
    "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "1899 Hoffenheim": "Hoffenheim",
    "1. FC Koln": "FC Koln",
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Saint-Etienne": "St Etienne",
    "AS Saint-Etienne": "St Etienne",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "Coventry City": "Coventry",
}


def _normalize_hist_team_name(name: str) -> str:
    """Strips common club suffixes and applies known aliases so official
    names compare cleanly against football-data.co.uk's short-form names."""
    n = (name or "").strip()
    for suffix in (" FC", " CF", " AFC", " Football Club"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break
    return HIST_TEAM_ALIASES.get(n, n).strip()


def _match_hist_team_rows(hist_df: pd.DataFrame, column: str, team_name: str) -> pd.DataFrame:
    """
    Finds this team's rows in the historical CSV.

    BUGFIX: the previous approach matched on `team_name[:5]` as a raw
    substring, which had two failure modes: (1) silently matched NOTHING for
    teams whose official name doesn't share a prefix with football-data.co.uk's
    short form — e.g. "Manchester City FC"[:5] = "Manch", which is not even a
    substring of "Man City" — quietly falling back to a generic 1.0
    average-team rating for exactly the sort of big favourite this tool needs
    to price well; and (2) silently corrupted results by merging DIFFERENT
    clubs together when they share a 5-char prefix — e.g. "Real Madrid",
    "Real Sociedad", and "Real Betis" all start with "Real ".

    This now: (1) normalizes/aliases the name first, (2) tries an exact
    (case-insensitive) match, (3) falls back to a substring match in EITHER
    direction — the official name inside the CSV name (e.g. "Tottenham" in
    "Tottenham Hotspur"), or the CSV's shorter name inside the official one
    (e.g. "Coventry" inside "Coventry City" — the direction the original
    version of this function was missing, which is why Coventry City fell
    all the way through to the neutral 1.0/1.0 fallback instead of using
    their real Championship record) — but ONLY if it resolves to a SINGLE
    unique team either way. It never silently blends two clubs' results
    together, and returns empty (safe fallback to league-average) rather
    than a guess when the match is ambiguous or absent in both directions.
    """
    target = _normalize_hist_team_name(team_name)
    if not target:
        return hist_df.iloc[0:0]

    exact = hist_df[hist_df[column].str.casefold() == target.casefold()]
    if not exact.empty:
        return exact

    candidates = hist_df[column].dropna().unique()

    forward_hits = [c for c in candidates if target.casefold() in c.casefold()]
    if len(forward_hits) == 1:
        return hist_df[hist_df[column] == forward_hits[0]]

    reverse_hits = [c for c in candidates if c.casefold() in target.casefold()]
    if len(reverse_hits) == 1:
        return hist_df[hist_df[column] == reverse_hits[0]]

    return hist_df.iloc[0:0]


def _team_hist_side_stats(team_name: str, side: str, primary_df: pd.DataFrame,
                           secondary_df: Optional[pd.DataFrame],
                           league_home_avg_g: float, league_away_avg_g: float) -> dict:
    """
    A team's goals-for / goals-against averages for a given side ('home' or
    'away'), trying the top-flight dataset first. If the team has ZERO
    matches there in the scanned window (e.g. newly promoted, no top-flight
    history yet), falls back to the second-tier dataset with a promotion
    adjustment — NOT a naming-mismatch case, a genuinely different source.
    Only truly defaults to a neutral 1.0-equivalent if found in neither.

    When the second-tier fallback is used, this also returns the RAW
    (pre-adjustment) attack/defense ratios relative to the second-tier's OWN
    average — e.g. Coventry's rating relative to the Championship average,
    before any cross-tier translation — so the UI can show both stages of
    the calculation transparently instead of only the final blended number.
    """
    team_col = 'HomeTeam' if side == 'home' else 'AwayTeam'
    for_col = 'FTHG' if side == 'home' else 'FTAG'
    against_col = 'FTAG' if side == 'home' else 'FTHG'
    # Same column -> league-average mapping used consistently for both the
    # primary and secondary dataset, so "attack"/"defense" always mean
    # "this column's rate, relative to this column's league average" no
    # matter which tier of data actually produced the raw goals.
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


def calculate_historical_lambdas(
    home_team: str, away_team: str, hist_df: pd.DataFrame,
    secondary_df: Optional[pd.DataFrame] = None,
) -> Tuple[float, float, dict, dict, float, float]:
    """Calculates team attack/defense relative strength parameters from historical data.
    Returns (lam_home, lam_away, home_info, away_info, league_home_avg, league_away_avg).
    home_info/away_info now also carry "attack"/"defense" ratios (relative to
    1.00 league average) alongside the raw goals — these are exactly the α/β
    values shown in the Team Rating Engine Controls panel, and the league_home_avg/
    league_away_avg returned here are the SAME baseline used to compute them,
    so an override can be recombined with `league_avg * attack * defense`
    and get an answer consistent with the auto-computed one."""
    empty_keys = {"attack": 1.0, "defense": 1.0, "raw_attack": None, "raw_defense": None,
                  "raw_goals_for": None, "raw_goals_against": None, "goals_for": None, "goals_against": None}
    if hist_df.empty:
        empty_info = {**empty_keys, "tier": "no historical data", "n": 0}
        return 1.55, 1.20, empty_info, empty_info, 1.55, 1.20

    league_home_avg_g = max(hist_df['FTHG'].mean(), 1.0)
    league_away_avg_g = max(hist_df['FTAG'].mean(), 1.0)

    home_info = _team_hist_side_stats(home_team, 'home', hist_df, secondary_df, league_home_avg_g, league_away_avg_g)
    away_info = _team_hist_side_stats(away_team, 'away', hist_df, secondary_df, league_home_avg_g, league_away_avg_g)

    lam_home = league_home_avg_g * home_info["attack"] * away_info["defense"]
    lam_away = league_away_avg_g * away_info["attack"] * home_info["defense"]

    return (round(float(lam_home), 3), round(float(lam_away), 3), home_info, away_info,
            round(float(league_home_avg_g), 3), round(float(league_away_avg_g), 3))


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
        # games_rated simulates squad rotation: most starters featured in most
        # of the last n_games, a few rotation players featured in fewer/none.
        games_rated = min(n_games, max(0, n_games - random.choice([0, 0, 0, 1, 2, n_games])))
        rating = round(np.random.normal(7.0, 0.5), 2) if games_rated > 0 else None
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
def fetch_player_recent_ratings(team_id: int, n_games: int = 5) -> dict:
    """
    Real per-match player ratings from the team's last n_games finished
    fixtures (API-Football /fixtures + /fixtures/players), NOT the season
    aggregate. This is what actually lets you see who's in form: a player's
    season rating can be a solid 7.1 while they've been poor (or benched
    entirely) the last 3 matches, and vice versa.

    Returns {player_name: {"avg_rating": float, "games_rated": int}} where
    games_rated is how many of the last n_games this specific player actually
    featured in and got a rating — 0 means they've dropped out of the side
    recently, which is itself a form signal worth showing, not hiding.
    """
    if not API_FOOTBALL_KEY:
        return {}
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r_fx = requests.get(
            f"https://{API_FOOTBALL_HOST}/fixtures",
            headers=headers, params={"team": team_id, "last": n_games}, timeout=8,
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
                    # Only count matches the player actually appeared in —
                    # API-Football sometimes lists unused subs with no rating.
                    if name and rating_raw and minutes:
                        ratings.setdefault(name, []).append(float(rating_raw))

        return {
            name: {"avg_rating": round(sum(vals) / len(vals), 2), "games_rated": len(vals)}
            for name, vals in ratings.items()
        }
    except Exception:
        return {}


def _parse_players_stats_page(response_items: list, pos_map: dict, recent_ratings: dict,
                               fallback_xg90: dict, fallback_xa90: dict) -> list:
    """Turns one page of API-Football /players response items into squad rows.
    Shared by both the current-season and previous-season attempts below so
    the parsing logic only exists once."""
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

        recent = recent_ratings.get(name)
        if recent:
            rating, games_rated = recent["avg_rating"], recent["games_rated"]
        else:
            season_rating_raw = games.get("rating")
            rating = round(float(season_rating_raw), 2) if season_rating_raw else 6.8
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


def _fetch_players_stats_all_pages(team_id: int, season: int, headers: dict) -> list:
    """Pages through API-Football's /players endpoint for one team+season,
    capped at 3 pages to respect free-tier rate limits. Returns the raw
    response items (not yet parsed into rows) so the caller can decide
    whether the season actually had any data before committing to it."""
    items, page, total_pages = [], 1, 1
    while page <= total_pages and page <= 3:
        r_stats = requests.get(
            f"https://{API_FOOTBALL_HOST}/players",
            headers=headers, params={"team": team_id, "season": season, "page": page}, timeout=8,
        )
        r_stats.raise_for_status()
        payload = r_stats.json()
        total_pages = payload.get("paging", {}).get("total", 1)
        items.extend(payload.get("response", []))
        page += 1
    return items


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_squad(team_name: str, seed_offset: int = 0, n_games: int = 5) -> pd.DataFrame:
    """
    BUGFIX: this previously made ONE call to /players?season=<current> and,
    if it came back empty, silently fell all the way to the fake-name mock
    squad. That endpoint is stats-driven — it only returns players who've
    accumulated match stats for that team in that specific season — so it
    returns empty for EVERY club during preseason (no matches played yet in
    the new season) or for a newly promoted club with zero top-flight
    appearances this season. Both are real, common situations, not edge
    cases, and the old code treated them identically to "API key invalid."

    Now tries three tiers before giving up, in order:
      1. Current season's real per-player stats (best case).
      2. Previous season's real per-player stats, if the current season has
         no data yet (preseason, or newly promoted with zero stats so far)
         — same players, still-useful per-90 rates, just one season stale.
      3. The roster-only /players/squads endpoint (names + positions,
         no season dependency at all) with position-average fallback rates,
         if even the previous season came back empty.
    Only falls to the fully-synthetic mock squad if the team can't be found
    at all. Every returned DataFrame carries a "data_source" column so the
    UI can show which tier actually produced it — no more silent fallback.
    """
    if API_FOOTBALL_KEY:
        try:
            headers = {"x-apisports-key": API_FOOTBALL_KEY}
            clean_name = team_name.replace(" FC", "").replace(" AFC", "").replace(" Football Club", "")
            r_team = requests.get(
                f"https://{API_FOOTBALL_HOST}/teams",
                headers=headers, params={"search": clean_name}, timeout=6,
            )
            r_team.raise_for_status()
            team_res = r_team.json().get("response", [])

            if team_res:
                team_id = team_res[0]["team"]["id"]
                season = _current_season_year()
                pos_map = {"Goalkeeper": "GK", "Defender": "DF", "Midfielder": "MF", "Attacker": "FW"}
                fallback_xg90 = {"FW": 0.30, "MF": 0.10, "DF": 0.02, "GK": 0.0}
                fallback_xa90 = {"FW": 0.15, "MF": 0.16, "DF": 0.03, "GK": 0.0}

                # Recent per-match ratings come from /fixtures?last=N, which
                # has no "season" parameter — it just returns the last N
                # played matches regardless of season boundary, so this
                # tier keeps working correctly through the preseason gap.
                recent_ratings = fetch_player_recent_ratings(team_id, n_games)

                # --- Tier 1: current season stats ---
                items = _fetch_players_stats_all_pages(team_id, season, headers)
                data_source = f"API-Football player stats ({season} season)"

                # --- Tier 2: previous season stats, if current is empty ---
                if not items:
                    items = _fetch_players_stats_all_pages(team_id, season - 1, headers)
                    data_source = f"API-Football player stats ({season - 1} season — {season} not yet underway)"

                rows = _parse_players_stats_page(items, pos_map, recent_ratings, fallback_xg90, fallback_xa90)

                # --- Tier 3: roster-only endpoint, if both seasons are empty ---
                if not rows:
                    r_squad = requests.get(
                        f"https://{API_FOOTBALL_HOST}/players/squads",
                        headers=headers, params={"team": team_id}, timeout=8,
                    )
                    r_squad.raise_for_status()
                    squad_res = r_squad.json().get("response", [])
                    if squad_res and "players" in squad_res[0]:
                        data_source = "API-Football roster (no season stats available yet)"
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
    df["data_source"] = "Mock (API-Football unavailable or team not found)"
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
            team_id = next((row["team"]["id"] for row in total_table if row["team"]["name"] == team_name), None)

            if team_id:
                r_m = requests.get(
                    f"{FOOTBALL_DATA_BASE}/teams/{team_id}/matches",
                    headers=headers, params={"status": "FINISHED", "limit": 5}, timeout=8,
                )
                r_m.raise_for_status()
                matches = r_m.json().get("matches", [])[-5:]
                rows = []
                for idx, m in enumerate(matches):
                    is_home = m["homeTeam"]["name"] == team_name
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
    if THE_ODDS_API_KEY:
        try:
            r = requests.get(f"{ODDS_API_BASE}/sports/{sport_key}/odds", params={"apiKey": THE_ODDS_API_KEY, "regions": "uk,eu", "markets": "h2h,totals", "oddsFormat": "decimal"}, timeout=6)
            r.raise_for_status()
            events = r.json()
            for ev in events:
                if home_team.lower() in ev["home_team"].lower():
                    book = ev["bookmakers"][0]
                    h2h = next(m for m in book["markets"] if m["key"] == "h2h")
                    prices = {o["name"]: o["price"] for o in h2h["outcomes"]}
                    return {
                        "1X2": {"home": prices.get(ev["home_team"]), "draw": prices.get("Draw"), "away": prices.get(ev["away_team"])},
                        "source": f"The Odds API ({book['title']}, {sport_key})"
                    }
        except Exception:
            pass
    return _mock_odds(home_team, away_team)

# ====================================================================================
# 5. QUANTITATIVE MODELING ENGINE
# ====================================================================================
def calculate_team_base_lambdas(
    home_team: str, away_team: str, team_stats: pd.DataFrame
) -> Tuple[float, float, dict, dict, float, float]:
    """Fallback lambda calculator from single-season standings data.
    Returns the same shape as calculate_historical_lambdas — (lam_home,
    lam_away, home_info, away_info, league_home_avg, league_away_avg) — so
    the Team Rating Engine Controls panel works identically regardless of which
    data source actually produced the auto values."""
    league_home_avg = max(team_stats['home_xG_for'].mean(), 1.0)
    league_away_avg = max(team_stats['away_xG_for'].mean(), 1.0)
    
    h_stat = team_stats.loc[team_stats['team'] == home_team]
    a_stat = team_stats.loc[team_stats['team'] == away_team]
    
    home_att = (h_stat['home_xG_for'].values[0] if not h_stat.empty else 1.55) / league_home_avg
    away_def = (a_stat['away_xG_against'].values[0] if not a_stat.empty else 1.20) / league_home_avg
    
    away_att = (a_stat['away_xG_for'].values[0] if not a_stat.empty else 1.20) / league_away_avg
    # BUGFIX (regression): this was dividing by league_home_avg. home_def
    # measures a team's home defensive record, which should be benchmarked
    # against what away teams typically score across the league — i.e.
    # league_away_avg, the same baseline away_att uses above. Dividing by
    # league_home_avg instead systematically mis-scaled every team's defense
    # term, and since fetch_historical_league_data() was silently failing
    # (see bugfix above), THIS function has been the one actually running
    # for every single match — so this bug has been live in production.
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


def form_multiplier(form_df: pd.DataFrame) -> float:
    weights = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    df = form_df.sort_values("matches_ago", ascending=False).reset_index(drop=True)
    if len(df) < 5:
        weights = weights[-len(df):] / weights[-len(df):].sum()
    match_scores = 0.4 * df["goals_for"] + 0.6 * df["xG_for"]
    weighted_score = float((match_scores * weights).sum())
    return round(weighted_score / 1.45, 3)


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
    form_mult: float
    form_rating: float
    weather_mult: float
    rest_mult: float
    squad_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def lambda_final(self) -> float:
        val = self.base_lambda * self.piv_multiplier * self.form_mult * self.weather_mult * self.rest_mult
        return round(max(val, 0.05), 3)


def dixon_coles_tau(x: int, y: int, lam_home: float, lam_away: float, rho: float = -0.06) -> float:
    if x == 0 and y == 0: return 1 - lam_home * lam_away * rho
    elif x == 0 and y == 1: return 1 + lam_home * rho
    elif x == 1 and y == 0: return 1 + lam_away * rho
    elif x == 1 and y == 1: return 1 - rho
    return 1.0


def scoreline_matrix(lam_home: float, lam_away: float, max_goals: int = 7) -> np.ndarray:
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away) * dixon_coles_tau(i, j, lam_home, lam_away)
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
st.sidebar.markdown("🟢 Historical CSV Engine (Football-Data.co.uk, no key needed)")
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

st.sidebar.divider()
st.sidebar.subheader("Trading Parameters")
target_margin = st.sidebar.slider("Target Model Margin (%)", 2.0, 8.0, 5.0, 0.5)
kelly_fraction = st.sidebar.slider("Kelly Fractional Sizing", 0.1, 1.0, 0.25, 0.05, help="0.25 = Quarter Kelly (Recommended)")

st.sidebar.subheader("Rest Differential (Days Rest)")
c_r1, c_r2 = st.sidebar.columns(2)
with c_r1:
    home_rest = st.number_input(f"{home_team}", min_value=1, max_value=14, value=7)
with c_r2:
    away_rest = st.number_input(f"{away_team}", min_value=1, max_value=14, value=7)

st.sidebar.divider()
st.sidebar.subheader("Player Availability")
rating_window = st.sidebar.slider(
    "Player Form Window (games)", 3, 5, 5,
    help="avg_rating below is each player's average rating over their last N matches (not season average) — the number to check for who's actually in form right now."
)
home_squad_raw = fetch_squad(home_team, seed_offset=1, n_games=rating_window)
away_squad_raw = fetch_squad(away_team, seed_offset=2, n_games=rating_window)

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

# Dual-layer Lambda Computation: Prefer multi-season historical CSV dataset, fallback to standings
hist_code = HIST_LEAGUE_MAP.get(league, "E0")
hist_df = fetch_historical_league_data(hist_code)
secondary_code = HIST_SECONDARY_LEAGUE_MAP.get(hist_code)
secondary_hist_df = fetch_historical_league_data(secondary_code) if secondary_code else pd.DataFrame()

# Diagnostics so a repeat of the mmz4235/mmz4281 URL bug — where the historical
# engine failed 100% silently for weeks — can never hide again. Shown in the UI below.
lambda_source = "mock"
home_tier_info = away_tier_info = {"tier": "n/a", "n": 0, "attack": 1.0, "defense": 1.0,
                                     "goals_for": None, "goals_against": None}
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

# --- Team Rating Engine Controls (Automated / Custom Manual Override) ---
# Adopts the radio-toggle UX and metric-column layout from the reference
# snippet, but keeps OUR backend: alias-matched team lookup, per-league
# dynamic baselines (league_home_avg_used/league_away_avg_used), and
# auto-detected promotion handling via home_tier_info/away_tier_info — none
# of that gets replaced. Deliberately does NOT call an LLM to estimate a
# rating when data is missing; the neutral 1.0/1.0 fallback already in
# home_tier_info/away_tier_info says "we don't know" honestly rather than
# injecting an unsourced home-favorite prior.
st.sidebar.divider()
st.sidebar.subheader("⚙️ Team Rating Engine Controls")

rating_mode = st.sidebar.radio(
    "Rating Mode",
    options=["Automated (Data-Driven)", "Custom Manual Override"],
    index=0,
    help="Automated uses whichever source produced the values above (historical CSV, live standings, or a labeled fallback). Custom lets you type in your own attack/defense ratings, pre-filled with the automated numbers as a starting point.",
)

def _overall_rating(attack: float, defense: float) -> float:
    return round(min(max(50 + (attack - defense) * 25, 0), 100), 1)

st.sidebar.caption(
    f"{home_team}: {home_tier_info['tier']}" + (f" ({home_tier_info['n']} matches)" if home_tier_info.get('n') else "")
    + f" · {away_team}: {away_tier_info['tier']}" + (f" ({away_tier_info['n']} matches)" if away_tier_info.get('n') else "")
)

if rating_mode == "Automated (Data-Driven)":
    st.sidebar.success("🟢 Ratings auto-synced from the data source above")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(f"{home_team} α (Att)", home_tier_info["attack"])
        st.metric(f"{home_team} β (Def)", home_tier_info["defense"])
        st.metric(f"{home_team} Overall", f"{_overall_rating(home_tier_info['attack'], home_tier_info['defense'])}/100")
    with col2:
        st.metric(f"{away_team} α (Att)", away_tier_info["attack"])
        st.metric(f"{away_team} β (Def)", away_tier_info["defense"])
        st.metric(f"{away_team} Overall", f"{_overall_rating(away_tier_info['attack'], away_tier_info['defense'])}/100")

    home_attack_eff, home_defense_eff = home_tier_info["attack"], home_tier_info["defense"]
    away_attack_eff, away_defense_eff = away_tier_info["attack"], away_tier_info["defense"]
    ratings_overridden = False
else:
    st.sidebar.warning("✏️ Custom override active — recombined using the same league_avg × attack × defense formula as automated mode")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.markdown(f"**{home_team}**")
        home_attack_eff = st.number_input("Attack (α)", 0.10, 3.00, value=float(home_tier_info["attack"]), step=0.05, key="home_attack_custom")
        home_defense_eff = st.number_input("Defense (β)", 0.10, 3.00, value=float(home_tier_info["defense"]), step=0.05, key="home_defense_custom")
        st.caption(f"Overall: **{_overall_rating(home_attack_eff, home_defense_eff)}/100**")
    with c2:
        st.markdown(f"**{away_team}**")
        away_attack_eff = st.number_input("Attack (α)", 0.10, 3.00, value=float(away_tier_info["attack"]), step=0.05, key="away_attack_custom")
        away_defense_eff = st.number_input("Defense (β)", 0.10, 3.00, value=float(away_tier_info["defense"]), step=0.05, key="away_defense_custom")
        st.caption(f"Overall: **{_overall_rating(away_attack_eff, away_defense_eff)}/100**")
    ratings_overridden = True

if ratings_overridden:
    base_lam_home = round(league_home_avg_used * home_attack_eff * away_defense_eff, 3)
    base_lam_away = round(league_away_avg_used * away_attack_eff * home_defense_eff, 3)
    lambda_source += " + custom manual override"

home_form_df = fetch_team_form(home_team, league)
away_form_df = fetch_team_form(away_team, league)

home_piv_mult, home_squad = player_impact_score(home_squad_raw, home_active)
away_piv_mult, away_squad = player_impact_score(away_squad_raw, away_active)

home_model = TeamModelInputs(
    name=home_team, base_lambda=base_lam_home, piv_multiplier=home_piv_mult,
    form_mult=form_multiplier(home_form_df), form_rating=team_form_rating_0_100(home_form_df),
    weather_mult=w_mult, rest_mult=rest_modifier(home_rest), squad_table=home_squad,
)

away_model = TeamModelInputs(
    name=away_team, base_lambda=base_lam_away, piv_multiplier=away_piv_mult,
    form_mult=form_multiplier(away_form_df), form_rating=team_form_rating_0_100(away_form_df),
    weather_mult=w_mult, rest_mult=rest_modifier(away_rest), squad_table=away_squad,
)

lam_home, lam_away = home_model.lambda_final, away_model.lambda_final
matrix = scoreline_matrix(lam_home, lam_away, max_goals=7)
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
            if "⚠️" in _msg:
                st.warning(_msg)
            else:
                st.info(_msg)

tier_summary = ""
if "historical CSV" in lambda_source:
    tier_summary = f" · {home_team}: {home_tier_info['tier']} ({home_tier_info['n']} matches) · {away_team}: {away_tier_info['tier']} ({away_tier_info['n']} matches)"
st.caption(f"λ base source: **{lambda_source}**{tier_summary}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("λ Home (Final xG)", lam_home)
    st.caption(f"Base {base_lam_home} → PIV×{home_piv_mult:.2f} · Rest×{home_model.rest_mult:.2f}")
with c2:
    st.metric("λ Away (Final xG)", lam_away)
    st.caption(f"Base {base_lam_away} → PIV×{away_piv_mult:.2f} · Rest×{away_model.rest_mult:.2f}")
with c3:
    st.metric(f"{home_team} Form", f"{home_model.form_rating}/100")
    st.metric(f"{away_team} Form", f"{away_model.form_rating}/100")
with c4:
    st.metric("🌡️ Temp", f"{weather['temperature_c']}°C")
    st.metric("💨 Wind", f"{weather['wind_speed_kmh']} km/h")

# ---- Rating Calculation Breakdown ----
# A transparent, formula-level view of how α/β and λ were actually derived —
# every number here is one we computed above, not re-estimated for display.
with st.expander("📊 Rating Calculation Breakdown"):
    primary_league_name = league
    secondary_league_name = SECOND_TIER_DISPLAY_NAMES.get(secondary_code, secondary_code or "n/a")

    summary_rows = []
    for team_name, info in ((home_team, home_tier_info), (away_team, away_tier_info)):
        if info["tier"] == "second-tier (promotion-adjusted)":
            summary_rows.append({
                "Team": team_name, "League Context": f"{secondary_league_name} Baseline",
                "Attack Rating (α)": info["raw_attack"], "Defense Rating (β)": info["raw_defense"],
            })
            summary_rows.append({
                "Team": team_name, "League Context": f"{primary_league_name} Adjusted",
                "Attack Rating (α)": info["attack"], "Defense Rating (β)": info["defense"],
            })
        elif info["tier"] == "top-flight":
            summary_rows.append({
                "Team": team_name, "League Context": f"{primary_league_name} Baseline",
                "Attack Rating (α)": info["attack"], "Defense Rating (β)": info["defense"],
            })
        else:
            summary_rows.append({
                "Team": team_name, "League Context": f"League Average ({info['tier']})",
                "Attack Rating (α)": info["attack"], "Defense Rating (β)": info["defense"],
            })

    st.markdown("**Rating Summary Table**")
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    st.markdown("**Key Metric Explanations**")
    for team_name, info in ((home_team, home_tier_info), (away_team, away_tier_info)):
        if info["tier"] == "second-tier (promotion-adjusted)":
            att_pct = round((info["raw_attack"] - 1) * 100, 1)
            def_pct = round((1 - info["raw_defense"]) * 100, 1)
            st.markdown(
                f"- **{team_name} ({secondary_league_name} raw)** — Attack (α={info['raw_attack']}): "
                f"{'generates ' + str(abs(att_pct)) + '% more' if att_pct >= 0 else 'generates ' + str(abs(att_pct)) + '% less'} "
                f"xG than a standard {secondary_league_name} team. "
                f"Defense (β={info['raw_defense']}): concedes "
                f"{'fewer' if def_pct >= 0 else 'more'} goals than average ({abs(def_pct)}% {'below' if def_pct>=0 else 'above'} the {secondary_league_name} mean)."
            )
            st.markdown(
                f"- **{team_name} ({primary_league_name} adjusted)** — applied cross-tier translation "
                f"(attack ×{PROMOTION_ATTACK_DISCOUNT}, defense ×{PROMOTION_DEFENSE_PENALTY}, "
                f"a heuristic prior, not fitted to this data): "
                f"α = {info['raw_attack']} × {PROMOTION_ATTACK_DISCOUNT} → **{info['attack']}**, "
                f"β = {info['raw_defense']} × {PROMOTION_DEFENSE_PENALTY} → **{info['defense']}**."
            )
        elif info["tier"] == "top-flight":
            st.markdown(
                f"- **{team_name}** — Attack (α={info['attack']}): "
                f"{'generates' if info['attack']>=1 else 'produces'} {abs(round((info['attack']-1)*100,1))}% "
                f"{'more' if info['attack']>=1 else 'less'} xG than a standard {primary_league_name} team. "
                f"Defense (β={info['defense']}): concedes {abs(round((info['defense']-1)*100,1))}% "
                f"{'less' if info['defense']<1 else 'more'} than the {primary_league_name} average "
                f"(across {info['n']} matches)."
            )
        else:
            st.markdown(f"- **{team_name}** — no usable match data in either tier; using the neutral league-average fallback (α=1.0, β=1.0).")

    st.markdown("**How These Metrics Combine in the Poisson Match Model**")
    st.markdown(
        f"Using this fixture's actual baselines (Home Avg Goals = {league_home_avg_used}, "
        f"Away Avg Goals = {league_away_avg_used}):"
    )
    st.latex(
        rf"\lambda_H = {league_home_avg_used} \times {home_tier_info['attack']} \times {away_tier_info['defense']} "
        rf"\approx {round(league_home_avg_used * home_tier_info['attack'] * away_tier_info['defense'], 3)}\ \text{{xG}}"
    )
    st.latex(
        rf"\lambda_A = {league_away_avg_used} \times {away_tier_info['attack']} \times {home_tier_info['defense']} "
        rf"\approx {round(league_away_avg_used * away_tier_info['attack'] * home_tier_info['defense'], 3)}\ \text{{xG}}"
    )
    st.caption(
        "These are the BASE λ values (before PIV, form, weather, and rest multipliers are applied — see the "
        "caption under the λ cards above for the fully adjusted figure). If you've applied a Custom Manual "
        "Override above, this section still reflects the AUTOMATED numbers, so you can see what changed."
    )

st.divider()

# ---- Squad & Player Form Section ----
st.subheader("👥 Player Form & Impact Penalties")
st.caption(f"avg_rating and games_rated reflect each player's actual performance over their last {rating_window} matches "
           f"(not season average) — games_rated = 0 means they haven't featured in that recent window.")
home_squad_source = home_squad["data_source"].iloc[0] if "data_source" in home_squad.columns and len(home_squad) else "unknown"
away_squad_source = away_squad["data_source"].iloc[0] if "data_source" in away_squad.columns and len(away_squad) else "unknown"
if "Mock" in home_squad_source or "Mock" in away_squad_source:
    st.warning(f"⚠️ Squad data source — {home_team}: **{home_squad_source}** · {away_team}: **{away_squad_source}**")
else:
    st.caption(f"Squad data source — {home_team}: **{home_squad_source}** · {away_team}: **{away_squad_source}**")
pc1, pc2 = st.columns(2)
display_cols = ["player", "position", "avg_rating", "games_rated", "xG90", "xA90", "status", "absence_penalty_%"]
with pc1:
    st.markdown(f"**{home_team} Lineup & Stats**")
    st.dataframe(
        home_squad[display_cols].sort_values("avg_rating", ascending=False, na_position="last"),
        hide_index=True,
    )
with pc2:
    st.markdown(f"**{away_team} Lineup & Stats**")
    st.dataframe(
        away_squad[display_cols].sort_values("avg_rating", ascending=False, na_position="last"),
        hide_index=True,
    )

st.divider()

# ---- Quantitative Odds & Devigged Market Comparison ----
st.subheader("💰 Odds Engine, Devigged Market & Kelly Staking")

book_1x2 = market_odds.get("1X2", {})
devig_1x2 = devig_proportional(book_1x2.get("home", 0), book_1x2.get("draw", 0), book_1x2.get("away", 0))
devig_ou = devig_two_way(market_odds.get("over_2_5"), market_odds.get("under_2_5"))
devig_btts = devig_two_way(market_odds.get("btts_yes"), market_odds.get("btts_no"))

def build_trade_row(market_label, model_prob, model_odd, book_odd, fair_mkt_prob):
    if not book_odd or book_odd <= 1.0:
        return {"Market": market_label, "Model Odds": model_odd, "Bookmaker Odds": "N/A", "Edge (pp)": 0, "Kelly Stake": "0.0%", "Signal": "N/A"}

    devigged = True
    if fair_mkt_prob is None:
        fair_mkt_prob = 1 / book_odd
        devigged = False

    edge_pp = round((model_prob - fair_mkt_prob) * 100, 2)
    stake_pct = kelly_stake(model_prob, book_odd, kelly_fraction)

    if not devigged:
        signal = "⚠️ NO DEVIG"
    else:
        signal = "🟢 VALUE" if edge_pp >= 2.0 and stake_pct > 0 else ("🔴 OVERPRICED" if edge_pp <= -2.0 else "⚪ FAIR")

    return {
        "Market": market_label,
        "Model Prob %": round(model_prob * 100, 1),
        "Model Odds": model_odd,
        "Bookmaker Odds": book_odd,
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

comparison_df = pd.DataFrame(trade_rows)
st.dataframe(comparison_df, hide_index=True)
st.caption("Edge (pp) = Model Probability − Devigged (Fair) Market Probability. Fractional Kelly sizing calculated using specified bankroll multiplier.")

st.divider()

# ---- Heatmap Section ----
st.subheader("🔥 Scoreline Probability Matrix Heatmap")

heat_labels = list(range(matrix.shape[0]))
fig = go.Figure(data=go.Heatmap(
    z=matrix * 100,
    x=[f"{away_team} {g}" for g in heat_labels],
    y=[f"{home_team} {g}" for g in heat_labels],
    colorscale="YlOrRd",
    text=np.round(matrix * 100, 1),
    texttemplate="%{text}%",
    textfont={"size": 11},
    hoverongaps=False
))

fig.update_layout(
    title=f"Scoreline Probability Distribution (%) — Combined xG Expectancy: {lam_home + lam_away:.2f}",
    xaxis_title=f"Away Goals ({away_team})",
    yaxis_title=f"Home Goals ({home_team})",
    height=500,
    margin=dict(l=40, r=40, t=50, b=40)
)

st.plotly_chart(fig)

st.divider()
st.subheader("📥 Export Analysis")

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Odds & Edge Analysis (CSV)",
    data=convert_df_to_csv(comparison_df),
    file_name=f"{home_team}_vs_{away_team}_odds_analysis.csv",
    mime="text/csv",
)
