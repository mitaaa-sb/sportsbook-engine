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

_STADIUM_CITIES = {
    "Premier League": ("London", 51.5072, -0.1276),
    "La Liga": ("Madrid", 40.4168, -3.7038),
    "Serie A": ("Milan", 45.4642, 9.1900),
    "Bundesliga": ("Munich", 48.1351, 11.5820),
    "Ligue 1": ("Paris", 48.8566, 2.3522),
}

# BUGFIX: fetch_market_odds previously hardcoded "soccer_epl" for every league.
# The Odds API needs the correct sport key per competition or it silently
# queries the wrong league's odds (and misses every non-EPL match).
ODDS_API_SPORT_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
}

# Shared team pools per league — used by both mock fixtures AND mock season
# stats so the two always stay in sync. Previously these were two separate
# hardcoded lists (in _mock_fixtures and _mock_season_stats) that had drifted:
# _mock_season_stats only covered 14 EPL/marquee teams while _mock_fixtures
# pulled from ~30 teams across 5 leagues, so most non-EPL (and several EPL)
# matchups silently fell back to a generic default lambda base instead of a
# team-specific one. Defining the pool once and reusing it everywhere fixes
# that gap and guarantees every fixture-eligible team has real mock stats.
LEAGUE_TEAM_POOLS = {
    "Premier League": ["Arsenal", "Man City", "Liverpool", "Chelsea", "Aston Villa", "Tottenham"],
    "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Real Sociedad", "Sevilla", "Villarreal"],
    "Serie A": ["Inter", "Juventus", "AC Milan", "Napoli", "Roma", "Atalanta"],
    "Bundesliga": ["Bayern Munich", "Dortmund", "RB Leipzig", "Leverkusen", "Union Berlin", "Freiburg"],
    "Ligue 1": ["PSG", "Monaco", "Marseille", "Lyon", "Lille", "Nice"],
}


def _current_season_year() -> int:
    """European football seasons span two calendar years; API-Football keys
    stats by the year the season STARTED (e.g. 2025 for the 2025/26 season)."""
    now = dt.datetime.now()
    return now.year if now.month >= 7 else now.year - 1


# ====================================================================================
# 2. MOCK DATA GENERATORS (DEMO FALLBACKS)
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


def _mock_squad(team: str, seed_offset: int = 0) -> pd.DataFrame:
    random.seed((hash(team) + seed_offset) % 10000)
    positions = ["GK", "DF", "DF", "DF", "DF", "MF", "MF", "MF", "FW", "FW", "FW"]
    names_pool = ["Silva", "Rodrigues", "Kovac", "Muller", "Dubois", "Novak", "Andersen", "Fernandez", "Costa", "Brandt", "Diaz"]
    rows = []
    for i, pos in enumerate(positions):
        name = f"{random.choice(names_pool)} {chr(65 + i)}"
        minutes = random.randint(600, 2500)
        xg90 = round(max(0, np.random.normal(0.38 if pos == "FW" else 0.12 if pos == "MF" else 0.02, 0.10)), 3)
        xa90 = round(max(0, np.random.normal(0.22 if pos in ("FW", "MF") else 0.03, 0.08)), 3)
        rating = round(np.random.normal(7.0, 0.4), 2)
        key_passes90 = round(max(0, np.random.normal(1.5 if pos == "MF" else 0.6, 0.5)), 2)
        rows.append({
            "player": name, "position": pos, "minutes": minutes,
            "xG90": xg90, "xA90": xa90, "key_passes90": key_passes90,
            "avg_rating": rating, "status": "Active",
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
# 3. LIVE DATA LAYER
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
    """
    Team-level attack/defense inputs consumed by calculate_team_base_lambdas.

    ROOT-CAUSE BUGFIX (this is why the model showed Arsenal at ~2.91 while
    every bookmaker had them at ~1.16): this function previously returned
    `_mock_season_stats(league_name)` UNCONDITIONALLY — there was no live
    branch at all, regardless of whether FOOTBALL_DATA_KEY was configured.
    That means every team's modelled attack/defense strength was random noise
    (np.random.uniform per team) with zero relationship to their real quality.
    The Odds API side of the app was showing real bookmaker consensus, but the
    statistical model's own inputs were completely disconnected from reality
    — so a big favorite could easily come out underpriced by the model purely
    by chance of the random seed.

    Now, when FOOTBALL_DATA_KEY is set, this pulls the LIVE competition
    standings (home/away split) from football-data.org and derives goals-for
    / goals-against per game as a real, direction-correct proxy for attack and
    defense strength. Note: this is a goals-based proxy, not true xG — football-
    data.org's free tier doesn't expose expected goals. If you want genuine xG
    here, wire in soccerdata's Understat reader (see comment at the bottom of
    this function) in place of the goals proxy.
    """
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
                        continue  # no matches played yet in that split — skip rather than divide by zero
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
        # --------------------------------------------------------------
        # >>> TRUE xG FETCHER CONFIGURATION POINT <<<
        # Replace/augment the block above with soccerdata's Understat reader
        # for real expected-goals data instead of the goals-based proxy:
        #   import soccerdata as sd
        #   understat = sd.Understat(leagues=UNDERSTAT_LEAGUE_MAP[league_name], seasons=_current_season_year())
        #   team_stats = understat.read_team_match_stats()  # has xG/xGA per match
        # --------------------------------------------------------------
    return _mock_season_stats(league_name)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_squad(team_name: str, seed_offset: int = 0) -> pd.DataFrame:
    """
    Live squad + per-player season stats from API-Football.

    BUGFIX: previously fetched the real squad list (names/positions) but then
    assigned every player a FIXED constant xG90/xA90 purely from their
    position (e.g. every midfielder got exactly 0.14/0.22) — so PIV could
    never tell a star player apart from a benchwarmer in the same position,
    and the "player impact" driving lambda wasn't actually reflecting who's
    good. This now calls API-Football's `/players` endpoint (team + season)
    to pull each player's real season goals/assists/minutes and derives
    per-90 rates as a goals-based proxy for xG90/xA90 (API-Football's free
    tier doesn't expose true xG — see the fetcher configuration comment
    below for wiring in real xG via soccerdata/Understat instead).
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
                # Position-average fallback ONLY for players with 0 minutes
                # logged yet this season (new signings, kids coming through) —
                # not applied blanket to the whole squad like before.
                fallback_xg90 = {"FW": 0.30, "MF": 0.10, "DF": 0.02, "GK": 0.0}
                fallback_xa90 = {"FW": 0.15, "MF": 0.16, "DF": 0.03, "GK": 0.0}

                rows, page, total_pages = [], 1, 1
                while page <= total_pages and page <= 3:  # cap pages to respect free-tier rate limits
                    # --------------------------------------------------------------
                    # >>> PLAYER STAT FETCHER CONFIGURATION POINT <<<
                    # This is the real live source for per-player season stats.
                    # --------------------------------------------------------------
                    r_stats = requests.get(
                        f"https://{API_FOOTBALL_HOST}/players",
                        headers=headers, params={"team": team_id, "season": season, "page": page}, timeout=8,
                    )
                    r_stats.raise_for_status()
                    payload = r_stats.json()
                    total_pages = payload.get("paging", {}).get("total", 1)

                    for p in payload.get("response", []):
                        info = p.get("player", {})
                        stat = (p.get("statistics") or [{}])[0]
                        games = stat.get("games", {}) or {}
                        goals = stat.get("goals", {}) or {}
                        minutes = games.get("minutes") or 0
                        pos = pos_map.get(games.get("position"), "MF")
                        goals_total = goals.get("total") or 0
                        assists_total = goals.get("assists") or 0
                        rating_raw = games.get("rating")
                        rating = round(float(rating_raw), 2) if rating_raw else 6.8

                        if minutes and minutes > 0:
                            xg90 = round(goals_total / minutes * 90, 3)
                            xa90 = round(assists_total / minutes * 90, 3)
                        else:
                            xg90 = fallback_xg90.get(pos, 0.08)
                            xa90 = fallback_xa90.get(pos, 0.05)

                        rows.append({
                            "player": info.get("name"),
                            "position": pos,
                            "minutes": minutes,
                            "xG90": xg90,
                            "xA90": xa90,
                            "key_passes90": round(assists_total / max(minutes, 1) * 90 * 1.8, 2),
                            "avg_rating": rating,
                            "status": "Active",
                        })
                    page += 1

                if rows:
                    return pd.DataFrame(rows)
        except Exception:
            pass
        # --------------------------------------------------------------
        # >>> TRUE xG/xA FETCHER CONFIGURATION POINT <<<
        # Replace the goals-based proxy above with real per-player xG/xA:
        #   import soccerdata as sd
        #   understat = sd.Understat(leagues=..., seasons=_current_season_year())
        #   player_stats = understat.read_player_season_stats()
        # --------------------------------------------------------------

    return _mock_squad(team_name, seed_offset)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_team_form(team_name: str, league_name: str = "Premier League") -> pd.DataFrame:
    """
    5-match rolling form (goals/xG-proxy for and against, most recent last).

    BUGFIX: previously returned `_mock_form(team_name)` unconditionally, same
    class of bug as fetch_team_season_stats above — a team's "recent form"
    was pure random noise regardless of API keys. Now pulls the team's last
    5 finished matches from football-data.org when FOOTBALL_DATA_KEY is set,
    using actual goals for/against as the xG proxy (again: goals, not true
    xG — see the fetcher configuration comment for wiring in real xG).
    """
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
                        "xG_for": gf, "xG_against": ga,  # goals-based proxy, not true xG
                    })
                if len(rows) >= 3:
                    return pd.DataFrame(rows)
        except Exception:
            pass
        # --------------------------------------------------------------
        # >>> TRUE xG FORM FETCHER CONFIGURATION POINT <<<
        # Replace the goals-based proxy above with soccerdata's Understat
        # reader for genuine per-match xG/xGA:
        #   understat.read_team_match_stats()  # then take last 5 rows
        # --------------------------------------------------------------
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
# 4. QUANTITATIVE MODELING ENGINE
# ====================================================================================

def calculate_team_base_lambdas(home_team: str, away_team: str, team_stats: pd.DataFrame) -> Tuple[float, float]:
    """Calculates team-specific base xG expectations based on relative attack and defense parameters."""
    league_home_avg = max(team_stats['home_xG_for'].mean(), 1.0)
    league_away_avg = max(team_stats['away_xG_for'].mean(), 1.0)
    
    h_stat = team_stats.loc[team_stats['team'] == home_team]
    a_stat = team_stats.loc[team_stats['team'] == away_team]
    
    home_att = (h_stat['home_xG_for'].values[0] if not h_stat.empty else 1.55) / league_home_avg
    away_def = (a_stat['away_xG_against'].values[0] if not a_stat.empty else 1.20) / league_home_avg
    
    away_att = (a_stat['away_xG_for'].values[0] if not a_stat.empty else 1.20) / league_away_avg
    home_def = (h_stat['home_xG_against'].values[0] if not h_stat.empty else 1.55) / league_away_avg
    
    base_lam_home = league_home_avg * home_att * away_def
    base_lam_away = league_away_avg * away_att * home_def
    
    return round(base_lam_home, 3), round(base_lam_away, 3)


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
    """Applies a penalty for short rest turnarounds."""
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
# 5. DEVIGGING & KELLY TRADING LAYER
# ====================================================================================

def fair_odds(prob: float) -> float:
    return round(1 / prob, 3) if prob > 1e-6 else float("inf")


def apply_margin(probs: dict, target_margin_pct: float) -> dict:
    margin_factor = 1 + (target_margin_pct / 100)
    return {k: round(fair_odds(v) / margin_factor, 2) for k, v in probs.items()}


def devig_proportional(home_o: float, draw_o: float, away_o: float) -> Optional[Tuple[float, float, float]]:
    """
    Removes bookmaker overround to find true fair market probabilities.

    BUGFIX: previously returned (0.0, 0.0, 0.0) when any single price was
    missing/invalid. Because these three values are computed ONCE and reused
    across all three 1X2 rows, a single missing Draw price used to zero out
    Home and Away's "fair market prob" too — inflating Edge (pp) to roughly
    the full model probability and falsely tripping the VALUE signal on
    outcomes whose prices were actually fine. Now returns None so the caller
    can fall back to a per-market naive comparison instead of a corrupted one.
    """
    if not all([home_o, draw_o, away_o]) or any(o <= 1.0 for o in [home_o, draw_o, away_o]):
        return None
    raw_h, raw_d, raw_a = 1 / home_o, 1 / draw_o, 1 / away_o
    overround = raw_h + raw_d + raw_a
    return round(raw_h / overround, 4), round(raw_d / overround, 4), round(raw_a / overround, 4)


def devig_two_way(odd_a: float, odd_b: float) -> Optional[Tuple[float, float]]:
    """
    Proportional devig for two-outcome markets (Over/Under 2.5, BTTS).
    Replaces the previous hardcoded fair_mkt_prob=0.50 placeholder, which
    ignored the actual bookmaker prices for these markets entirely.
    """
    if not odd_a or not odd_b or odd_a <= 1.0 or odd_b <= 1.0:
        return None
    raw_a, raw_b = 1 / odd_a, 1 / odd_b
    overround = raw_a + raw_b
    return round(raw_a / overround, 4), round(raw_b / overround, 4)


def kelly_stake(model_prob: float, book_odds: float, fraction: float = 0.25) -> float:
    """Calculates Fractional Kelly Criterion bankroll stake %."""
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
# 6. STREAMLIT SIDEBAR & INPUTS
# ====================================================================================
st.sidebar.title("⚽ Sportsbook Odds Engine")
st.sidebar.caption("Quant Modeling & Devigged Edge Detection")

if DEMO_MODE:
    st.sidebar.warning("⚠️ **DEMO MODE**: Set API keys in `st.secrets` for live feeds.")

st.sidebar.subheader("API Connections")
st.sidebar.markdown(f"{'🟢' if API_FOOTBALL_KEY else '🔴'} API-Football")
st.sidebar.markdown(f"{'🟢' if FOOTBALL_DATA_KEY else '🔴'} football-data.org")
st.sidebar.markdown(f"{'🟢' if THE_ODDS_API_KEY else '🔴'} The Odds API")
st.sidebar.markdown("🟢 Open-Meteo Weather")

if st.sidebar.button("🔄 Refresh Rosters & Live Stats", use_container_width=True):
    # All fetch_* functions are @st.cache_data — this clears every cached
    # response (fixtures, season stats, squads, form, odds, weather) so the
    # next run pulls fresh data instead of a stale cached roster/injury list.
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("Cached for 15–30 min per source. Use refresh after squad/injury news breaks.")

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
home_squad_raw = fetch_squad(home_team, seed_offset=1)
away_squad_raw = fetch_squad(away_team, seed_offset=2)

with st.sidebar.expander(f"🏠 {home_team} Lineup"):
    home_active = {row["player"]: st.checkbox(f"{row['player']} ({row['position']})", value=True, key=f"h_{row['player']}") for _, row in home_squad_raw.iterrows()}

with st.sidebar.expander(f"🚗 {away_team} Lineup"):
    away_active = {row["player"]: st.checkbox(f"{row['player']} ({row['position']})", value=True, key=f"a_{row['player']}") for _, row in away_squad_raw.iterrows()}

# ====================================================================================
# 7. COMPUTATION & DATA PROCESSING
# ====================================================================================
city, lat, lon = _STADIUM_CITIES.get(league, ("London", 51.5072, -0.1276))
weather = fetch_weather(lat, lon, kickoff)
w_mult = weather_modifier(weather)

season_stats = fetch_team_season_stats(league)
base_lam_home, base_lam_away = calculate_team_base_lambdas(home_team, away_team, season_stats)

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
# 8. DASHBOARD DISPLAY
# ====================================================================================
st.title(f"{home_team} vs {away_team}")
st.caption(f"{league} · Kickoff {kickoff.strftime('%A %d %B, %H:%M')} · Venue: {city}")

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

st.divider()

# ---- Squad & Player Form Section ----
st.subheader("👥 Player Form & Impact Penalties")
pc1, pc2 = st.columns(2)
with pc1:
    st.markdown(f"**{home_team} Lineup & Stats**")
    st.dataframe(home_squad[["player", "position", "avg_rating", "xG90", "xA90", "status", "absence_penalty_%"]], hide_index=True)
with pc2:
    st.markdown(f"**{away_team} Lineup & Stats**")
    st.dataframe(away_squad[["player", "position", "avg_rating", "xG90", "xA90", "status", "absence_penalty_%"]], hide_index=True)

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

    # BUGFIX: when the devig couldn't run (missing/partial bookmaker prices),
    # fair_mkt_prob arrives as None here instead of a silently-wrong 0.0.
    # Fall back to the raw (non-devigged) implied probability of this specific
    # price and flag the row so it's clear the margin wasn't removed — this
    # avoids fabricating a huge false edge on markets whose own price is fine.
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
