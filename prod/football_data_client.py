from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, MutableMapping, Optional

import pandas as pd
import requests


class FootballDataAPIError(RuntimeError):
    """Error genérico para respuestas no exitosas del API."""


@dataclass
class FootballDataClient:
    """
    Pequeño cliente para https://www.football-data.org/documentation/quickstart

    La API tiene un límite de 10 requests por minuto, por lo que este cliente
    aplica un `request_interval` simple entre llamadas consecutivas.
    """

    token: Optional[str] = None
    base_url: str = "https://api.football-data.org/v4"
    timeout: int = 30
    request_interval: float = 6.5  # ~9 req/min para estar del lado seguro
    session: Optional[requests.Session] = None
    _last_request_ts: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.token is None:
            self.token = os.getenv("FOOTBALL_DATA_API_TOKEN")
        if not self.token:
            raise FootballDataAPIError(
                "Debes definir el token mediante FOOTBALL_DATA_API_TOKEN o al instanciar FootballDataClient(token='...')."
            )
        self.session = self.session or requests.Session()

    # --------- Public helpers ---------
    def get_competitions(self, plan: str = "TIER_ONE", areas: Optional[Iterable[int]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"plan": plan}
        if areas:
            params["areas"] = ",".join(str(area) for area in areas)
        return self._request("/competitions", params)

    def get_standings(self, competition_code: str, season: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if season:
            params["season"] = season
        return self._request(f"/competitions/{competition_code}/standings", params)

    def get_matches(
        self,
        competition_code: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        matchday: Optional[int] = None,
        team_id: Optional[int] = None,
        season: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if status:
            params["status"] = status
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        if matchday:
            params["matchday"] = matchday
        if team_id:
            params["teams"] = team_id
        if season:
            params["season"] = season

        if competition_code:
            endpoint = f"/competitions/{competition_code}/matches"
        else:
            endpoint = "/matches"
        return self._request(endpoint, params)

    def get_team(self, team_id: int) -> Dict[str, Any]:
        return self._request(f"/teams/{team_id}")

    # --------- Private helpers ---------
    def _request(self, endpoint: str, params: Optional[MutableMapping[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        self._respect_rate_limit()
        headers = {
            "X-Auth-Token": self.token,
            "Accept": "application/json",
        }
        response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        self._last_request_ts = time.time()
        if not response.ok:
            raise FootballDataAPIError(self._format_error(response))
        return response.json()

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)

    @staticmethod
    def _format_error(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        message = payload.get("message") or response.text
        return f"FootballData API error ({response.status_code}): {message}"


# --------- DataFrame helpers ---------
def standings_to_dataframe(payload: Dict[str, Any], table_type: str = "TOTAL") -> pd.DataFrame:
    """
    Convierte la respuesta de standings en un DataFrame listo para Streamlit.

    Args:
        payload: dict retornado por `client.get_standings`.
        table_type: TOTAL / HOME / AWAY segun documentaciòn del API.
    """
    standings = payload.get("standings", [])
    selected = next((table for table in standings if table.get("type") == table_type.upper()), None)
    if not selected:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    competition = payload.get("competition", {})
    season = payload.get("season", {})
    for entry in selected.get("table", []):
        team = entry.get("team", {})
        rows.append(
            {
                "competition": competition.get("name"),
                "season_start": season.get("startDate"),
                "season_end": season.get("endDate"),
                "position": entry.get("position"),
                "team_id": team.get("id"),
                "team": team.get("shortName") or team.get("name"),
                "played": entry.get("playedGames"),
                "won": entry.get("won"),
                "draw": entry.get("draw"),
                "lost": entry.get("lost"),
                "points": entry.get("points"),
                "goals_for": entry.get("goalsFor"),
                "goals_against": entry.get("goalsAgainst"),
                "goal_diff": entry.get("goalDifference"),
            }
        )

    return pd.DataFrame.from_records(rows)


def matches_to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Convierte la respuesta de `client.get_matches` en un DataFrame compacto.
    """
    matches = payload.get("matches", [])
    rows: List[Dict[str, Any]] = []
    for match in matches:
        home = match.get("homeTeam", {})
        away = match.get("awayTeam", {})
        score = match.get("score", {})
        full_time = score.get("fullTime", {}) or {}
        odds = match.get("odds", {})
        rows.append(
            {
                "match_id": match.get("id"),
                "competition": (match.get("competition") or {}).get("code"),
                "season": (match.get("season") or {}).get("startDate"),
                "utc_date": match.get("utcDate"),
                "status": match.get("status"),
                "matchday": match.get("matchday"),
                "home_team": home.get("shortName") or home.get("name"),
                "away_team": away.get("shortName") or away.get("name"),
                "home_team_id": home.get("id"),
                "away_team_id": away.get("id"),
                "full_time_home": full_time.get("home"),
                "full_time_away": full_time.get("away"),
                "winner": (score.get("winner") or "").lower(),
                "prob_home_win": odds.get("homeWin"),
                "prob_draw": odds.get("draw"),
                "prob_away_win": odds.get("awayWin"),
                "referee": (match.get("referees") or [{}])[0].get("name"),
            }
        )

    return pd.DataFrame.from_records(rows)


def latest_matchday_df(client: FootballDataClient, competition_code: str) -> pd.DataFrame:
    """Devuelve un DataFrame con la ultima jornada disponible para la liga."""
    payload = client.get_matches(competition_code=competition_code, status="FINISHED")
    df = matches_to_dataframe(payload)
    if df.empty or "matchday" not in df.columns:
        return df

    df = df.dropna(subset=["matchday"])
    if df.empty:
        return df

    last_matchday = df["matchday"].max()
    return df[df["matchday"] == last_matchday].sort_values("utc_date")


__all__ = [
    "FootballDataClient",
    "FootballDataAPIError",
    "standings_to_dataframe",
    "matches_to_dataframe",
    "latest_matchday_df",
]
