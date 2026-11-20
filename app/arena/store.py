"""Saves arena battles and votes in SQLite (a single file, built into Python), and scores the models.

Elo is the chess rating system: everyone starts at 1000; beating a stronger model gains more points
than beating a weaker one. The ratings are recomputed from all votes, in the order they happened.
"""
import sqlite3
import time
import uuid
from pathlib import Path

ELO_START, ELO_K = 1000.0, 32  # K = how many points one result can move a rating at most


class ArenaStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS battles (
            id TEXT PRIMARY KEY, prompt TEXT NOT NULL, created_at REAL NOT NULL,
            model_a TEXT NOT NULL, model_b TEXT NOT NULL,
            answer_a TEXT, answer_b TEXT, tps_a REAL, tps_b REAL,
            winner TEXT)""")  # winner: a | b | tie | both_bad | NULL (not voted yet)

    def create(self, prompt: str, model_a: str, model_b: str) -> str:
        battle_id = uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO battles (id, prompt, created_at, model_a, model_b) VALUES (?,?,?,?,?)",
                            (battle_id, prompt, time.time(), model_a, model_b))
        return battle_id

    def get(self, battle_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM battles WHERE id = ?", (battle_id,)).fetchone()

    def save_answer(self, battle_id: str, side: str, text: str, tokens_per_sec: float) -> None:
        assert side in ("a", "b")  # side is part of a column name, so only these two are allowed
        with self.db:
            self.db.execute(f"UPDATE battles SET answer_{side} = ?, tps_{side} = ? WHERE id = ?",
                            (text, tokens_per_sec, battle_id))

    def vote(self, battle_id: str, winner: str) -> None:
        with self.db:
            self.db.execute("UPDATE battles SET winner = ? WHERE id = ?", (winner, battle_id))

    def voted(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM battles WHERE winner IS NOT NULL ORDER BY created_at").fetchall()


def leaderboard(battles: list) -> list[dict]:
    """Wins, losses, ties, win rate, average speed and Elo per model, best first."""
    table: dict[str, dict] = {}

    def row(model: str) -> dict:
        return table.setdefault(model, {"model": model, "elo": ELO_START, "battles": 0,
                                        "wins": 0, "losses": 0, "ties": 0, "_speeds": []})

    for b in battles:
        a, bb = row(b["model_a"]), row(b["model_b"])
        score_a = {"a": 1.0, "b": 0.0}.get(b["winner"], 0.5)  # tie and both_bad count as a draw
        expected_a = 1 / (1 + 10 ** ((bb["elo"] - a["elo"]) / 400))
        a["elo"] += ELO_K * (score_a - expected_a)
        bb["elo"] -= ELO_K * (score_a - expected_a)
        for r, score, tps in ((a, score_a, b["tps_a"]), (bb, 1 - score_a, b["tps_b"])):
            r["battles"] += 1
            r["wins" if score == 1 else "losses" if score == 0 else "ties"] += 1
            if tps:
                r["_speeds"].append(tps)

    result = []
    for r in table.values():
        speeds = r.pop("_speeds")
        r["elo"] = round(r["elo"])
        r["win_rate"] = round(r["wins"] / r["battles"], 2)
        r["avg_tokens_per_sec"] = round(sum(speeds) / len(speeds), 1) if speeds else None
        result.append(r)
    return sorted(result, key=lambda r: r["elo"], reverse=True)
