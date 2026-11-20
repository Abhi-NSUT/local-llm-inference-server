"""Model Arena: battles, votes, leaderboard (Elo) and the preference-data export."""
import json

from app.arena.store import leaderboard

TWO_MODELS = '["fake-b"]'  # plus the default "fake-model" = 2 arena models


async def run_battle(c, prompt="hi") -> str:
    battle_id = (await c.post("/arena/battles", json={"prompt": prompt})).json()["battle_id"]
    for side in ("a", "b"):
        r = await c.get(f"/arena/battles/{battle_id}/stream/{side}")
        assert r.text.endswith("data: [DONE]\n\n")
    return battle_id


async def test_full_battle_vote_and_leaderboard(make_client):
    async with make_client(ALLOWED_MODELS=TWO_MODELS) as c:
        battle_id = await run_battle(c)
        vote = (await c.post(f"/arena/battles/{battle_id}/vote", json={"winner": "a"})).json()
        board = (await c.get("/arena/leaderboard")).json()
    assert {vote["model_a"], vote["model_b"]} == {"fake-model", "fake-b"}  # revealed after voting
    assert board["total_votes"] == 1
    winner, loser = board["models"]
    assert winner["model"] == vote["model_a"] and winner["elo"] == 1016 and winner["wins"] == 1
    assert loser["elo"] == 984 and loser["losses"] == 1


async def test_vote_rules(make_client):
    async with make_client(ALLOWED_MODELS=TWO_MODELS) as c:
        battle_id = (await c.post("/arena/battles", json={"prompt": "hi"})).json()["battle_id"]
        early = await c.post(f"/arena/battles/{battle_id}/vote", json={"winner": "a"})
        for side in ("a", "b"):
            await c.get(f"/arena/battles/{battle_id}/stream/{side}")
        again = await c.get(f"/arena/battles/{battle_id}/stream/a")
        first = await c.post(f"/arena/battles/{battle_id}/vote", json={"winner": "tie"})
        second = await c.post(f"/arena/battles/{battle_id}/vote", json={"winner": "b"})
        unknown = await c.post("/arena/battles/nope/vote", json={"winner": "a"})
    assert early.status_code == 409    # answers not finished yet
    assert again.status_code == 409    # an answer can't be regenerated
    assert first.status_code == 200
    assert second.status_code == 409   # one vote per battle
    assert unknown.status_code == 404


async def test_arena_needs_two_models(make_client):
    async with make_client() as c:
        r = await c.post("/arena/battles", json={"prompt": "hi"})
    assert r.status_code == 400


async def test_export_has_only_decisive_votes(make_client):
    async with make_client(ALLOWED_MODELS=TWO_MODELS) as c:
        decisive = await run_battle(c, "q1")
        await c.post(f"/arena/battles/{decisive}/vote", json={"winner": "b"})
        tie = await run_battle(c, "q2")
        await c.post(f"/arena/battles/{tie}/vote", json={"winner": "tie"})
        r = await c.get("/arena/export.jsonl")
    rows = [json.loads(line) for line in r.text.splitlines()]
    assert len(rows) == 1
    assert rows[0]["prompt"] == "q1"
    assert set(rows[0]) == {"prompt", "chosen", "rejected", "chosen_model", "rejected_model"}


def test_elo_rewards_upsets_more():
    # "small" beats "big" twice, then "big" wins once: small should still be ahead.
    battles = [{"model_a": "small", "model_b": "big", "winner": w, "tps_a": 30.0, "tps_b": 10.0}
               for w in ("a", "a", "b")]
    board = leaderboard(battles)
    assert [r["model"] for r in board] == ["small", "big"]
    assert board[0]["wins"] == 2 and board[0]["losses"] == 1
    assert board[0]["win_rate"] == 0.67 and board[0]["avg_tokens_per_sec"] == 30.0
    assert board[0]["elo"] + board[1]["elo"] == 2000  # points move between models, never created
