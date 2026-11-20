"""Model Arena: two hidden models answer the same question, a human picks the better one.

The votes become a "preference dataset" (prompt, chosen answer, rejected answer), the same kind of
data AI labs collect to fine-tune chat models with RLHF / DPO.
"""
import json
import random
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.api.routes import error, sse_response, stream_chat
from app.backends.base import GenerationParams

from .store import leaderboard

router = APIRouter(prefix="/arena", tags=["arena"])


class BattleRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)


class VoteRequest(BaseModel):
    winner: Literal["a", "b", "tie", "both_bad"]


@router.post("/battles")
async def create_battle(req: BattleRequest, request: Request):
    settings = request.app.state.settings
    models = settings.arena_models or settings.models
    if len(models) < 2:
        return error(400, "invalid_request", "The arena needs at least 2 models.")
    model_a, model_b = random.sample(models, 2)  # random pair, random sides
    battle_id = request.app.state.arena.create(req.prompt, model_a, model_b)
    return {"battle_id": battle_id}  # model names stay secret until the vote


@router.get("/battles/{battle_id}/stream/{side}")
async def stream_side(battle_id: str, side: Literal["a", "b"], request: Request):
    state = request.app.state
    battle = state.arena.get(battle_id)
    if battle is None:
        return error(404, "not_found", "Unknown battle.")
    if battle[f"answer_{side}"] is not None:
        return error(409, "conflict", "This answer was already generated.")
    if state.limiter.is_full():
        return error(429, "overloaded", "Server is busy, please retry shortly.")

    params = GenerationParams(max_tokens=256, model=battle[f"model_{side}"])
    messages = [{"role": "user", "content": battle["prompt"]}]

    def save(text, usage, metrics):
        state.arena.save_answer(battle_id, side, text, metrics["tokens_per_sec"])

    return sse_response(stream_chat(state.backend, state.limiter, state.settings, messages, params,
                                    on_complete=save))


@router.post("/battles/{battle_id}/vote")
async def vote(battle_id: str, req: VoteRequest, request: Request):
    store = request.app.state.arena
    battle = store.get(battle_id)
    if battle is None:
        return error(404, "not_found", "Unknown battle.")
    if battle["winner"] is not None:
        return error(409, "conflict", "This battle already has a vote.")
    if battle["answer_a"] is None or battle["answer_b"] is None:
        return error(409, "conflict", "Both answers must finish before voting.")
    store.vote(battle_id, req.winner)
    return {"winner": req.winner, "model_a": battle["model_a"], "model_b": battle["model_b"]}


@router.get("/leaderboard")
async def get_leaderboard(request: Request):
    battles = request.app.state.arena.voted()
    return {"total_votes": len(battles), "models": leaderboard(battles)}


@router.get("/export.jsonl")
async def export(request: Request):
    """One JSON line per decisive vote: the format used for DPO preference training."""
    lines = []
    for b in request.app.state.arena.voted():
        if b["winner"] not in ("a", "b"):
            continue  # ties and "both bad" don't say which answer is preferred
        win, lose = (b["winner"], "b" if b["winner"] == "a" else "a")
        lines.append(json.dumps({"prompt": b["prompt"],
                                 "chosen": b[f"answer_{win}"], "rejected": b[f"answer_{lose}"],
                                 "chosen_model": b[f"model_{win}"], "rejected_model": b[f"model_{lose}"]}))
    return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""), media_type="application/jsonl",
                             headers={"Content-Disposition": "attachment; filename=preferences.jsonl"})
