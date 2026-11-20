"""Choosing a model per request, and JSON mode."""
import json

from tests.conftest import chat_body

EXTRA = '["fake-small", "fake-large"]'


async def test_models_lists_default_first_then_allowed(make_client):
    async with make_client(ALLOWED_MODELS=EXTRA) as c:
        r = await c.get("/v1/models")
    assert [m["id"] for m in r.json()["data"]] == ["fake-model", "fake-small", "fake-large"]


async def test_request_can_pick_an_allowed_model(make_client):
    async with make_client(ALLOWED_MODELS=EXTRA) as c:
        r = await c.post("/v1/chat/completions", json=chat_body(model="fake-small"))
    assert r.status_code == 200
    assert r.json()["model"] == "fake-small"


async def test_unknown_model_is_rejected(make_client):
    async with make_client() as c:
        r = await c.post("/v1/chat/completions", json=chat_body(model="gpt-99"))
    assert r.status_code == 400
    assert "gpt-99" in r.json()["error"]["message"]


async def test_json_object_mode(make_client):
    async with make_client() as c:
        r = await c.post("/v1/chat/completions",
                         json=chat_body(response_format={"type": "json_object"}))
    body = r.json()
    assert json.loads(body["choices"][0]["message"]["content"]) == {"ok": True}
    assert body["metrics"]["valid_json"] is True


async def test_json_schema_mode_is_accepted(make_client):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    async with make_client() as c:
        r = await c.post("/v1/chat/completions", json=chat_body(
            response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}))
    assert r.status_code == 200
    assert r.json()["metrics"]["valid_json"] is True
