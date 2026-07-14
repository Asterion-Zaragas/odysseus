"""Route-level per-user preset isolation.

/api/presets* resolve the owner from `effective_user(request)` and are no
longer admin-gated — any authenticated user manages *their own* presets
(auth itself is enforced by the app-level AuthMiddleware, not per-route).
Two users must see and edit independent stores; owner=None (no-auth mode)
keeps the legacy shared behavior.
"""
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from routes.preset_routes import setup_preset_routes
from src.preset_manager import PresetManager


class _StampUserMiddleware(BaseHTTPMiddleware):
    """Stand-in for the app AuthMiddleware: trust an X-Test-User header."""

    async def dispatch(self, request: Request, call_next):
        user = request.headers.get("X-Test-User")
        request.state.current_user = user or None
        return await call_next(request)


def _make_client(tmp_path) -> TestClient:
    app = FastAPI()
    app.add_middleware(_StampUserMiddleware)
    app.include_router(setup_preset_routes(PresetManager(str(tmp_path))))
    return TestClient(app)


def _h(user):
    return {"X-Test-User": user} if user else {}


def test_non_admin_users_manage_their_own_presets(tmp_path):
    client = _make_client(tmp_path)

    # Alice (a regular user — no admin gate anywhere) saves her persona.
    r = client.post("/api/presets/custom", headers=_h("alice"), json={
        "temperature": 0.7, "max_tokens": 1000,
        "system_prompt": "alice persona", "name": "Aria", "enabled": True,
    })
    assert r.status_code == 200 and r.json()["success"] is True

    # Bob saves a different one.
    r = client.post("/api/presets/custom", headers=_h("bob"), json={
        "temperature": 0.3, "max_tokens": 500,
        "system_prompt": "bob persona", "name": "Bort", "enabled": True,
    })
    assert r.status_code == 200 and r.json()["success"] is True

    # Each sees only their own custom persona; built-ins present for both.
    a = client.get("/api/presets", headers=_h("alice")).json()
    b = client.get("/api/presets", headers=_h("bob")).json()
    assert a["custom"]["system_prompt"] == "alice persona"
    assert b["custom"]["system_prompt"] == "bob persona"
    for key in PresetManager.DEFAULT_PRESETS:
        assert key in a and key in b


def test_templates_and_groups_are_per_user(tmp_path):
    client = _make_client(tmp_path)

    r = client.post("/api/presets/templates", headers=_h("alice"),
                    json={"name": "A-tmpl", "system_prompt": "p"})
    assert r.status_code == 200
    tid = r.json()["template"]["id"]

    assert client.get("/api/presets/templates", headers=_h("bob")).json() == []
    names = [t["name"] for t in client.get("/api/presets/templates", headers=_h("alice")).json()]
    assert names == ["A-tmpl"]

    r = client.post("/api/presets/groups", headers=_h("alice"), json={"groups": [{"name": "G"}]})
    assert r.status_code == 200
    assert client.get("/api/presets/groups", headers=_h("alice")).json()["groups"] == [{"name": "G"}]
    assert client.get("/api/presets/groups", headers=_h("bob")).json()["groups"] == []

    # Deleting Alice's template doesn't need admin and doesn't touch Bob.
    r = client.delete(f"/api/presets/templates/{tid}", headers=_h("alice"))
    assert r.status_code == 200 and r.json()["success"] is True
    assert client.get("/api/presets/templates", headers=_h("alice")).json() == []


def test_no_auth_mode_still_works(tmp_path):
    client = _make_client(tmp_path)

    r = client.post("/api/presets/custom", json={
        "temperature": 1.0, "max_tokens": 0,
        "system_prompt": "single-user persona", "name": "Solo", "enabled": True,
    })
    assert r.status_code == 200 and r.json()["success"] is True
    assert client.get("/api/presets").json()["custom"]["system_prompt"] == "single-user persona"
