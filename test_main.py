# test_main.py

import json
from main import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_get_welcome():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome": "Welcome!"}