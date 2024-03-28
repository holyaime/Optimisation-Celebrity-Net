# -*- coding: utf-8 -*-
import pytest

from celebrity_recognition_ai.app.api import app


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
