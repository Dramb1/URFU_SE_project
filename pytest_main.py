# test_main.py

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Reid person" in response.text
    assert "Add new person" in response.text


def test_get_reidentification():
    response = client.get("/reidentification")
    assert response.status_code == 200
    assert "Upload Image" in response.text
    assert "Submit" in response.text


def test_post_reidentification():
    with open("./data/allegri.jpg", "rb") as image_file:
        files = {"file": ("allegri.jpg", image_file, "image/jpeg")}

        response = client.post("/reidentification", files=files)
        assert response.status_code == 200
        assert "Back to home page" in response.text
        assert "person ID: Unknown" in response.text


def test_get_add_person():
    response = client.get("/add_person")
    assert response.status_code == 200
    assert "Upload Image" in response.text
    assert "Id person" in response.text
    assert "Save person" in response.text
    assert "Back to home page" in response.text


def test_post_add_person():
    with open("./data/mbappe.jpg", "rb") as image_file:
        files = {"file": ("mbappe.jpg", image_file, "image/jpeg")}
        data = {"id_person": "Mbappe"}

        response = client.post("/add_person", files=files, data=data)
        assert response.status_code == 200
        assert "Person saved successfully" in response.text


def test_post_bad_add_person():
    with open("./main.py", "rb") as image_file:
        files = {"file": ("main.py", image_file, "image/jpeg")}
        data = {"id_person": "Mbappe"}

        response = client.post("/add_person", files=files, data=data)
        assert response.status_code == 500


def test_post_correct_reidentification():
    with open("./data/mbappe2.jpg", "rb") as image_file:
        files = {"file": ("mbappe2.jpg", image_file, "image/jpeg")}

        response = client.post("/reidentification", files=files)
        assert response.status_code == 200
        assert "Back to home page" in response.text
        assert "person ID: Mbappe" in response.text
