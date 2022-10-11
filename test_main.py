# test_main.py

import json
from main import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_get_welcome():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome": "Welcome!"}

def test_inference_above50k():
    input_data = {
        "age" : 37,
        "workclass" : "Private",
        "fnlgt" : 280464,
        "education" : "Some-college",
        "education-num" : 10,
        "marital-status" : "Married-civ-spouse",
        "occupation" : "Exec-managerial",
        "relationship" : "Husband",
        "race" : "Black",
        "sex" : "Male",
        "capital-gain" : 0,
        "capital-loss" : 0,
        "hours-per-week" : 80,
        "native-country" : "United-States",                
    }

    response_post = client.post("/inference/", json=input_data)
    print(response_post.json)
    assert response_post.status_code == 200, "Response code is not successful. {}".format(response_post.json())
    assert response_post.json() == {"prediction":'>50K'}, "Wrong prediction. expectation '>50K'"

def test_inference_under50k():
    input_data = {
        "age": 53,
        "workclass": "Private",
        "fnlgt": 234721,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Married-civ-spouse",
        "occupation": "Handlers-cleaners",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",              
    }

    response_post = client.post("/inference/", json=input_data)
    print(response_post.json)
    assert response_post.status_code == 200, "Response code is not successful. {}".format(response_post.json())
    assert response_post.json() == {"prediction":'<=50K'}, "Wrong prediction. expectation '<=50K'"

        