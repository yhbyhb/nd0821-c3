'''
# test_main.py
Unit test for main.py

'''

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_get_welcome():
    '''
    test GET
    '''
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"welcome": "Welcome!"}

def test_inference_above50k():
    '''
    test POST and response is ">50K"
    '''
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
    assert response_post.status_code == 200, \
        f"Response code is not successful. {response_post.json()}"
    assert response_post.json() == {"prediction":'>50K'}, \
        "Wrong prediction. expectation '>50K'"

def test_inference_under50k():
    '''
    test POST and response is "<=50K"
    '''
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
    assert response_post.status_code == 200, \
        f"Response code is not successful. {response_post.json()}"
    assert response_post.json() == {"prediction":'<=50K'}, \
        "Wrong prediction. expectation '<=50K'"
