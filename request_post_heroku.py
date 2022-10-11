import requests

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


response = requests.post('https://yhbyhb-nd0821-c3.herokuapp.com/inference/', json=input_data)

print("response status code: ", response.status_code)
print("response: ", response.json())