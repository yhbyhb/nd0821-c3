# main.py
import os
import pandas as pd
import pathlib
import pickle

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

class DataModel(BaseModel):
    age: int 
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
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
        }

@app.get("/")
async def get_welcome():
    return {"welcome": "Welcome!"}


@app.post("/inference/")
async def predict(input_data: DataModel):
    """
    """
    # convert data to DataFrame
    data = jsonable_encoder(input_data)
    df = pd.DataFrame(data=data.values(), index=data.keys()).T
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # load model
    cur_path = pathlib.Path(__file__).parent.resolve()
    model_path = os.path.join(cur_path, "model");

    model_dir = 'model'
    model_path = os.path.join(os.path.abspath(os.curdir), model_dir)

    with open(os.path.join(model_path, "model.pkl"), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_path, "model_encoder.pkl"), 'rb') as f:
        encoder = pickle.load(f)
    with open(os.path.join(model_path, "model_lb.pkl"), 'rb') as f:
        lb = pickle.load(f)

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False)

    preds = inference(model, X)
    prediction = {"prediction": lb.inverse_transform(preds)[0]}

    return prediction

