# main.py

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def get_welcome():
    return {"welcome": "Welcome!"}
