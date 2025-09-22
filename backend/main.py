from fastapi import FastAPI
from pymongo import MongoClient
import os

app = FastAPI()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://appuser:strongpassword@db:27017/appdb?authSource=admin")
client = MongoClient(MONGO_URI)

@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok", "db": "ok"}
    except Exception as e:
        return {"status": "error", "db": str(e)}

# example backend route
@app.get("/backend/hello")
def hello():
    return {"message": "Hello from Backend service"}
