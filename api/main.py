from fastapi import FastAPI
from pymongo import MongoClient
import os

app = FastAPI()

# MongoDB connection (from env var in docker-compose.yml)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://neuralnet:NNetBlr@db:27017/crosstabtb?authSource=admin")
client = MongoClient(MONGO_URI)

@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok", "db": "ok"}
    except Exception as e:
        return {"status": "error", "db": str(e)}

# example API route
@app.get("/api/hello")
def hello():
    return {"message": "Hello from API service"}