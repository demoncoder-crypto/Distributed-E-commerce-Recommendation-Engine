from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from app.models import UserInteractionCreate, UserInteractionDB, ProductDB
from app.kafka_producer import get_kafka_producer
import redis
import json

app = FastAPI()
redis_client = redis.Redis(host="redis", port=6379, db=0)
producer = get_kafka_producer()

# Database dependency
def get_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine("postgresql://user:password@db:5432/recsys")
    return sessionmaker(autocommit=False, bind=engine)()

@app.post("/track")
async def track_event(interaction: UserInteractionCreate, db: Session = Depends(get_db)):
    # Write to DB
    db.add(UserInteractionDB(**interaction.dict()))
    db.commit()
    
    # Send to Kafka
    producer.send("user_events", interaction.dict())
    return {"status": "Event tracked"}