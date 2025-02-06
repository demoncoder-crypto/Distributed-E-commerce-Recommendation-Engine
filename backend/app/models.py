from pydantic import BaseModel
from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# --- Pydantic Schemas (API Validation) ---
class UserInteractionCreate(BaseModel):
    user_id: str
    item_id: str
    event_type: str  # "click", "purchase", "view"

# --- SQLAlchemy Models (Database) ---
class UserInteractionDB(Base):
    __tablename__ = "user_interactions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50))
    item_id = Column(String(50))
    event_type = Column(String(20))

class ProductDB(Base):
    __tablename__ = "products"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100))
    category = Column(String(50))
    features = Column(String(500))  # JSON-encoded