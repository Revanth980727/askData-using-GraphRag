import streamlit as st
from sqlalchemy import create_engine, MetaData, text
from langchain.chat_models import ChatOpenAI
import networkx as nx

# Replace with your actual MySQL credentials
DATABASE_URL = "mysql+pymysql://USER:Password@localhost:3306/Database"
engine = create_engine(DATABASE_URL)
metadata = MetaData()
metadata.reflect(bind=engine)
