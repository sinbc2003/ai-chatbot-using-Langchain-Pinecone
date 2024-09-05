import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
import streamlit as st

# Load environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
    raise ValueError("Missing required environment variables. Please set OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME.")

client = OpenAI(api_key=OPENAI_API_KEY)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(PINECONE_INDEX_NAME)

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2, include_metadata=True)
    
    matches = result.get('matches', [])
    if not matches:
        return "No matching results found."
    
    texts = []
    for match in matches:
        if 'metadata' in match and 'text' in match['metadata']:
            texts.append(match['metadata']['text'])
    
    return "\n".join(texts) if texts else "No text content found in the matches."

def query_refiner(conversation, query):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base."},
            {"role": "user", "content": f"Conversation Log:\n{conversation}\n\nQuery: {query}"}
        ],
        temperature=0.7,
        max_tokens=256,
    )
    return response.choices[0].message.content

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += f"Human: {st.session_state['requests'][i]}\n"
        conversation_string += f"Bot: {st.session_state['responses'][i+1]}\n"
    return conversation_string
