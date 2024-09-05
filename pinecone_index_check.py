import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# 필요한 변수들을 환경에서 가져옵니다
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

# Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(PINECONE_INDEX_NAME)

# 인덱스 통계 확인
stats = index.describe_index_stats()
print(f"인덱스 통계:\n{stats}")

# 임베딩 모델 초기화
model = SentenceTransformer('all-MiniLM-L6-v2')

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

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

def get_ai_response(query, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'I don't know'"},
            {"role": "user", "content": f"Context:\n {context} \n\n Query:\n{query}"}
        ],
        temperature=0.7,
        max_tokens=256,
    )
    return response.choices[0].message.content

# 테스트 쿼리 실행
test_query = "What is the main topic of the documents?"
context = find_match(test_query)
response = get_ai_response(test_query, context)

print(f"\n테스트 쿼리: {test_query}")
print(f"찾은 컨텍스트: {context}")
print(f"AI 응답: {response}")
