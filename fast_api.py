"""
fast api ile /ask endpointi olusturma
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# langchain modulleri içeri aktar
from langchain_classic.agents import initialize_agent, AgentType # ajan tanımlama
from langchain_core.tools import Tool # tool tanımlama
from langchain_classic.memory import ConversationBufferMemory # memory tool için
from langchain_google_genai import ChatGoogleGenerativeAI # google gemini 2.5 flash modeli için
from serpapi import GoogleSearch

# custom tool'lar
from tools.rag_tool import create_rag_tool
from tools.calculator_tool import calculator_tool
from tools.custom_discount_tool import discount_calculator

# .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")

# LLM yapılandırması
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.7,
    google_api_key = GOOGLE_API_KEY
)

# Tools
# 1. Serp api
def web_search(query: str) -> str:
    if not SERP_API_KEY:
        return "Hata: SERPAPI_API_KEY tanımlı değil."

    search = GoogleSearch(
        {
            "api_key": SERP_API_KEY,
            "engine": "google",
            "q": query,
            "hl": "tr",
            "gl": "tr",
        }
    )
    results = search.get_dict()
    organic_results = results.get("organic_results", [])[:3]

    if not organic_results:
        return "Arama sonucu bulunamadı."

    formatted_results = []
    for item in organic_results:
        title = item.get("title", "Başlıksız sonuç")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        formatted_results.append(f"{title}\n{snippet}\n{link}".strip())

    return "\n\n".join(formatted_results)

# 2. matematiksel işlemler
# 3. custom indirim hesaplayıcı
# 4. rag
rag_tool = create_rag_tool("data/musteri_destek.pdf",llm)

# tüm tool ları listeye ekleyelim
tools = [
    Tool(
        name="Web Search",
        func=web_search,
        description="Google araması yaparak güncel bilgi sağlar. Soruya uygun anahtar kelimelerle arama yapar ve sonuçları döndürür."
    ),
    calculator_tool,
    discount_calculator,
    rag_tool
]   

# memory kullanımı

user_memories = {}
def get_user_memory(user_id: str):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory(memory_key="history", return_messages=True)
    return user_memories[user_id]

# fast api
app = FastAPI(title = "Gemini Çok Araçlı AI Agent API", version="1.0")

class UserRequest(BaseModel):
    user_id:str
    message: str

# api endpoint
@app.post("/ask")
async def ask_agent(request: UserRequest):

    user_id = request.user_id
    user_input = request.message

    # kullanıcı hafızasını al
    memory = get_user_memory(user_id)

    # geçmiş konuşmayı string haline getir
    chat_history = "\n".join(
        [f"Kullanıcı: {msg.content}" if msg.type == "human" else f"Asistan: {msg.content}" for msg in memory.chat_memory.messages]
    )

    prompt_with_memory = f"{chat_history}\nKullanıcı:{user_input}\nAsistan:"

    # ajanı olustur
    agent = initialize_agent(
        tools = tools,
        llm = llm,
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    try:
        response = agent.run(prompt_with_memory)
    except Exception as e:
        raise HTTPException(status_code=500, detail = str(e))
    
    # soruyu ve yanıtı hafızaya ekle
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response)

    return {"user_id":user_id, "response":response}
