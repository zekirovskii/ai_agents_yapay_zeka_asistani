"""
Problem Tanımı:
    - Bu projenin amacı, Google gemini 2.5 Flash modeli kullanarak çok araçlı (multi-tool) bir yapay zeka ajanı geliştirmek.
    - Ajan, kullanıcıdan gelen doğal dil girdilerini anlayıp, uygun aracı seçerek (tool call) görevleri otonom şeklinde yerine getirecek.
    - Ajan, langchain altyapısını kullanacak ve gerçek dünya senaryolarını smüle eden 5 farklı tool a sahip olacak.
        - RAG : belge ile konuşma
        - Calculator : matematiksel işlemleri çözme
        - Discount Tool : indirim hesaplama
        - Web search (serpAPI) : internet üzerinden bilgi arama
        - Memory : konuşma geçmişini hatırlama
    - Bu sistem, kullanıcı mesajşarını analiz eder, uygun aracı seçer, görev sonucunu üretir ve yanıtı doğal dilde oluşturur.
    - Aynı zamanda geçiş ve geçmiş konuşmalarını hatırlayarak bağlamlı bir sohbet deneyimi sağlar.

Kullanılan Teknolojiler:
    - LLM : google gemini 2.5 flash (hızlı, düşük maliyetli, metin tabanlı, tool çağırabilen)
    - Framework : langchain (ajan yapısı, memory yönetimi, tool entegrasyonu)
    - vektör veritabanı : FAISS (pdf içeriğin vektörleştirerek hızlı bir şekilde benzerlik araması)
    - Embedding modeli : LaBSE : çok dilli metin vektörleştirme
    - API ve Arayüz : fastapi ve streamlit
    - Diğer:
        - web search : serpAPI (güncel bilgi arama)
        - dotenv
        - requests (client isteği göndermek için)

Kullancılacak Olan Tool'lar:
    - RAG tool
        - Amaç: Yüklü pdf belgesinden (müsteri_destek.pdf) bilgi çekmek ve kullanıcı sorularını yanıtlamak.
        - Çalışma şekli:
            1. PDF içeriği vektörleştirilir ve FAISS veritabanına kaydedilir.
            2. Kullanıcı bir soru sorduğunda, ajan bu soruyu vektörleştirir ve FAISS veritabanında benzer içerik araması yapar.
            3. En benzer içerik bulunur ve RAG tool'u bu içeriği kullanarak kullanıcı sorusunu yanıtlar.

    - Calculator tool
        - Amaç: Matematiksel işlemleri çözmek.
        - Python eval ile yapılacak 

    - Discount tool
        - Ürün fiyatına % x indirim uygular
        - metinden sayısal fiyat bilgisi çekilir ve  0.x katsayısı ile çarpılır ve indirim uygulanır

    - Web search tool
        - Amaç: İnternet üzerinden güncel bilgi aramak.
        - SerpAPIWrapper kullanarak Google araması yapar ve sonuçları kullanıcıya sunar.

    - Memory tool
        - Amaç: Konuşma geçmişini hatırlamak ve bağlamlı bir sohbet deneyimi sağlamak.
        - Langchain'in ConversationBufferMemory sınıfı kullanılarak, ajan önceki konuşmaları hatırlayabilir ve bu bilgiyi yeni sorulara yanıt verirken kullanabilir.


Plan/Program Akış Şeması:
    - Başlangıç: api anahtarlarının oluşturulması ve okunması
    - Toolların hazırlanması
        - her bir tool için ayrı .py dosyası oluşturulacak
    - Ajan oluşturma
        - ajanı tanımla ve toolları ajana yani llme bağla
    - Hafıza yönetimi
        - kullanıcıya özel memory nesnesi oluşturulur
        - her mesaj sonunda hafızada güncellenir
    - Fast API Katmanı
        - /ask endpointi üzerinden json mesaj alınır
    - İstemci katmanı (client.py)
        - request modülünü kullanarak /ask endpointine istek gönderir
    - Arayüz katmanı (streamlit)

Sistem Çalışma Akışı:

kullanıcı streamit üzerinden sorgu yapar
FastAPI /ask url
Agent -> tool seçimi (RAG, calculator, discount, web search, memory)
Langchain -> tool çağırma + LLM reasoning
Gemini 2.5 Flash -> cevap üretimi
Memory -> geçmişi saklama
Yanıt -> fastapi -> streamlit -> kullanıcıya gösterme

Sonuç: 
    - Bu proje üretken yapay zeka ajanlarının nasıl bir şekilde düşünebilen sistemler haline geldiğini gösteren bir proje olacak.
    - Langchain ve gemini entegrasyonu sayesinde
        - çok araçlı
        - hafızalı
        - belge tabanlı
        - etkileşimli (api, arayüz destekli)
        - - - - - - - - - - - - - - - - - - - - - - - gerçek hayat
        - mlops devops (docker, deployment)
        - monitor
        - gpu balancing
    bir yapay zeka ajanı geliştirilmiş olacak.

Kurulumlar

pip install langchain langchain-google-genai google-generativeai langchain-community faiss-cpu python-dotenv serpapi streamlit google-search-results pypdf sentence-transformers fastapi uvicorn requests
"""

import os
from dotenv import load_dotenv

from langchain_classic.agents import initialize_agent, AgentType # ajan tanımlama
from langchain_core.tools import Tool # tool tanımlama
from langchain_classic.memory import ConversationBufferMemory # memory tool için
from langchain_google_genai import ChatGoogleGenerativeAI # google gemini 2.5 flash modeli için
from serpapi import Client as SerpApiClient # web search

# custom tool'lar
from tools.rag_tool import create_rag_tool
from tools.calculator_tool import calculator_tool
from tools.custom_discount_tool import discount_calculator

# .env dosyasından API anahtarlarını yükle
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# llm yapılandırması
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY)

# Tool 1: SerpAPI Web Search Tool
search_client = SerpApiClient(api_key=SERPAPI_API_KEY) if SERPAPI_API_KEY else None

def web_search(query: str) -> str:
    if not search_client:
        return "Hata: SERPAPI_API_KEY tanımlı değil."

    results = search_client.search({"engine": "google", "q": query, "hl": "tr", "gl": "tr"})
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

# Tool 2: Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Tool 3: Matematik İşlemler (Calculator)

# Tool 4: İndirim Hesaplama (Discount Tool)

# Tool 5: RAG Tool (PDF'den bilgi çekme)
rag_tool = create_rag_tool("data/musteri_destek.pdf", llm)

# Tüm tool'ları bir liste halinde topla
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

# ajan oluşturma (zero shot ReAct)

agent = initialize_agent(
    tools = tools, 
    llm=llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True, 
    handle_parsing_errors=True
)

# örnek kullanım 

if __name__ == "__main__":
    print("Ajan hazır! Konuşmaya başlayabilirsiniz.")

    while True:
        user_input = input("Siz: ")
        if user_input.lower() in ["exit","q","çık"]:
            break

        chat_history = "\n".join(
            [f"Kullanıcı: {msg.content}" if msg.type == "human" else f"Asistan: {msg.content}"for msg in memory.chat_memory.messages]
        )

        prompt_with_memory = f"{chat_history}\nKullanıcı: {user_input}\nAsistan:"
        response = agent.run(prompt_with_memory)

        # yanıtı hafızaya kaydet
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response)
        print(f"Ajan: {response}\n")

"""
- Greeting : Merhaba sen kimsin
- Calculator Tool : bana 15*10 + 33 sorusunun cevabını verebilir misin? 
- Discount Tool : telefon satmak istiyorum. telefonun fiyatı 5600tl .buna indirim uygula
- Web Search Tool : bugün kırklarelide hava kaç derece
- RAG Tool : merhaba bir ürün aldım geri iade edebilir miyim?
- Memory : şimdiye kadar seninle ne konuştum?
"""