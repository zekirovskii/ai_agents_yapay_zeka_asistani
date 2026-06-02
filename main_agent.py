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
from langchain_community.utilities import SerpAPIWrapper # web search

# .env dosyasından API anahtarlarını yükle
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Tool 1: SerpAPI Web Search Tool
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY) #