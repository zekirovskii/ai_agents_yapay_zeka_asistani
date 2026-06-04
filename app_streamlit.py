"""
ui yani kullanıcı arayüzünü streamlit kütüphanesi ile implement edelim
"""

import streamlit as st
import requests

# fast api bağlantı ayarları
API_URL = "http://127.0.0.1:8000/ask"
USER_ID = "ysf"

# streamlit ile sayfa ayarları
st.set_page_config(page_title="Gemini AI Agent Chat",page_icon="🤖",layout="centered")
st.title("🤖 Gemini Çok Amaçlı AI Agent")
st.caption("Gemini 2.5 Falsh + RAG + Math + Discount + Memory + Web Search")

# session state
if "messages" not in st.session_state:
    st.session_state["messages"]=[]

# mesaj gönderme
def send_messages_to_api(message: str):
    payload = {"user_id": USER_ID, "message":message}
    try:
        response = requests.post(API_URL,json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("response","Sunucudan yanıt alınamadı.")
        else: 
            return f"Hata({response.status_code}): {response.text}"
    except Exception as e:
        return f"Bağlantı hatası: {e}"
    
# streamlit ile sohbet arayüzü
user_input = st.chat_input("Bir mesaj yazın ... ")

if user_input:
    # kullanıcı mesajını session state ekle
    st.session_state["messages"].append({"role":"user","content":user_input})

    # API ye gönder
    with st.spinner("Ajan düşünüyor ... "):
        response = send_messages_to_api(user_input)

    # ajan cevabını ui ekle
    st.session_state["messages"].append({"role":"assistant","content":response})

# mesajları görüntüle
for msg in st.session_state["messages"]:
    if msg["role"]=="user":
        with st.chat_message("user"):
            st.markdown(f"**Siz:** {msg["content"]}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**Ajan:** {msg["content"]}")

