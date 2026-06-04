"""
Yapay zeka ajanımızı kullanmak için istemci oluştur
"""

import requests
import json

API_URL = "http://127.0.0.1:8000/ask"

# user_id tanımla, aynı kullanıcıy sorgu atmak icin
USER_ID = "ysf"

def send_message(message: str):
    payload = {
        "user_id": USER_ID,
        "message": message
    }

    try: 
        response = requests.post(API_URL,json=payload)
        response.raise_for_status() # hata varsa
        data = response.json()
        print(f"Soru : {message}")
        print(f"Cevap: {data.get(response)}")
    except Exception as e:
        print(f"Hata Oluştu {e}")

if __name__ == "__main__":
    print("AI Ajanınız başladı.")

    while True:
        user_input = input("Siz: ")
        send_message(user_input)