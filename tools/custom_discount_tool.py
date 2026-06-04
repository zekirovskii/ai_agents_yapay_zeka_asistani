"""
ürün fiyatını alır ve %10 indirim uygular
"""
from langchain.tools import tool # decorator
import re

@tool("discount_calculator")
def discount_calculator(product_info: str) -> str:
    """
    Ürün fiyatını alır ve %10 indirim uygular
    """
    try: 
        price = float(re.findall(r'\d+\.?\d*', product_info)[0]) # metinden sayısal fiyat bilgisi çekilir
        discounted_price = price * 0.9 # %10 indirim uygulanır
        return f"İndirimli Fiyat: {discounted_price:.2f}" 
    except Exception as e:
        return f"Hata: Fiyat bulunamadı. {e}"
    
