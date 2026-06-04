"""
Temel bir matematik hesaplayıcı
"""
from langchain.tools import tool # decorator

@tool("calculator")
def calculator(expression: str) -> str: #expressionu string olarak alır ve sonucu string olarak return eder
    """
     Basit metamatiksel ifadeleri değerlendirir
     Örnek: '25 * (5 + 3)' --> 'Answer: 200'
    """
    try:
        result = eval(expression) #eval fonksiyonu verilen string ifadeyi python kodu olarak çalıştırır
        return f"Answer: {result}" # Answer kelimesi react ajanları için önemli çünkü bu şekilde cevabın ne olduğunu anlayabilirler
    except Exception as e:
        return f"Error: {e}" 
    
# langchain formatına uygun hale getirelim
calculator_tool = calculator

#print(calculator_tool.invoke('25 * (3 + 3)'))
