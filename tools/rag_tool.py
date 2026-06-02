"""
RAG: Bilgi getirme, müsteri_destek.pdf dosyasından bilgi çekmek için kullanılacak tool.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter # chunking işlemi için
from langchain_classic.chains import ConversationalRetrievalChain # zincir oluşturmak için
from langchain_core.tools import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_rag_tool(pdf_path: str, llm):
    """
    Belirtilen PDF dosyasını yükler, içeriği vektörleştirir ve bir RAG tool'u oluşturur.
    """
    # PDF dosyasını yükler
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Belirtilen metin parçacıklarına böler
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Vektör deposunu oluşturur
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # en benzer 3 sonucu getir

    # RAG zincirini oluşturur
    rag_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=False)

    

    def rag_query(query:str):
        """
        Soruya göre en benzer içeriği bulur ve yanıt üretir
        """
        response = rag_chain.run("question: " + query,"chat_history: []") # chat_history boş çünkü her seferinde yeni bir sorgu yapacağız
        return f"RAG Tool Cevabı: {response}"
    
    return Tool(
        name="rag_tool",
        func=rag_query,
        description="Bu tool, müsteri_destek.pdf dosyasından bilgi çekmek için kullanılır. Kullanıcı bir soru sorduğunda, bu tool PDF içeriğinden en benzer bilgiyi bulur ve yanıt üretir." 
    )

"""
if __name__ == "__main__":
    print("RAG tool dosyası başarıyla yüklendi.")

"""