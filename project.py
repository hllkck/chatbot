import os
import streamlit as st 
from dotenv import load_dotenv
from operator import itemgetter
from pathlib import Path
import re 
from langchain_community.document_loaders import TextLoader 
from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.documents import Document 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("ERROR: GEMINI API KEY not found. Please set it in Streamlit Secrets.")
    st.stop()

# ==================================================================
# 0. AYARLAR
# ==================================================================
DATA_PATH = "data/words.txt" # Örnek dosya yolu
CHROMA_DB_DIR = "chroma_db/chroma_db_streamlit" # Streamlit için ayrı bir DB klasörü
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
GENERATION_MODEL = "gemini-2.5-flash" 

# API anahtarını çek
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Streamlit'te hata gösterme
    st.error("ERROR: Failed to load GEMINI API KEY. Check your .env file or Streamlit secrets.")
    st.stop()

# Veri dosyası kontrolü
if not Path(DATA_PATH).exists():
    st.error(f"ERROR: Data file not found: {DATA_PATH}. Please create file 'data/words.txt'.")
    st.stop()

# Sistem Prompt'u (LLM'e rolünü ve görevinin sınırlarını tanımlar)
RAG_SYSTEM_PROMPT = """
Sen, sana verilen verisetindeki kelimeleri analiz eden, İngilizce kelimelerin Türkçe anlamlarını ve Türkçe kelimelerin İngilizce anlamlarını ve örnek cümlelerini sağlayan bir Dil eğitmenisin.

***GÖREV KURALLARI***

1.  **Sorgu Tipi Belirleme:** Kullanıcının sorusunun amacına göre hareket et:
    **Tekil Kelime Sorgusu (Örn: "jacket kelimesi ne demek?"):** Sadece kullanıcının sorduğu İngilizce kelimeyi Bağlam'da ara. Bulursan, Bağlam'daki seviye bilgisini (A1, B2 vb.) kullanarak formatı *sadece o kelime* için uygula ve o kelimeye ait İngilizce örnek cümleler üret.
    **Seviye/Toplu Sorgu (Örn: "A1 kelimelerini ver."):** Bağlam'dan istenen seviyeye uygun (varsa) **minimum 3 kelime** seç ve formatı uygula.

2.  **Cümle Kuralı:** Seçtiğin her İngilizce kelime için, kelimenin farklı kullanım tonlarını gösteren **bağlamdan bağımsız en az 3 farklı doğru İngilizce örnek cümle** kur.
3.  **Format Kuralı:** Cevabını sadece aşağıdaki **Örnek Cevap Formatına** uygun olarak düzenle.

---
Bağlam (Context):
{context}
---
Soru (Question): {input}

***
Örnek Cevap Formatı:

### 1. [İngilizce Kelime]
- **Türkçe Anlamı:** [Anlam]
- **Örnek Cümleler:**
    1. [Cümle 1]
    2. [Cümle 2]
    3. [Cümle 3]
### 2. [Türkçe Kelime]
- **İngilizce Anlamı:** [Anlam]
- **Örnek Cümleler:**
    1. [Cümle 1]
    2. [Cümle 2]
    2. [Cümle 3]
***
"""

# ==================================================================
# 1. RAG FONKSİYONLARI (Streamlit Önbellekleme ile)
# ==================================================================

@st.cache_resource
def index_data(data_file_path: str, db_dir: str):
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Vektör Veritabanı (ChromaDB) Kalıcılık Kontrolü
    if Path(db_dir).exists():
        try:
            vectorstore = Chroma(
                persist_directory=db_dir, 
                embedding_function=embeddings
            )
            # Eğer kayıt varsa, direkt yükle ve bitir.
            if vectorstore._collection.count() > 0:
                st.info(f"Vector database loaded from disk. Total records: {vectorstore._collection.count()}")
                return vectorstore
        except Exception as e:
            # Hata oluşursa (bozuk dosya vs.), yeniden oluşturmaya geç
            st.warning(f"Error loading ChromaDB ({e}). Rebuilding...")
    
    # Eğer diskte yoksa veya bozuksa: Veri Yükleme ve Bölme
    with st.spinner("The data is being loaded, split by line, and vectorized... This may take some time."):
        
        # Veri Yükleyici
        loader = TextLoader(data_file_path, encoding='utf-8')
        raw_documents = loader.load()

        # Satır Bazlı Bölme (Chunking)
        splits = []
        if raw_documents and raw_documents[0].page_content:
            # Her satırı (kelime kaydını) ayrı bir Document olarak ele al
            for i, line in enumerate(raw_documents[0].page_content.splitlines()):
                cleaned_line = line.strip()
                if cleaned_line:
                    splits.append(
                        Document(
                            page_content=cleaned_line,
                            metadata={"source": Path(data_file_path).name, "line": i + 1}
                        )
                    )
        
        if not splits:
            raise ValueError(f"'{data_file_path}' file is empty or unreadable.")
            
        st.write(f"-> {len(raw_documents)} document, {len(splits)} divided into piece.")
        
        # Vektör Veritabanı Oluşturma ve Kaydetme
        st.warning("Vektör database is creating...")
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            persist_directory=db_dir 
        )
        vectorstore.persist() 
        st.success(f"Vector database created and saved in {db_dir} folder. Total records: {vectorstore._collection.count()}")
        
    return vectorstore

@st.cache_resource
def create_rag_chain(_vectorstore: Chroma):
    with st.spinner(f"RAG Chain ({GENERATION_MODEL}) is being established..."):

        retriever = _vectorstore.as_retriever(search_kwargs={"k": 7}) 

        
        llm = ChatGoogleGenerativeAI(
            model=GENERATION_MODEL, 
            temperature=0.7,
            google_api_key=GEMINI_API_KEY
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{input}")
        ])
        
        # LCEL RAG Zinciri Oluşturma
        rag_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("input") | retriever
            )
            | prompt
            | llm
            | StrOutputParser()
        )
    st.success("RAG Chain is installed and ready to use.")
    return rag_chain

# ==================================================================
# 3. STREAMLIT ARAYÜZÜ
# ==================================================================

def main():
    st.set_page_config(page_title="Translation Bot", layout="wide")
    st.title("📚 RAG-Powered Translation Bot")
    st.caption("LangChain (LCEL) + Gemini Flash + ChromaDB")

    # --- RAG Sistemini Kurma ---
    try:
        # ChromaDB'yi diskten yükle veya oluştur
        _vectorstore = index_data(DATA_PATH, CHROMA_DB_DIR) 
        
        rag_chain = create_rag_chain(_vectorstore)
    except Exception as e:
        st.error(f"A critical error occurred during system installation: {e}")
        return 

    # --- Sohbet Geçmişi Yönetimi ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sohbet Geçmişini Göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Kullanıcı Girişi ve Yanıt Üretme ---
    if prompt := st.chat_input("Ask me a word..."):
        # Kullanıcı mesajını kaydet ve göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Asistan yanıtını üret ve göster
        with st.chat_message("assistant"):
            with st.spinner("Searching for and creating answers..."):
                try:
                    input_data = {"input": prompt}
                    result = rag_chain.invoke(input_data)
                    
                    st.markdown(result)
                    
                    # Cevabı geçmişe kaydet
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    error_msg = f"An error occurred while generating the answer: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()

