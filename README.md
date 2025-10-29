📚 Dil Eğitmeni Chatbotu

Bu proje, LangChain ve Google Gemini modelini kullanarak kullanıcıların özel bir kelime listesi (veri seti) üzerinden anlam ve örnek cümle alabileceği interaktif bir RAG (Retrieval-Augmented Generation) uygulamasıdır.
Arayüz için Streamlit kullanılmıştır.

🧠 Proje Özeti

Mevcut büyük dil modellerinin (LLM) genel bilgisini, kendi özel kelime veri setimdeki anlam gibi spesifik bilgilerle birleştirerek, doğru ve yapılandırılmış cevaplar üreten bir bot geliştirdim.

⚙️ Çalışma Prensibi

🔹 Veri Seti

Kelime, anlam ve seviye bilgilerini içeren özel bir kelime listesi kullanılır.

🔹 Indexing

Veri setindeki her satır, ChromaDB vektör veritabanına bağımsız bir parça (document) olarak kaydedilir.
Bu işlem yalnızca ilk çalıştırmada yapılır ve veriler diskte kalıcı olarak saklanır.

🔹 Retrieval

Kullanıcı bir kelime sorduğunda, Retriever (bilgi çekici), ChromaDB’den sorguya en alakalı kelime kayıtlarını çeker.

🔹 Generation

Çekilen kelime kaydı (context) ve kullanıcı sorgusu, Gemini (via LangChain LCEL) modeline gönderilir.
Model, belirlenen prompt kurallarına uygun şekilde:

Kelimenin anlamını,

Üç farklı örnek cümleyi
üreterek yapılandırılmış bir çıktı oluşturur.

📄 Veri Seti Formatı

Veri seti satır bazlı olmalıdır.
Her satır, kelime, anlam ve isteğe bağlı seviye bilgisini içermelidir.

Örnek:

A1
complete: tamamlamak [f.]  tam [s.]  bütün [s.]

B2
affordable: bütçeye uygun [s.]  ekonomik [s.]  düşük maliyetli [s.]

🛠️ Kullanılan Teknolojiler

LLM (Büyük Dil Modeli), Google Gemini Flash, LangChain Expression Language (LCEL), ChromaDB, Streamlit

🚀 Kurulum ve Çalıştırma

Projeyi çalıştırmak için temel Python ortamını hazırlamanız ve API ayarlarını yapmanız gerekir.

1️⃣ Ön Koşullar

Python 3.11

Google Gemini API Anahtarı

2️⃣ Adımlar
🔹 Bağımlılıkları Yükleyin
pip install -r requirements.txt

🔹 API Anahtarını Ayarlayın

Projenin ana dizininde .env dosyası oluşturun ve aşağıdaki satırı ekleyin:

GEMINI_API_KEY="AIzaSy...SizinGercekAPIAnahtarınız"

🔹 Veri Setini Hazırlayın

Her kelime ve anlam kaydını tek satır olacak şekilde data/words.txt dosyasına ekleyin:

procrastinate: ertelemek, ağırdan almak
jacket: ceket

🔹 Uygulamayı Başlatın
streamlit run app.py

💡 Gelecek Geliştirmeler

Seviye bilgisinin otomatik algılanması

Çoklu dil desteği

Türkçe–İngilizce çeviri için iyileştirmeler


## 🚀 Demo
Projeyi canlı olarak deneyin: [Dil Eğitmeni Chatbotu](https://chatbot-v1-0.streamlit.app/)

