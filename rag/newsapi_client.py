import os
import requests
from datetime import datetime, timedelta
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chromadb.config import Settings
import chromadb

from pathlib import Path
import sys
import os

# Nastavení cest
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import config

try:
    import config
except ImportError:
    config = None

# Kombinace zdrojů pro API klíče
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY") or getattr(config, "NEWSAPI_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getattr(config, "OPENAI_API_KEY", None)

if not NEWSAPI_KEY:
    raise ValueError("❌ NEWSAPI_KEY není nastaven. Nastavte ho v config.py nebo v aplikaci.")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY není nastaven. Nastavte ho v config.py nebo v aplikaci.")


class TechNewsRAG:
    """Univerzální třída pro česká i zahraniční technologická média"""
    
    def __init__(self):
        self.newsapi_key = config.NEWSAPI_KEY
        self._init_chroma()
        self._init_models()
        self._refresh_data()

    def _init_chroma(self):
        """Inicializuje ChromaDB klienta a kolekci"""
        self.admin_client = chromadb.AdminClient(
            Settings(persist_directory="chroma_storage", anonymized_telemetry=False)
        )
        self._init_tenant()
        
        self.client = chromadb.PersistentClient(
            path="chroma_storage",
            settings=Settings(persist_directory="chroma_storage", anonymized_telemetry=False),
            tenant="default_tenant",
            database="default_database"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="tech_news_global",
            embedding_function=chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                api_key=config.OPENAI_API_KEY,
                model_name="text-embedding-3-small"
            ),
            metadata={"hnsw:M": 16, "hnsw:construction_ef": 100, "hnsw:space": "cosine"}
        )

    def _init_tenant(self):
        """Vytvoří tenant a databázi pro Chroma"""
        try:
            self.admin_client.create_tenant("default_tenant")
            self.admin_client.create_database("default_database", "default_tenant")
        except Exception:
            pass

    def _init_models(self):
        """Inicializuje LLM a embedding modely"""
        self.embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=config.OPENAI_API_KEY)

    def _refresh_data(self):
        """Aktualizuje data z obou zdrojů"""
        self.collection.delete(where={"source": {"$ne": ""}})
        
        # Načtení českých i zahraničních článků
        docs_cz = self.fetch_news(language="cs", query="technologie OR AI OR umělá inteligence")
        docs_int = self.fetch_news(language="en", query="technology OR AI OR artificial intelligence")
        
        if docs_cz + docs_int:
            self.collection.upsert(
                ids=[str(i) for i in range(len(docs_cz + docs_int))],
                documents=[doc.page_content for doc in docs_cz + docs_int],
                metadatas=[doc.metadata for doc in docs_cz + docs_int]
            )

    def fetch_news(self, language: str, query: str):
        """Získává články pro zadaný jazyk"""
        domains = {
            "cs": "technet.idnes.cz,zive.cz,root.cz,lupa.cz,cnews.cz,cc.cz,chip.cz,itbiz.cz",
            "en": "techcrunch.com,theverge.com,wired.com,engadget.com,arstechnica.com"
        }.get(language, "")
        
        params = {
            "q": query,
            "domains": domains,
            "language": language,
            "sortBy": "relevancy",
            "from": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "pageSize": 100,
            "apiKey": self.newsapi_key
        }
        
        try:
            response = requests.get("https://newsapi.org/v2/everything", params=params)
            return self._process_articles(response.json().get('articles', []), language)
        except Exception:
            return []

    def _process_articles(self, articles, language: str):
        """Zpracuje články s ohledem na jazyk"""
        processed = []
        for art in articles:
            if not art.get('title') or not self._is_valid_source(art['url'], language):
                continue
                
            processed.append(Document(
                page_content=f"{art['title']}\n{art.get('description','')}",
                metadata={
                    "source": art['source']['name'],
                    "date": art['publishedAt'],
                    "url": art['url'],
                    "language": language
                }
            ))
        return processed

    def _is_valid_source(self, url: str, language: str):
        """Validuje domény podle jazyka"""
        domains = {
            "cs": ["technet.idnes.cz", "zive.cz", "root.cz", "lupa.cz", "cnews.cz", "cc.cz", "chip.cz", "itbiz.cz"],
            "en": ["techcrunch.com", "theverge.com", "wired.com", "engadget.com", "arstechnica.com"]
        }
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower().replace("www.", "")
        return any(d in domain for d in domains.get(language, []))

    def query(self, user_input: str):
        """Zpracuje dotaz včetně obou jazykových verzí"""
        self._refresh_data()
        
        if self.collection.count() == 0:
            return "Nenalezeny žádné relevantní články v češtině ani angličtině.", []

        results = self.collection.query(
            query_texts=[user_input],
            n_results=min(10, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        context = self._build_context(results)
        answer = self._generate_answer(user_input, context)
        
        return answer, results

    def _build_context(self, results):
        """Vytvoří multijazyčný kontext"""
        return "\n\n".join([
            f"Jazyk: {meta['language']}\nČlánek: {doc}\nZdroj: {meta['source']}\nURL: {meta['url']}"
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ])

    def _generate_answer(self, query: str, context: str):
        """Generuje univerzální odpověď"""
        prompt_template = """
        Jsi expert na technologické novinky. Vypracuj souhrn na základě následujícího kontextu, 
        který obsahuje články v češtině i angličtině. Odpověď poskytni v jazyce původního dotazu.

        Dotaz: {question}

        Kontext:
        {context}

        Formát odpovědi:
        1. Souhrn hlavních témat včetně českých i mezinárodních zdrojů
        2. Konkrétní příklady z obou jazykových verzí
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"question": query, "context": context})

    def _is_low_confidence(self, results, threshold=0.3):
        """Detekuje nízkou relevanci výsledků"""
        return (not results.get('distances') or 
                (max(results['distances'][0]) < threshold if results['distances'][0] else True))
    
