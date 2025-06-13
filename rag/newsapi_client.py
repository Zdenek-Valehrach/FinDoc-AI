import os
import requests
from datetime import datetime, timedelta
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS  
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from pathlib import Path
import sys
import os

# Nastavení cest
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class TechNewsRAG:
    """Univerzální třída pro technologická média s flexibilními API klíči"""
    
    def __init__(self, newsapi_key: str, openai_api_key: str):
        """
        Parameters:
        newsapi_key (str): API klíč pro NewsAPI (z UI/config.py)
        openai_api_key (str): API klíč pro OpenAI (z UI/config.py)
        """
        self.newsapi_key = newsapi_key
        self.openai_api_key = openai_api_key
        self._init_models()
        self.vectorstore = None  # Bude inicializováno při _refresh_data
        self._refresh_data()

    def _init_models(self):
        """Inicializuje LLM a embedding modely"""
        self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=self.openai_api_key)

    def _refresh_data(self):
        """Aktualizuje data z obou zdrojů"""
        
        # Načtení českých i zahraničních článků
        docs_cz = self.fetch_news(language="cs", query="technologie OR AI OR umělá inteligence")
        docs_int = self.fetch_news(language="en", query="technology OR AI OR artificial intelligence")
        
        # Kombinace článků
        combined_docs = docs_cz + docs_int
        
        if combined_docs:
            # Vytvoření FAISS vektorového úložiště
            self.vectorstore = FAISS.from_documents(combined_docs, self.embeddings)
        else:
            # Prázdné úložiště pro případ, že nejsou žádné články
            self.vectorstore = FAISS.from_texts(["Žádné články nenalezeny"], self.embeddings)

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
        
        if not self.vectorstore:
            return "Nenalezeny žádné relevantní články v češtině ani angličtině.", []

        # Získání relevantních dokumentů pomocí similarity_search
        relevant_docs = self.vectorstore.similarity_search(user_input, k=10)
        
        if not relevant_docs:
            return "Nenalezeny žádné relevantní články v češtině ani angličtině.", []
        
        context = self._build_context(relevant_docs)
        answer = self._generate_answer(user_input, context)
        
        # Formátování výsledků pro zobrazení
        results = {
            "documents": [[doc.page_content for doc in relevant_docs]],
            "metadatas": [[doc.metadata for doc in relevant_docs]],
            "distances": [[0.0] * len(relevant_docs)]  
        }
        
        return answer, results

    def _build_context(self, docs):
        """Vytvoří multijazyčný kontext z dokumentů"""
        return "\n\n".join([
            f"Jazyk: {doc.metadata['language']}\nČlánek: {doc.page_content}\nZdroj: {doc.metadata['source']}\nURL: {doc.metadata['url']}"
            for doc in docs
        ])

    def _generate_answer(self, query: str, context: str):
        """Generuje univerzální odpověď"""
        prompt_template = """
        Jsi expert na technologické novinky. Vypracuj souhrn na základě následujícího kontextu, 
        který obsahuje články v češtině i angličtině. Odpověď poskytni v jazyce dotazu.

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

