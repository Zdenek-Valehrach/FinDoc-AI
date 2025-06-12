import streamlit as st
from newsapi_client import TechNewsRAG

"""
Samostatná Streamlit aplikace pro testování dotazů.
Poskytuje stejnou funkcionalitu jako v hlavní aplikaci, 
ale může být spuštěna nezávisle pro rychlejší vývoj a testování.

Spuštění: streamlit run rag/news_app_standalone.py
"""

# Konfigurace aplikace
st.set_page_config(page_title="Tech News Analyzátor", layout="wide")
st.title("🔍 Analýza technologických novinek")

# Inicializace RAG systému
try:
    rag = TechNewsRAG()
    st.success("✅ Systém úspěšně inicializován!")
except Exception as e:
    st.error(f"❌ Chyba při inicializaci: {str(e)}")
    st.stop()

# Hlavní funkcionalita
query = st.text_input("Zadejte dotaz v přirozeném jazyce:", "")

if st.button("Analyzovat"):
    with st.spinner("🔍 Vyhledávám relevantní články a generuji odpověď..."):
        try:
            # Získání odpovědi a výsledků vyhledávání
            answer, results = rag.query(query)
            
            st.markdown("---")
            st.subheader("📝 Výsledky analýzy")
            st.markdown(answer)
            
            # Zobrazení zdrojových článků
            st.markdown("---")
            st.subheader("📰 Zdrojové články")
            
            if results and len(results['documents']) > 0:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    with st.expander(doc.split("\n")[0]):
                        st.markdown(f"**Zdroj:** {meta['source']}")
                        st.markdown(f"**Datum:** {meta['date']}")
                        st.markdown(f"**URL:** {meta['url']}")
            else:
                st.warning("Nebyly nalezeny žádné relevantní články.")
                
        except Exception as e:
            st.error(f"❌ Chyba při zpracování: {str(e)}")
