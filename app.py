import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os
import tempfile
import joblib

# Získání absolutní cesty ke kořenové složce projektu
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from data_processing.pdf_text_extractor import extract_invoice_data
from data_processing.entity_extractor import create_invoice_dataframe
from ml_models.predict_pdf_batch import preprocess_data
from llm_query.query_config import QUERY_CONFIG, process_query
from rag.newsapi_client import TechNewsRAG

# Cesty k modelu a pomocným souborům
MODEL_PATH = os.path.join(PROJECT_ROOT, "ml_models", "xgb_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "ml_models", "scaler.pkl")
ENCODERS_PATH = os.path.join(PROJECT_ROOT, "ml_models", "label_encoders.pkl")
csv_path = os.path.join(PROJECT_ROOT, "utils", "synthetic_project_data.csv")


# Navigace mezi stránkami
st.sidebar.title("Navigace")
page = st.sidebar.radio("Vyberte stránku", ["Úvod", "Načtení a zpracování PDF", "Analytika", "Tech Novinky"])

# Pole pro zadání API klíčů
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<span style="font-size:14px;">Pro <b>Analytiku</b> stačí zadat <b>OpenAI API klíč</b>, pro <b>Tech Novinky</b> zadejte <b>oba klíče</b>.</span>',
    unsafe_allow_html=True
)
openai_key = st.sidebar.text_input("OpenAI API klíč", type="password", key="openai_api_key")
news_key = st.sidebar.text_input("News API klíč", type="password", key="news_api_key")
st.sidebar.markdown(
    '<a href="https://newsapi.org/" target="_blank" rel="noopener noreferrer" style="font-size:16px;">Newsapi.org pro bezplatný API klíč</a>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

try:
    import config
except ImportError:
    config = None

newsapi_key = news_key or (config.NEWSAPI_KEY if config else None)
openai_api_key = openai_key or (config.OPENAI_API_KEY if config else None)

def display_anomalies(df):
    """Zobrazí detekované anomálie s barevným zvýrazněním"""
    st.subheader("Detekované anomálie")

    # Legenda nad tabulkou 
    st.markdown("""
    <style>
    .legend-label {
    font-size: 15px;
    font-weight: 500;
    /* Výchozí barva pro světlý režim */
    color: #222;
    }
    @media (prefers-color-scheme: dark) {
    .legend-label {
        color: #fff !important;
    }
    }
    </style>
    <div style="display: flex; gap: 32px; align-items: center; margin-bottom: 18px; flex-wrap: wrap;">
    <div style="display: flex; align-items: center; gap: 7px;">
        <div style="width: 18px; height: 18px; background: #FA5252; border: 2px solid #FA5252; border-radius: 4px;"></div>
        <span class="legend-label">Vysoká částka + krátká splatnost</span>
    </div>
    <div style="display: flex; align-items: center; gap: 7px;">
        <div style="width: 18px; height: 18px; background: #FFA54C; border: 2px solid #FFA54C; border-radius: 4px;"></div>
        <span class="legend-label">Nesoulad položek + datum</span>
    </div>
    <div style="display: flex; align-items: center; gap: 7px;">
        <div style="width: 18px; height: 18px; background: #57FD57; border: 2px solid #57FD57; border-radius: 4px;"></div>
        <span class="legend-label">Neobvyklý počet položek</span>
    </div>
    <div style="display: flex; align-items: center; gap: 7px;">
        <div style="width: 18px; height: 18px; background: #5B9DFF; border: 2px solid #5B9DFF; border-radius: 4px;"></div>
        <span class="legend-label">Neobvyklá služba</span>
    </div>
    </div>
    """, unsafe_allow_html=True)



    # Převod na číselné typy
    numeric_cols = {
        "total_amount": float,
        "Jistota": float,
    }

    for col, dtype in numeric_cols.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        
    # Definice barevného schématu
    color_palette = {
        "Vysoká částka + krátká splatnost": "#FA5252",
        "Nesoulad položek + datum": "#FFA54C",
        "Neobvyklý počet položek": "#57FD57",
        "Neobvyklá služba": "#5B9DFF"
    }
        
    # Funkce pro aplikaci stylů
    def apply_style(row):
        anomaly_type = row["Typ anomálie"]
        color = color_palette.get(anomaly_type, "#FFFFFF")
        return [f'background-color: {color}'] * len(row)
        
    # Vytvoření stylované tabulky
    styled_df = df.style.apply(apply_style, axis=1)
        
    # Zobrazení tabulky
    st.dataframe(
        styled_df,
        height=min(400, 35 * len(df)),
        use_container_width=True,
        hide_index=True
    )

    # Tlačítko pro stažení
    st.download_button(
        label="Stáhnout detekované anomálie jako CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="detekovane_anomalie.csv",
        mime="text/csv"
    )

# Hlavní logika pro každou stránku
if page == "Úvod":
    st.title("Vítejte v aplikaci FinDoc AI")
    st.write("""
        **Tento projekt představuje funkční aplikaci pro komplexní správu faktur a analýzu dat s integrací AI. Hlavní funkce pokrývají tyto oblasti:**

        - **Načítání a zpracování PDF faktur**
            - Automatická extrakce dat z faktur (dodavatel, částka, datum apod.)
            - Detekce anomálií – identifikace nestandardních transakcí nebo chyb v datech 
            - *Testování:* Pro tuto sekci **stačí stáhnout ukázkový ZIP soubor** s předpřipravenými PDF fakturami
        - **Analytika s podporou LLM**
            - Interaktivní vizualizace s kontextově obohacenými doporučeními
            - *Požadavek:* Pro plné využití je nutné zadat **OpenAI API klíč**
        - **Tech novinky (RAG pipeline)**
            - Vyhledávání aktuálních článků pomocí News API
            - Shrnutí obsahu a odpovědi na dotazy v přirozeném jazyce
            - *Požadavek:* Je vyžadován **News API klíč i OpenAI API klíč**
        """)

    # Tlačítko pro stažení ZIP s PDF fakturami
    zip_path = os.path.join(os.path.dirname(__file__), "PDF.zip")
    with open(zip_path, "rb") as fp:
        st.download_button(
            label="📥 Stáhnout ukázkové PDF faktury (ZIP)",
            data=fp,
            file_name="demo_faktury.zip",
            mime="application/zip"
        )

    st.markdown("---")

    # Rozbalovací okno s README.md
    with st.expander("Zobrazit kompletní dokumentaci (README.md)"):
        readme_content = Path("README.md").read_text(encoding="utf-8")
        st.markdown(readme_content, unsafe_allow_html=True)


if page == "Načtení a zpracování PDF":
    st.title("📄 Načtení a zpracování PDF faktur")
    
    uploaded_files = st.file_uploader(
        "Nahrajte PDF faktury (můžete vybrat více souborů najednou)",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"Načteno {len(uploaded_files)} souborů.")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_paths = []
            for file in uploaded_files:
                file_path = os.path.join(tmpdir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                file_paths.append(file_path)

            ocr_results = []
            progress_bar = st.progress(0)
            for i, path in enumerate(file_paths):
                try:
                    data = extract_invoice_data(path)
                    ocr_results.append(data)
                except Exception as e:
                    st.warning(f"Chyba u {Path(path).name}: {str(e)}")
                progress_bar.progress((i + 1) / len(file_paths))

            if ocr_results:
                df = create_invoice_dataframe(ocr_results)
                st.success("✅ Základní zpracování dokončeno!")
                
                # Zobrazení kompletních dat
                st.write("Kompletní přehled faktur:")
                st.dataframe(df, height=300, use_container_width=True)

                # Tlačítko pro stažení kompletních dat
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="Stáhnout kompletní data jako CSV",
                    data=csv,
                    file_name="kompletni_data.csv",
                    mime="text/csv",
                    key="full_data"
                )

                # Sekce pro predikci anomálií
                st.markdown("---")
                if st.button("🔍 Spustit detekci anomálií", type="primary"):
                    with st.spinner("Analyzuji faktury..."):
                        try:
                            # Načtení modelu a pomocných objektů
                            model = joblib.load(MODEL_PATH)
                            scaler = joblib.load(SCALER_PATH)
                            encoders = joblib.load(ENCODERS_PATH)
                            
                            # Příprava dat
                            df_preprocessed = preprocess_data(df.copy())
                            
                            # Predikce
                            y_pred = model.predict(df_preprocessed)
                            y_proba = model.predict_proba(df_preprocessed).max(axis=1)
                            
                            # Přidání výsledků
                            df["Kód anomálie"] = y_pred
                            df["Jistota"] = y_proba
                            df["Typ anomálie"] = df["Kód anomálie"].map({
                                0: "Vysoká částka + krátká splatnost",
                                1: "Nesoulad položek + datum",
                                2: "Žádná anomálie",
                                3: "Neobvyklý počet položek",
                                4: "Neobvyklá služba"
                            })
                            
                            # Filtrace a zobrazení anomálií
                            df_anomalies = df[df["Kód anomálie"] != 2]
                            
                            if not df_anomalies.empty:
                                display_anomalies(df_anomalies)
                            else:
                                st.success("🎉 Všechny faktury jsou v pořádku, žádné anomálie nebyly detekovány!")
                                
                        except Exception as e:
                            st.error(f"❌ Chyba při analýze: {str(e)}")


elif page == "Analytika":
    if not openai_api_key:
        st.warning("Pro Analytiku zadejte OpenAI API klíč v postranním panelu.")
    else:
        st.title("Analytické přehledy")
        invoices_df = pd.read_csv(csv_path)
        invoices_df['total_amount'] = invoices_df['total_amount'].round().astype(int)

        # Výběr dotazu
        query_options = [config["question"] for config in QUERY_CONFIG.values()]
        selected_question = st.selectbox(
            "Vyberte analytický dotaz:",
            options=query_options,
            index=0
        )
        selected_key = next(
            key for key, config in QUERY_CONFIG.items()
            if config["question"] == selected_question
        )

        # Zpracování dotazu
        result = process_query(selected_key, invoices_df, openai_api_key)

        st.subheader(result["question"])

        # Kontrola, jestli existuje renderer, a pokud ano, použij ho
        config = QUERY_CONFIG[selected_key]
        if "renderer" in config:
            config["renderer"](result["data"])
        else:
            # Fallback pro dotazy bez rendereru
            with st.expander("Zobrazit data"):
                st.dataframe(result["data"])

        # Zobrazení analýzy pouze pro dotazy, které ji negenerují ve svém rendereru
        if selected_key != "payment_distribution":
            st.subheader("Analýza")
            st.write(result["analysis"])

elif page == "Tech Novinky":
    if not newsapi_key or not openai_api_key:
        st.warning("Pro Tech Novinky zadejte oba API klíče v postranním panelu.")
    else:
        st.title("🔍 Technologické novinky")

        # Inicializace RAG systému
        try:
            # Předání obou API klíčů při inicializaci
            rag = TechNewsRAG(newsapi_key=newsapi_key, openai_api_key=openai_api_key)
            st.success("✅ Systém úspěšně inicializován!")
        except Exception as e:
            st.error(f"❌ Chyba při inicializaci: {str(e)}")
            st.stop()

        # Hlavní funkcionalita
        query = st.text_input("Zadejte dotaz v přirozeném jazyce:", "")

        if st.button("Souhrn"):
            with st.spinner("🔍 Vyhledávám relevantní články a generuji odpověď..."):
                try:
                    # Získání odpovědi a výsledků vyhledávání
                    answer, results = rag.query(query)
                        
                    st.markdown("---")
                    st.write("📝 Souhrn:")
                    st.markdown(answer)
                        
                    # Zobrazení zdrojových článků
                    st.markdown("---")
                    st.write("📰 Zdrojové články:")
                        
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

