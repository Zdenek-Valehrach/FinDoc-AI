import streamlit as st
import pandas as pd
from query_config import QUERY_CONFIG, process_query
from pathlib import Path
import sys
import os


"""
Samostatná Streamlit aplikace pro testování analytických dotazů.
Poskytuje stejnou funkcionalitu jako analytická sekce v hlavní aplikaci, 
ale může být spuštěna nezávisle pro rychlejší vývoj a testování.

Spuštění: streamlit run llm_query/query_app_standalone.py
"""


# Nastavení cest
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import config

# Explicitní nastavení API klíče
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# Cesty k souborům
script_dir = Path(__file__).parent.resolve()
csv_path = script_dir / 'synthetic_project_data.csv'

def main():
    st.title("Analýza faktur")
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
    result = process_query(selected_key, invoices_df)

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

if __name__ == "__main__":
    main()
