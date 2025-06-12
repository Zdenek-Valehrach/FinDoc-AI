import pandas as pd
import plotly.express as px
from openai import OpenAI
import streamlit as st
import inspect
from pathlib import Path
import sys
import os

# Nastavení cest
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import config

# Explicitní nastavení API klíče
try:
    import config
except ImportError:
    config = None

# Kombinace zdrojů pro API klíč
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getattr(config, "OPENAI_API_KEY", None)

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY není nastaven. Nastavte ho v config.py nebo v aplikaci.")

# Cesty k souborům
project_root = Path(__file__).parent.parent
csv_path = project_root / 'utils' / 'synthetic_project_data.csv'

# Načtení dat
df = pd.read_csv(csv_path)

# client pro OpenAI API
client = OpenAI(api_key=OPENAI_API_KEY)

# Globální formátovací funkce
def format_czk(value):
    return f"{int(round(value)):,}".replace(",", " ") + ",- Kč"

CZECH_MONTHS = {
    1: 'Leden', 2: 'Únor', 3: 'Březen', 4: 'Duben',
    5: 'Květen', 6: 'Červen', 7: 'Červenec', 8: 'Srpen',
    9: 'Září', 10: 'Říjen', 11: 'Listopad', 12: 'Prosinec'
}

CZECH_MONTHS_ORDER = [
    "Leden 2024", "Únor 2024", "Březen 2024", "Duben 2024", "Květen 2024", "Červen 2024",
    "Červenec 2024", "Srpen 2024", "Září 2024", "Říjen 2024", "Listopad 2024", "Prosinec 2024"
]

def set_month_order(df):
    df['month'] = pd.Categorical(df['month'], categories=CZECH_MONTHS_ORDER, ordered=True)
    return df.sort_values('month')

# --- Definice dotazů ---
QUERY_CONFIG = {
    "monthly_cashflow": {
        "question": "Měsíční cashflow",
        "agg_func": lambda df: (
            df.assign(month=pd.to_datetime(df['invoice_date']).dt.to_period('M').astype(str))
            .groupby(['month', 'transaction_type'])['total_amount']
            .sum()
            .unstack(fill_value=0)
            .reset_index()
            .assign(month=lambda x: x['month'].apply(
                      lambda m: f"{CZECH_MONTHS[int(m.split('-')[1])]} {m.split('-')[0]}"
                  ))
            .astype({'Příjmy': int, 'Výdaje': int})
        ),
        "format_func": lambda data: set_month_order(
            data.assign(
                Příjmy_formatted=data['Příjmy'].apply(format_czk),
                Výdaje_formatted=data['Výdaje'].apply(format_czk),
                Bilance=data['Příjmy'] - data['Výdaje']
            )
        ),
        "prompt_func": lambda data: f"""
            Analyzuj měsíční cashflow na základě těchto dat:
            {data[['month', 'Příjmy_formatted', 'Výdaje_formatted']].to_markdown(index=False)}
            
            Číselné hodnoty pro výpočty:
            - Celkové příjmy: {data['Příjmy'].sum():,}
            - Celkové výdaje: {data['Výdaje'].sum():,}
            - Bilance: {data['Bilance'].sum():,}
            
            Výstup formátuj s použitím českého číselného formátu (30 336 677,- Kč).
            Výstup: 
            1. Celkové cashflow a trendy
            2. Identifikace 3 nejlepších a nejhorších měsíců
            3. Doporučení pro optimalizaci
        """,
        "renderer": lambda data: (
            st.line_chart(data.set_index('month')[['Příjmy', 'Výdaje']]),
            st.write("**Data:**"),
            st.dataframe(
                data[['month', 'Příjmy_formatted', 'Výdaje_formatted']]
                .rename(columns={
                    'Příjmy_formatted': 'Příjmy',
                    'Výdaje_formatted': 'Výdaje'
                })
            ),
            st.write(f"**Celková bilance:** {format_czk(data['Bilance'].sum())}")
        ),
    },
    "top_customers": {
        "question": "Top 5 odběratelů podle objemu příjmů",
        "agg_func": lambda df: (
            df[df['transaction_type'] == 'Příjmy']
            .groupby('customer_name')['total_amount']
            .sum()
            .nlargest(5)
            .reset_index()
            .assign(
                podil=lambda x: (x['total_amount'] / x['total_amount'].sum() * 100).round(1)
            )
        ),
        "format_func": lambda data: (
            data.assign(
                total_amount_formatted=data['total_amount'].apply(format_czk),
                podil_formatted=data['podil'].apply(lambda x: f"{x} %")
            )
        ),
        "prompt_func": lambda data: f"""
            Analyzuj největší odběratele podle těchto dat:
            {data[['customer_name', 'total_amount_formatted', 'podil_formatted']].to_markdown(index=False)}
            
            Číselné hodnoty pro výpočty:
            - Celkové příjmy od těchto odběratelů: {data['total_amount'].sum():,.0f} Kč
            - Průměrný podíl na příjmech: {data['podil'].mean():.1f} %
            
            Výstup formátuj jako:
            1. Shrnutí významu těchto odběratelů
            2. Doporučení pro další spolupráci
            3. Rizika přílišné závislosti
            """,
        "renderer": lambda data: (
            fig := px.bar(
                data,
                x='customer_name',
                y='total_amount',
                text='total_amount_formatted',
                color='customer_name',
                color_discrete_sequence=px.colors.qualitative.Set3
            ),
            fig.update_traces(          # Zajistí, že text bude mimo sloupec
                textposition='outside',
                cliponaxis=False
            ),
            fig.update_xaxes(
                tickangle=45,
                automargin=True
            ),
            fig.update_layout(
                showlegend=False,
                yaxis_title="",
                xaxis_title=""
            ),
            fig.update_traces(textposition='outside'),
            st.plotly_chart(fig, use_container_width=True),
            st.write("**Data:**"),
            st.dataframe(
                data[['customer_name', 'total_amount_formatted', 'podil_formatted']]
                .rename(columns={
                    'customer_name': 'Odběratel',
                    'total_amount_formatted': 'Celková částka',
                    'podil_formatted': 'Podíl na příjmech'
                })
            )
        )
    },
    "expense_by_category": {
        "question": "Rozložení výdajů podle kategorií",
        "agg_func": lambda df: (
            df[df['transaction_type'] == 'Výdaje']
            .groupby('category', as_index=False)['total_amount']
            .sum()
            .assign(
                podil=lambda x: (x['total_amount'] / x['total_amount'].sum() * 100).round(1)
            )
        ),
        "format_func": lambda data: (
            data.assign(
                total_amount_formatted=data['total_amount'].apply(format_czk),
                podil_formatted=data['podil'].apply(lambda x: f"{x} %")
            )
        ),
        "prompt_func": lambda data: f"""
            Analyzuj rozložení výdajů podle kategorií:
            {data[['category', 'total_amount_formatted', 'podil_formatted']].to_markdown(index=False)}

            Číselné hodnoty pro výpočty:
            - Celkové výdaje: {data['total_amount'].sum():,} Kč

            Výstup formátuj s použitím českého číselného formátu (30 336 677,- Kč).
            Výstup formátuj jako:
            1. Shrnutí hlavních kategorií výdajů
            2. Doporučení na optimalizaci největších položek
            3. Identifikace kategorií s nejvyšším podílem
        """,
        "renderer": lambda data: (
            st.plotly_chart(
                px.treemap(
                    data,
                    path=['category'],
                    values='total_amount',
                    color='total_amount',
                    color_continuous_scale='Blues',
                    title='Rozložení výdajů podle kategorií'
                ),
                use_container_width=True
            ),
            st.write("**Data:**"),
            st.dataframe(
                data[['category', 'total_amount_formatted', 'podil_formatted']]
                .rename(columns={
                    'category': 'Kategorie',
                    'total_amount_formatted': 'Celková částka',
                    'podil_formatted': 'Podíl na výdajích'
                })
            )
        )
    },
    "payment_distribution": {
        "question": "Distribuce splatností podle typu faktury",
        "agg_func": lambda df: (
            df.assign(
                delay_bucket=pd.cut(
                    df['delay_days'],
                    bins=[-1, 0, 14, 30, 60, float('inf')],
                    labels=['V termínu', '1-14 dní', '15-30 dní', '31-60 dní', '60+ dní'],
                    ordered=False
                )
            )
            .groupby(['transaction_type', 'delay_bucket'], observed=False)
            .agg(
                count=('invoice_id', 'count'),
                avg_delay=('delay_days', 'mean')
            )
            .assign(
                total=lambda x: x.groupby('transaction_type')['count'].transform('sum')
            )
            .reset_index()
        ),
        "format_func": lambda data: (
            data
            # Výpočet celkového počtu pro KAŽDÝ TYP zvlášť
            .assign(
                total_per_type=lambda x: x.groupby('transaction_type')['count'].transform('sum')
            )
            # Výpočet procent pro KAŽDÝ ŘÁDEK
            .assign(
                percentage=lambda x: (x['count'] / x['total_per_type'] * 100).round(1)
            )
            # Formátování
            .assign(
                count_formatted=lambda x: x['count'].apply(lambda v: f"{v} ks"),
                percentage_formatted=lambda x: x['percentage'].apply(lambda v: f"{v} %"),
                avg_delay_formatted=lambda x: x['avg_delay'].apply(lambda v: f"{v:.1f} dní")
            )
            # Odstranění pomocného sloupce
            .drop(columns=['total_per_type'])
        ),
        "prompt_func": lambda data, typ: f"""
            Analyzuj distribuci splatností faktur pro {typ.lower()}:
            {data[data['transaction_type'] == typ][['delay_bucket', 'count_formatted', 'percentage_formatted', 'avg_delay_formatted']].to_markdown(index=False)}
            
            Výstup formátuj jako:
            1. Shrnutí platební morálky pro {typ.lower()}
            2. Riziková období
            3. Doporučení pro zlepšení
        """,
        "renderer": lambda data: (
            # Výběr typu faktury
            (typ := st.radio(
                "Vyberte typ faktur:",
                options=["Příjmy", "Výdaje"],
                index=0,
                horizontal=True
            )),
            # Filtrování dat podle typu
            (filtered_data := data[data['transaction_type'] == typ]),
            # Graf
            st.plotly_chart(
                px.pie(
                    filtered_data,
                    names='delay_bucket',
                    values='count',
                    hole=0.3,
                    color_discrete_sequence=['#00FF00', '#FFA500', '#FF6347', '#8B0000', '#4B0082'],
                    category_orders={'delay_bucket': ['V termínu', '1-14 dní', '15-30 dní', '31-60 dní', '60+ dní']},
                    title=f'Distribuce splatností – {typ}'
                ),
                use_container_width=True
            ),
            # Tabulka
            st.write("**Podrobná data:**"),
            st.dataframe(
                filtered_data[['delay_bucket', 'count_formatted', 'avg_delay_formatted']]
                .rename(columns={
                    'delay_bucket': 'Splatnost',
                    'count_formatted': 'Počet faktur',
                    'avg_delay_formatted': 'Průměrné zpoždění'
                })
            ),
            # Analýza se generuje dynamicky podle volby
            st.subheader("Analýza"),
            st.write(
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "user",
                        "content": f"""
                            Analyzuj distribuci splatností faktur pro {typ.lower()}:
                            {filtered_data[['delay_bucket', 'count_formatted', 'percentage_formatted', 'avg_delay_formatted']].to_markdown(index=False)}
                            
                            Výstup formátuj jako:
                            1. Shrnutí platební morálky pro {typ.lower()}
                            2. Riziková období
                            3. Doporučení pro zlepšení
                        """
                    }],
                    temperature=0,
                    max_tokens=500
                ).choices[0].message.content
            )
        )
    },
    "anomaly_analysis": {
        "question": "Analýza anomálií ve fakturách",
        "agg_func": lambda df: (
            df[df['is_anomaly'] == True]
            .groupby('anomaly_type')
            .agg(
                count=('invoice_id', 'count'),
                total_amount=('total_amount', 'sum')
            )
            .reset_index()
            .sort_values('count', ascending=False)
        ),
        "format_func": lambda data: (
            data.assign(
                total_amount_formatted=data['total_amount'].apply(format_czk),
                count_formatted=data['count'].apply(lambda x: f"{x} ks")
            )
        ),
        "prompt_func": lambda data: f"""
            Analyzuj výskyt anomálií ve fakturách:
            {data[['anomaly_type', 'count_formatted', 'total_amount_formatted']].to_markdown(index=False)}

            Výstup formátuj jako:
            1. Nejčastější typy anomálií a jejich dopad
            2. Doporučení pro prevenci a kontrolu
            3. Identifikace nejrizikovějších oblastí
        """,
        "renderer": lambda data: (
            st.plotly_chart(
                px.bar(
                    data,
                    x='anomaly_type',
                    y='count',
                    text='count_formatted',
                    title='Počet anomálií podle typu',
                    color='count',
                    color_continuous_scale='Reds'
                ),
                use_container_width=True
            ),
            st.write("**Podrobná data:**"),
            st.dataframe(
                data[['anomaly_type', 'count_formatted', 'total_amount_formatted']]
                .rename(columns={
                    'anomaly_type': 'Typ anomálie',
                    'count_formatted': 'Počet výskytů',
                    'total_amount_formatted': 'Celková částka'
                })
            )
        )
    }
}

def process_query(query_key: str, df: pd.DataFrame, typ: str = None) -> dict:
    config = QUERY_CONFIG[query_key]
    
    # 1. Zpracování dat
    result_data = config["agg_func"](df)
    formatted_data = config["format_func"](result_data)
    
    # 2. Dynamické volání prompt_func
    sig = inspect.signature(config["prompt_func"])
    params = sig.parameters
    
    if len(params) == 2:  # Pro dotazy s parametrem typ
        if typ is None:
            typ = "Příjmy"  # Výchozí hodnota
        prompt = config["prompt_func"](formatted_data, typ)
    else:  # Pro ostatní dotazy
        prompt = config["prompt_func"](formatted_data)
    
    # 3. Volání LLM
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )
    
    return {
        "question": config["question"],
        "data": formatted_data,
        "analysis": response.choices[0].message.content
    }