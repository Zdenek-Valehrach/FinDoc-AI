# FinDoc AI – Komplexní systém pro správu faktur s AI asistencí

Tento projekt představuje end-to-end řešení pro zpracování fakturačních dat pomocí moderních AI technik. Od extrakce dat z PDF faktur, přes detekci anomálií pomocí strojového učení, až po analytické dotazování v přirozeném jazyce a RAG pipeline pro technologické novinky.

## O projektu

FinDoc AI demonstruje kompletní workflow při práci s fakturačními daty. Projekt je zaměřený na simulaci podnikového prostředí, nikoliv na práci s reálnými daty – veškerá data jsou syntetická a slouží výhradně k prezentaci a testování konceptů. Data pokrývají období roku 2024, zahrnující simulované vzory výdajů a příjmů v různých kategoriích (IT, Energie, Mzdy, Doprava, Služby) a cíleně vložených anomálií (cca 9% faktur).

Jakákoliv podobnost s reálnými firmami v syntetických datech je čistě náhodná. Systém je navržen jako praktická ukázka moderních data science a AI workflow ve finančním kontextu.

Projekt kombinuje několik klíčových komponent:
- **Extrakce textu z PDF faktur** s transformací do strukturovaných dat
- **Detekce anomálií** pomocí XGBoost modelu
- **Analytické dotazy** s využitím LLM pro interpretaci výsledků
- **RAG pipeline** pro získávání a analýzu technologických novinek

---

## 📂 Struktura projektu

```

app.py                      
config.py                   
data_processing/
├─ __init__.py
├─ pdf_text_extractor.py            
├─ entity_extractor.py             
└─ batch_processor.py           
llm_query/
├─ __init__.py
├─ query_app_standalone.py     
└─ query_config.py          
ml_models/
├─ __init__.py
├─ model.py                 
└─ predict_pdf_batch.py     
rag/
├─ __init__.py
├─ news_app_standalone.py         
└─ newsapi_client.py        
utils/
├─ __init__.py
└─ synthetic_data.ipynb                       

```


### Poznámka ke standalone aplikacím

Soubory `query_app_standalone.py` a `news_app_standalone.py` jsou samostatné Streamlit aplikace, které umožňují testovat a vyvíjet jednotlivé komponenty projektu nezávisle na hlavní aplikaci. Tyto samostatné verze byly použity během vývoje pro iterativní testování funkcionalit před jejich integrací do hlavní aplikace. Můžete je spustit přímo pro izolované testování konkrétního modulu:

```bash
# Spuštění samostatné verze analytického modulu
streamlit run llm_query/query_app_standalone.py

# Spuštění samostatné verze RAG pipeline pro technologické novinky
streamlit run rag/news_app_standalone.py
```

### Dávkové zpracování PDF faktur

Pro zpracování většího množství PDF faktur mimo interaktivní rozhraní, je možné využít skript z příkazové řádky `batch_processor.py`, který umožňuje efektivně zpracovat celou složku s fakturami najednou:

```bash
# Z kořenového adresáře projektu
python -m data_processing.batch_processor --pdf_dir "/cesta/k/pdf/fakturám"

# Nebo s vlastním názvem výstupního souboru
python -m data_processing.batch_processor --pdf_dir "/cesta/k/pdf/fakturám" --output "moje_faktury.csv"
```
---

## Co projekt umí

- **Generuje syntetická data o fakturách** – s měsíční distribucí výdajů a příjmů v kategoriích (IT, Energie, Mzdy, Doprava, Služby, SaaS, Konzultace, Licence) a cíleně vloženými anomáliemi (9% faktur).
- **Vytváří PDF faktury ze syntetických dat** – včetně plně strukturovaných položek, dodavatelských údajů a částek s použitím ReportLab.
- **Extrahuje data z PDF faktur** – pomocí PyPDF s rozpoznáním strukturovaných informací z textové vrstvy dokumentů
- **Detekuje anomálie pomocí XGBoost modelu** – identifikace nestandardních vzorů jako vysoké částky s krátkou splatností nebo neobvyklé služby.
- **Poskytuje analytické přehledy s LLM interpretací** – cashflow analýza, kategorizace výdajů a příjmů s vysvětleními a doporučeními v přirozeném jazyce.
- **Integruje RAG pipeline s NewsAPI** – vyhledávání a analýza technologických novinek z českých a zahraničních zdrojů s využitím FAISS pro vektorové ukládání.
- **Nabízí interaktivní Streamlit dashboard** – se čtyřmi moduly (úvod, zpracování PDF, analytika, tech novinky) a pokročilými vizualizacemi.

---

## Hlavní komponenty aplikace

### 1. Streamlit Dashboard (app.py)

Aplikace obsahuje čtyři hlavní sekce:
- **Úvod** - představení projektu a dokumentace
- **Načtení a zpracování PDF** - upload a extrakce dat z faktur s následnou detekcí anomálií
- **Analytika** - předdefinované analytické pohledy s interaktivními grafy a interpretací výsledků v přirozeném jazyce pomocí LLM
- **Tech Novinky** - RAG pipeline pro analýzu technologických článků s využitím FAISS

### 2. Generování a zpracování dat (utils/ a data_processing/)

- **Generátor syntetických dat** (synthetic_data.ipynb) - vytváří simulovaná data faktur s 9% cíleně vloženými anomáliemi
- **PDF generátor** (synthetic_data.ipynb) - vytváří PDF faktury ze syntetických dat pomocí ReportLab
- **PDF text extractor** (pdf_text_extractor.py) - extrakce strukturovaných dat z textové vrstvy PDF dokumentů pomocí PyPDF
- **Entity extractor** (entity_extractor.py) - identifikace a kategorizace entit jako dodavatelé, odběratelé a částky

### Poznámka k struktuře PDF faktur

Vygenerované PDF faktury mají zjednodušenou textovou strukturu, která se liší od typického formátování reálných faktur. Použité faktury obsahují plnou textovou vrstvu (nejsou to naskenované obrázky), což umožňuje extrakci textu přímo pomocí knihovny PyPDF bez nutnosti skutečného OCR. Toto rozhodnutí bylo přijato z následujících důvodů:

- **Konzistentní extrakce dat** - standardizovaná struktura umožňuje spolehlivé získání všech klíčových informací jako jsou dodavatel, odběratel, částky a položky
- **Zaměření na workflow** - hlavním cílem projektu je demonstrovat celkový workflow zpracování dat, nikoli dokonalé OCR řešení pro komplexní formáty dokumentů
- **Omezení systému** - extraktor je optimalizován právě pro tuto strukturu, při použití externích faktur s odlišným formátováním by nemusel fungovat správně
- **Reprodukovatelnost výsledků** - díky konzistentní struktuře je zajištěna vysoká míra reprodukovatelnosti při testování celého zpracovatelského řetězce

Pro reálné nasazení by bylo nutné implementovat robustnější OCR řešení s pokročilou detekcí layoutu, které by zvládlo různé formáty faktur a rozmístění informací.

### 3. ML modely pro detekci anomálií (ml_models/)

- **XGBoost model** (model_prediction.ipynb) - trénování a evaluace modelu pro detekci anomálií
- **Feature engineering** - transformace kategorických proměnných, výpočet statistických metrik a normalizace
- **Batch predikce** (predict_pdf_batch.py) - dávkové zpracování faktur a identifikace anomálií

### 4. Analytické dotazování (llm_query/)

- **Query configuration** (query_config.py) - definice analytických dotazů a jejich interpretace
- **OpenAI integrace** - využití API pro přirozené dotazování a analýzu fakturačních dat
- **Vizualizace výsledků** - přehledné grafy a interpretace výsledků v přirozeném jazyce

### 5. RAG pipeline (rag/)

- **NewsAPI client** (newsapi_client.py) - získávání a zpracování technologických článků z českých a zahraničních zdrojů
- **FAISS vektorové úložiště** - ukládání a vyhledávání relevantních článků pro dotazy
- **Kontextově obohacené odpovědi** - generování odpovědí na základě nalezených relevantních článků

---

## Detekce anomálií

Projekt obsahuje detekci anomálií v platebních datech, která je realizována kombinací explicitního generování anomálií v syntetických datech a strojového učení (XGBoost):

- **Typy generovaných a detekovaných anomálií:**
    - **Vysoká částka + krátká splatnost** - extrémně vysoké částky s krátkou dobou splatnosti
    - **Nesoulad položek + datum** - nesoulad v položkách faktury a nestandardní datum splatnosti
    - **Neobvyklý počet položek** - faktury s abnormálním počtem položek (15-20 položek)
    - **Neobvyklá služba** - služby, které neodpovídají běžnému profilu dodavatele/odběratele

### Implementace detekce

- **Generování anomálií:**
  - Při vytváření syntetických dat je cca 9% faktur cíleně označeno jako anomálních a uloženo s příznakem `is_anomaly` a typem v `anomaly_type`
  - Anomálie jsou generovány s různými vahami (35% nesoulad položek, 35% vysoká částka, 20% neobvyklý počet položek, 10% neobvyklá služba)
  - Každý typ anomálie má specifické charakteristiky (např. částky nad 400 000 Kč s dobou splatnosti 1-3 dny)

- **Model pro detekci:**
  - Implementace využívá XGBoost model trénovaný na rozsáhlém datasetu (50 000 syntetických faktur)
  - Model využívá 16 různých features včetně `total_amount`, `items_count`, `is_month_end`, `days_to_due` a statistických metrik
  - Kategorické proměnné jsou zpracovány pomocí Label Encodingu a uloženy pro konzistentní použití v produkci
  - Všechny numerické features jsou normalizovány pomocí StandardScaler

- **Příprava dat a feature engineering:**
  - Pro každého dodavatele a odběratele jsou vypočítány statistické metriky (průměrná částka, směrodatná odchylka)
  - Kategorizace dodavatelů a odběratelů podle četnosti transakcí (`Top Supplier`, `Active Supplier`, `Special`)
  - Výpočet průměrné hodnoty položky jako dodatečný feature (`avg_item_value`)
  - Standardizace numerických proměnných pro zlepšení výkonu modelu

- **Výsledky a vizualizace:**
  - Model dosahuje na testovacích datech přesnosti >97% při identifikaci anomálií
  - Implementace dávkového zpracování faktur pomocí `predict_pdf_batch.py` pro praktické nasazení
  - Výsledkem je označení anomálních faktur ve vizualizaci Streamlit dashboardu s barevným rozlišením typů anomálií
  - Pro každou detekovanou anomálii je uvedena i míra jistoty predikce (`anomaly_confidence`)

### Limitace a reálné nasazení

- **Úspěšnost modelu je částečně dána charakterem syntetických dat** - anomálie jsou generovány podle předem definovaných vzorů, což modelu usnadňuje jejich detekci
- **V reálném prostředí lze očekávat nižší přesnost** - skutečné anomálie mohou být subtilnější nebo zcela jiného charakteru než ty v trénovacím datasetu
- **Pro produkční nasazení doporučuji:**
  - Pravidelné přetrénování modelu na nových datech s označenými anomáliemi
  - Implementaci zpětné vazby od finančních specialistů pro zlepšování modelu
  - Kombinaci ML přístupu s explicitními business pravidly pro specifické typy anomálií
  - Nastavení různých prahů detekce podle požadované citlivosti/specificity

---

## Integrace LLM a RAG pipeline

Projekt implementuje dva typy LLM integrace, které výrazně zlepšují uživatelskou zkušenost a přidávají analytickou hodnotu nad rámec běžných BI nástrojů:

### 1. Analytické dotazy s kontextovou interpretací (llm_query/)

- **Předefinované dotazy s pokročilými agregacemi** - systém nabízí 5 typů analytických pohledů:
  - Měsíční cashflow s analýzou trendů a bilancí
  - Top 5 odběratelů podle objemu příjmů s analýzou rizik
  - Rozložení výdajů podle kategorií s optimalizačními doporučeními
  - Distribuce splatností podle typu faktury s analýzou platební morálky
  - Analýza anomálií ve fakturách s identifikací rizikových oblastí

- **Promyšlená struktura dotazovacího engine:**
  - Oddělení datové agregace (`agg_func`), formátování (`format_func`) a generování promptů (`prompt_func`)
  - Dynamické formátování výstupů v českém formátu (např. "30 336 677,- Kč")
  - Adaptivní generování promptů na základě aktuálních dat pro maximální relevanci odpovědí

- **Kontextově obohacené LLM interpretace:**
  - GPT-3.5-turbo zpracovává numerická data s přesnými statistikami
  - Výstupy jsou strukturovány do konkrétních sekcí (shrnutí, doporučení, rizika)
  - Analýzy reflektují aktuální trendy v datech a upozorňují na anomálie

- **Interaktivní vizualizace s Plotly:**
  - Line charts pro časové řady (měsíční cashflow)
  - Bar charts s dynamickým formátováním pro top zákazníky
  - Treemap vizualizace pro hierarchické rozložení výdajů
  - Pie charts s barevným rozlišením pro distribuci splatností
  - Vizualizace s možností interaktivního zkoumání dat

### 2. RAG pipeline pro technologické novinky (rag/)

- **Multijazyčná NewsAPI integrace:**
  - Paralelní získávání článků z českých i zahraničních technologických zdrojů
  - Pokrytí širokého spektra médií (technet.idnes.cz, zive.cz, root.cz, techcrunch.com, theverge.com)
  - Flexibilní filtrování podle relevance a času publikace (30 dní)
  - Dynamické refreshování dat při každém dotazu pro zajištění aktuálnosti

- **Unifikované vektorové vyhledávání s FAISS:**
  - Společné ukládání českých i anglických článků v jednotném vektorovém úložišti
  - Efektivní in-memory implementace pro rychlé vyhledávání
  - Sémantické vyhledávání relevantních článků napříč jazyky
  - Efektivní práce s většími objemy vícejazyčných textových dat

- **Jazykově adaptivní odpovědi:**
  - Generování shrnujících odpovědí založených na kombinaci českých i mezinárodních zdrojů
  - Automatická detekce jazyka dotazu a přizpůsobení výstupu
  - Propojení informací z českého i mezinárodního kontextu
  - Využití modelu GPT-3.5-turbo pro generování odpovědí

- **Transparentní zobrazení multijazyčných zdrojů:**
  - Přehledné označení jazyka článku u každého zdroje
  - Možnost zobrazení původních článků a jejich metadat bez ohledu na jazyk
  - Kontrola relevance a věrohodnosti informačních zdrojů
  - Přímé odkazy na původní články pro hlubší studium

Obě integrace využívají OpenAI API, přičemž model GPT-3.5-turbo je používán jak pro analýzu strukturovaných dat, tak pro generování odpovědí na základě vektorově vyhledaných článků. Pro maximální flexibilitu systém umožňuje zadat API klíče přímo v aplikaci nebo je konfigurovat v souboru `config.py`.

---

## Jak spustit projekt

1. **Naklonujte repozitář a nainstalujte závislosti:**
    ```
    git clone <repo-url>
    cd <repo-directory>
    pip install -r requirements.txt
    ```

2. **Vytvořte konfigurační soubor s API klíči:**
```python
# config.py
OPENAI_API_KEY = "váš-openai-api-klíč"
NEWSAPI_KEY = "váš-newsapi-klíč"
```

3. **Vygeneruj syntetická data:**
    ```
    python -m utils.synthetic_data
    ```

4. **Spusť Streamlit aplikaci:**
    ```
    streamlit run app.py
    ```

---

## Poznámka k využití AI při vývoji

Při vývoji tohoto projektu jsem aktivně využíval generativní AI nástroje (GitHub Copilot, Perplexity a částečně ChatGPT) jako asistenty pro:
- Konzultaci a diskuzi architektonických rozhodnutí
- Návrhy implementace jednotlivých komponent
- Optimalizaci existujícího kódu
- Pomoc s debugováním a řešením problémů

Jsem přesvědčen, že v moderním vývojářském prostředí je efektivní využití AI nástrojů důležitou dovedností. Mým cílem nebylo pouze získat funkční kód, ale především:

1. **Porozumět každé části implementace** - Veškerý kód generovaný AI jsem analyzoval, upravoval a integroval se snahou plného pochopení jeho funkce
2. **Kriticky hodnotit navrhovaná řešení** - Ne všechny návrhy AI jsou optimální, proto jsem aktivně vyhodnocoval a často upravoval navržené postupy
3. **Překonávat omezení AI** - Řešil jsem problémy s kontextovým omezením, nekonzistencí a neúplnými implementacemi
4. **Učit se z interakce** - Využil jsem možnost diskutovat o různých implementačních možnostech a tím rozšířit své znalosti

Tento přístup mi umožnil vytvořit komplexnější projekt, než bych byl schopen realizovat samostatně, a zároveň prohloubit své porozumění všem implementovaným technologiím. Věřím, že schopnost efektivně spolupracovat s AI nástroji je v současnosti cennou dovedností, která doplňuje - nikoliv nahrazuje - základní technické znalosti.

---

## Vývojový proces a technická rozhodnutí

Během vývoje tohoto projektu jsem řešil několik zajímavých technických výzev:

- **Extrakce strukturovaných dat z PDF** – Pro spolehlivou extrakci jsem použil PyPDF s vlastní logikou pro identifikaci klíčových entit a kategorií, což umožňuje konzistentní zpracování faktur ve standardizovaném formátu.
- **Multijazyčná RAG pipeline** – Vyřešil jsem výzvu vyhledávání ve dvou jazycích pomocí jednotného vektorového úložiště v FAISS, což umožňuje propojit informace z českých i mezinárodních zdrojů.
- **Modulární architektura** – Jednotlivé komponenty jsou navrženy jako nezávislé moduly s jasně definovanými rozhraními, což umožnilo paralelní vývoj a testování jednotlivých částí aplikace.
- **Experimenty s hybridním RAG + SQL přístupem** – Experimentoval jsem s generováním SQL dotazů z přirozeného jazyka a následnou interpretací výsledků pomocí LLM. Přestože tento přístup fungoval dobře pro základní analytické dotazy (celkové náklady, porovnání příjmů a výdajů, atd), u složitějších dotazů jsem narážel na potřebu rozsáhlého ošetření edge cases a synonymických výrazů. Na základě těchto experimentů jsem se rozhodl pro pragmatičtější řešení s předdefinovanými analytickými pohledy v `query_config.py`, které nabízí optimální rovnováhu mezi flexibilitou a spolehlivostí.

---

## Doporučení a poznámky

- **Vždy spouštěj skripty z kořenového adresáře projektu** kvůli správnému fungování importů.
- **Pro pokročilou analytiku lze v budoucnu přidat sekci s detailními metrikami a vizualizacemi.**
- **Pro detekci anomálií doporučuji vylepšit kombinaci ML a business pravidel, případně využít zpětnou vazbu uživatelů pro zpřesnění modelu v reálném prostředí.**
- **Výměna ChromaDB za FAISS** - Odstranění závislosti na SQLite, která způsobovala problémy na Streamlit Cloud a použití in-memory vektorového úložiště FAISS pro efektivnější práci s embeddingy.


