import pandas as pd
from datetime import datetime
from .pdf_text_extractor import extract_invoice_data  
from pandas.tseries.offsets import MonthEnd

def is_month_end_or_two_days_before(date):
    """Vrátí True pokud datum je poslední den měsíce nebo 2 dny před koncem měsíce"""
    if pd.isna(date):
        return False
    date = pd.to_datetime(date).normalize()
    last_day = (date + MonthEnd(0)).normalize()
    return last_day - pd.Timedelta(days=2) <= date <= last_day

def process_extracted_data(extracted_data):
    """Zpracuje výstup z pdf_text_extractor.py do DataFrame"""
    
    # Základní struktura výsledného záznamu
    record = {
        "invoice_id": extracted_data.get("invoice_id", [""])[0],
        "supplier_name": extracted_data.get("supplier_name", [""])[0],
        "supplier_ico": extracted_data.get("supplier_ico", [""])[0],
        "supplier_dic": extracted_data.get("supplier_dic", [""])[0],
        "supplier_account": extracted_data.get("supplier_account", [""])[0],
        "customer_name": extracted_data.get("customer_name", [""])[0],
        "customer_ico": extracted_data.get("customer_ico", [""])[0],
        "customer_dic": extracted_data.get("customer_dic", [""])[0],
        "invoice_date": parse_date(extracted_data.get("invoice_date", [""])[0]),
        "due_date": parse_date(extracted_data.get("due_date", [""])[0]),
        "variable_symbol": extracted_data.get("variable_symbol", [""])[0],
        "items_count": process_items(extracted_data.get("items", [])),
        "category": extracted_data.get("note", [""])[0].strip(),
        "transaction_type": "Příjmy" if "FinDoc AI" in extracted_data.get("supplier_name", [""]) else "Výdaje",
        "note": extracted_data.get("note", [""])[0],
        "total_amount": extracted_data.get("total_amount", [""])[0].replace(" ", "").replace(",", ".") if extracted_data.get("total_amount") else "",
        "is_month_end": is_month_end_or_two_days_before(
            pd.to_datetime(extracted_data.get("invoice_date", [""])[0], format="%d.%m.%Y", errors='coerce'))
    }
    
    return pd.DataFrame([record])

def parse_date(date_str):
    """Převádí datum z řetězce na datetime objekt"""
    try:
        return datetime.strptime(date_str, "%d.%m.%Y") if date_str else None
    except:
        return None

def extract_category(note):
    """Vrátí celou hodnotu poznámky jako kategorii"""
    return note.strip() if note else ""

def process_items(items):
    """Zpracuje položky faktury"""
    if isinstance(items, list):
        return len(items)
    if isinstance(items, str):
        return len(items.split("; "))
    return 0

def create_invoice_dataframe(extracted_data_list):
    """Vytvoří DataFrame z listu extrahovaných textových dat"""
    return pd.concat([process_extracted_data(data) for data in extracted_data_list], ignore_index=True)

# Příklad použití
# if __name__ == "__main__":
#     # Předpokládáme, že máme list výstupů z pdf_text_extractor.py
#     extracted_results = [extract_invoice_text_data("Cesta k souboru.pdf")]
#     
#     # Vytvoření finálního DataFrame
#     df = create_invoice_dataframe(extracted_results)
#     print(df)

