import pandas as pd
import joblib
import os


def load_model(model_path):
    """Načte uložený model."""
    return joblib.load(model_path)

def preprocess_data(df):
    """Kompletní preprocessing dat a příprava pro model."""
    # Kopie DataFrame pro bezpečnou manipulaci
    df = df.copy()
    
    # 1. Datové typy
    categorical_columns = ["supplier_name", "customer_name", "category", "note", "transaction_type"]
    df[categorical_columns] = df[categorical_columns].astype("category")
        
    # 2. Průměrná hodnota položky
    # Explicitní konverze všech číselných sloupců
    numeric_cols = ["total_amount", "items_count", "is_month_end"]
    for col in numeric_cols:
        if col in df.columns:  # Kontrola, zda sloupec existuje
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Odstranění neplatných záznamů
    df = df.dropna(subset=["total_amount", "items_count"])
    
    # Výpočet průměrné hodnoty položky
    df["avg_item_value"] = (df["total_amount"] / df["items_count"].astype(float)).round(2)
    
    # 3. Kategorizace dodavatelů a odběratelů (zůstává stejné)
    supplier_frequency = df["supplier_name"].value_counts()
    customer_frequency = df["customer_name"].value_counts()

    supplier_treshold = 3000
    customer_treshold = 4500

    def categorize_supplier(supplier_name):
        if supplier_name == "FinDoc AI":
            return "Special"
        elif supplier_name in supplier_frequency and supplier_frequency[supplier_name] > supplier_treshold:
            return "Top Supplier"
        else:
            return "Active Supplier"
        
    def categorize_customer(customer_name):
        if customer_name == "FinDoc AI":
            return "Special"
        elif customer_name in customer_frequency and customer_frequency[customer_name] > customer_treshold:
            return "Top Customer"
        else:
            return "Active Customer"
        
    df["supplier_category"] = df["supplier_name"].apply(categorize_supplier).astype("category")
    df["customer_category"] = df["customer_name"].apply(categorize_customer).astype("category")
    
    # 4. Časové charakteristiky
    df["days_to_due"] = (df["due_date"] - df["invoice_date"]).dt.days
    
    # 5. Statistické charakteristiky 
    customer_avg = df.groupby("customer_name", observed=True)["total_amount"].mean()
    customer_std = df.groupby("customer_name", observed=True)["total_amount"].std()

    supplier_avg = df.groupby("supplier_name", observed=True)["total_amount"].mean()
    supplier_std = df.groupby("supplier_name", observed=True)["total_amount"].std()

    df["customer_mean"] = df["customer_name"].map(customer_avg).astype(float).round(2)
    df["customer_std"] = df["customer_name"].map(customer_std).astype(float).round(2)
    df["supplier_mean"] = df["supplier_name"].map(supplier_avg).astype(float).round(2)
    df["supplier_std"] = df["supplier_name"].map(supplier_std).astype(float).round(2)
    
    # 6. Label Encoding
    ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "label_encoders.pkl")
    label_encoders = joblib.load(ENCODERS_PATH)
    
    for col, le in label_encoders.items():
        if col in df.columns:  # Ověřit, že sloupec existuje
            # Pokud je některá hodnota neznámá pro encoder, nahraď ji nejčastější hodnotou
            unique_values = df[col].unique()
            encoder_classes = le.classes_
            
            for value in unique_values:
                if value not in encoder_classes:
                    # Nahraď nejčastější hodnotou ve sloupci
                    most_common = df[col].value_counts().index[0]
                    df.loc[df[col] == value, col] = most_common
            
            df[col + "_encoded"] = le.transform(df[col])

    # 7. Finální výběr příznaků
    features = ['total_amount', 'is_month_end', 'items_count', 'avg_item_value',
       'days_to_due', 'customer_mean', 'customer_std', 'supplier_mean',
       'supplier_std', 'supplier_name_encoded', 'customer_name_encoded',
       'category_encoded', 'transaction_type_encoded', 'note_encoded',
       'supplier_category_encoded', 'customer_category_encoded']
    
    # Kontrola a oprava chybějících hodnot
    for feature in features:
        if feature in df.columns:
            # Nahradit NaN hodnoty nulami
            df[feature] = df[feature].fillna(0)
            # Zajistit číselný typ
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
        else:
            print(f"VAROVÁNÍ: Sloupec {feature} chybí v datasetu!")
            # Vytvořit chybějící sloupec s nulami
            df[feature] = 0
    
    # 8. Scaling 
    SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    scaler = joblib.load(SCALER_PATH)
    
    # Kontrola formátu dat před transformací
    print(f"Kontrola typů dat před transformací: {df[features].dtypes}")
    
    # Převést všechny hodnoty na float64 pro jistotu
    df_features = df[features].astype(float)
    
    # Nyní transformovat
    X_scaled = scaler.transform(df_features)
    
    return pd.DataFrame(X_scaled, columns=features)

def predict_anomalies(model, X):
    """Vrátí predikce a pravděpodobnosti."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X).max(axis=1)
    return y_pred, y_proba

def batch_predict(input_csv, model_path, output_csv):
    """
    Načte data z CSV, provede preprocessing, načte model, provede predikci a uloží výsledek.
    """
    try:
        # 1. Načtení dat
        df = pd.read_csv(input_csv, parse_dates=["invoice_date", "due_date"])
        print(f"Data načtena, tvar: {df.shape}")
        
        # 2. Preprocessing
        try:
            X = preprocess_data(df)
            print(f"Preprocessing dokončen, tvar: {X.shape}")
            # Kontrola datových typů po preprocessingu
            print("Datové typy po preprocessingu:")
            print(X.dtypes)
        except Exception as e:
            print(f"Chyba při preprocessingu: {e}")
            raise
        
        # 3. Načtení modelu
        model = load_model(model_path)
        
        # 4. Predikce
        y_pred, y_proba = predict_anomalies(model, X)
        
        # 5. Uložení výsledků
        df['anomaly_type_pred'] = y_pred
        df['anomaly_confidence'] = y_proba
        df.to_csv(output_csv, index=False)
        print(f"Výsledky uloženy do {output_csv}")
    
    except Exception as e:
        import traceback
        print(f"Chyba při zpracování: {e}")
        print(traceback.format_exc())
        raise

# Příklad volání:
# if __name__ == "__main__":
#     batch_predict(
#         input_csv="vysledky_faktur.csv",
#         model_path="xgb_model.pkl",
#         output_csv="predikce_pdf.csv"
#     )
