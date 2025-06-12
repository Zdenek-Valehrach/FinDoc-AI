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
    # Explicitní konverze číselných sloupců
    numeric_cols = ["total_amount", "items_count"]
    for col in numeric_cols:
        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
    
    # Odstranění neplatných záznamů
    df = df.dropna(subset=numeric_cols)
    df["avg_item_value"] = (df["total_amount"] / df["items_count"].astype(float)).round(2)
    
    # 3. Kategorizace dodavatelů a odběratelů
    supplier_frequency = df["supplier_name"].value_counts()
    customer_frequency = df["customer_name"].value_counts()

    supplier_treshold = 3000
    customer_treshold = 4500

    def categorize_supplier(supplier_name):
        if supplier_name == "FinDoc AI":
            return "Special"
        elif supplier_frequency[supplier_name] > supplier_treshold:
            return "Top Supplier"
        else:
            return "Active Supplier"
        
    def categorize_customer(customer_name):
        if customer_name == "FinDoc AI":
            return "Special"
        elif customer_frequency[customer_name] > customer_treshold:
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
        df[col + "_encoded"] = le.transform(df[col])

    
    # 7. Finální výběr příznaků
    features = ['total_amount', 'is_month_end', 'items_count', 'avg_item_value',
       'days_to_due', 'customer_mean', 'customer_std', 'supplier_mean',
       'supplier_std', 'supplier_name_encoded', 'customer_name_encoded',
       'category_encoded', 'transaction_type_encoded', 'note_encoded',
       'supplier_category_encoded', 'customer_category_encoded']
    
    # 8. Scaling 
    SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(df[features])
    
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
    # 1. Načtení dat
    df = pd.read_csv(input_csv, parse_dates=["invoice_date", "due_date"])
    
    # 2. Preprocessing
    X = preprocess_data(df)
    
    # 3. Načtení modelu
    model = load_model(model_path)
    
    # 4. Predikce
    y_pred, y_proba = predict_anomalies(model, X)
    
    # 5. Uložení výsledků
    df['anomaly_type_pred'] = y_pred
    df['anomaly_confidence'] = y_proba
    df.to_csv(output_csv, index=False)
    print(f"Výsledky uloženy do {output_csv}")

# Příklad volání:
# if __name__ == "__main__":
#     batch_predict(
#         input_csv="vysledky_faktur.csv",
#         model_path="xgb_model.pkl",
#         output_csv="predikce_pdf.csv"
#     )
