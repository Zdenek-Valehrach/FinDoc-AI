import os
import argparse
from .pdf_text_extractor import extract_invoice_data
from .entity_extractor import create_invoice_dataframe

def get_pdf_files(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

# Zpracování faktur
if __name__ == "__main__":
    # Argumenty příkazové řádky
    parser = argparse.ArgumentParser(description='Dávkové zpracování PDF faktur')
    parser.add_argument('--pdf_dir', type=str, required=True, help='Cesta k adresáři s PDF fakturami')
    parser.add_argument('--output', type=str, default="vysledky_faktur.csv", 
                       help='Název výstupního CSV souboru (výchozí: vysledky_faktur.csv)')
    args = parser.parse_args()
    
    pdf_files = get_pdf_files(args.pdf_dir)
    print(f"Nalezeno {len(pdf_files)} PDF souborů.")

    # Extrakce dat z každého PDF
    pdf_results = []
    for path in pdf_files:
        try:
            data = extract_invoice_data(path)
            pdf_results.append(data)
            print(f"Zpracováno: {os.path.basename(path)}")
        except Exception as e:
            print(f"Chyba při zpracování {path}: {str(e)}")

    # Vytvoření DataFrame
    df = create_invoice_dataframe(pdf_results)

    # Uložení výsledků
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Hotovo! Výsledky jsou uloženy v souboru {args.output}")