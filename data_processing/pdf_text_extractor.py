import re
from pypdf import PdfReader
from collections import defaultdict

def parse_invoice_text(text):
    result = defaultdict(list)
    
    # Zpracování základních informací
    result['invoice_id'] = re.findall(r'Číslo faktury:\s*(\d+)', text)
    result['variable_symbol'] = re.findall(r'Variabilní symbol:\s*(\d+)', text)
    result['invoice_date'] = re.findall(r'Datum vystavení:\s*(\d{1,2}\.\d{1,2}\.\d{4})', text)
    result['due_date'] = re.findall(r'Datum splatnosti:\s*(\d{1,2}\.\d{1,2}\.\d{4})', text)
    
    # Zpracování poznámky (kategorie)
    note_match = re.search(r'Faktura za:\s*(.+?)(?=\n|$)', text)
    if note_match:
        result['note'] = [note_match.group(1).strip()]
    
    # Zpracování položek
    items_section = re.search(r'Faktura za:.+?\n(.+?)Celkem:', text, re.DOTALL)
    if items_section:
        items = []
        lines = items_section.group(1).split('\n')
        for line in lines:
            line = line.strip()
            if line and 'CZK' in line:
                # Rozdělení na popis a částku
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 2:
                    amount = parts[-1].replace(' ', '')
                    if not amount.endswith('CZK'):
                        amount += 'CZK'
                    items.append({
                        'description': ' '.join(parts[:-1]).strip(),
                        'amount': amount
                    })
        result['items'] = items
    
    # Zpracování dodavatele
    supplier_block = re.search(r'Dodavatel:(.+?)(?=Odběratel:|\n\n)', text, re.DOTALL)
    if supplier_block:
        supplier_text = supplier_block.group(1)
        result['supplier_name'] = re.findall(r'^\s*(.+?)\n', supplier_text)
        result['supplier_ico'] = re.findall(r'IČO:\s*(\d+)', supplier_text)
        result['supplier_dic'] = re.findall(r'DIČ:\s*(CZ\d+)', supplier_text)
        result['supplier_account'] = re.findall(r'Č. účtu:\s*([A-Z0-9/]+)', supplier_text)
    
    # Zpracování odběratele
    customer_block = re.search(r'Odběratel:(.+?)(?=Variabilní symbol:|\n\n)', text, re.DOTALL)
    if customer_block:
        customer_text = customer_block.group(1)
        result['customer_name'] = re.findall(r'^\s*(.+?)\n', customer_text)
        result['customer_ico'] = re.findall(r'IČO:\s*(\d+)', customer_text)
        result['customer_dic'] = re.findall(r'DIČ:\s*(CZ\d+)', customer_text)
    
    # Celková částka
    total_match = re.search(r'Celkem:\s*([\d\s.,]+)\s*CZK', text)
    if total_match:
        result['total_amount'] = [total_match.group(1).replace(' ', '')]
    
    return dict(result)

def extract_invoice_data(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text(
            extraction_mode="layout",
            layout_mode_scale_weight=2.0,
            layout_mode_strip_rotated=True,
            layout_mode_space_vertically=False
        ) + "\n\n"
    
    return parse_invoice_text(text)

# # Příklad použití
# if __name__ == "__main__":
#     data = extract_invoice_data("cesta_k_souboru.pdf")
#     print(data)


