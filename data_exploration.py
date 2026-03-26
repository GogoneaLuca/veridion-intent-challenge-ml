import pandas as pd
import numpy as np
import ast
from collections import Counter

# ==========================================
# 1. FUNCȚII DE CURĂȚARE ȘI PARSARE
# ==========================================

def clean_dict_string(val):
    """Transformă string-urile cu aspect de dicționar în dicționare reale."""
    if isinstance(val, dict): return val
    if pd.isna(val): return None
    if isinstance(val, str):
        try: return ast.literal_eval(val)
        except: return None
    return None

def clean_list(val):
    """Extrage listele în siguranță, ignorând NaN sau formate greșite."""
    if isinstance(val, list): return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else []
        except: return []
    return []

def extract_tld(website):
    """Extrage extensia domeniului (ex: ro, com, de) din website."""
    if pd.isna(website) or not isinstance(website, str): return None
    try:
        # Ștergem eventualele 'http://', 'www.' și luăm ce e după ultimul punct
        domain = website.split('//')[-1].split('/')[0].split('www.')[-1]
        return domain.split('.')[-1].lower()
    except:
        return None

# ==========================================
# 2. FUNCȚIA PRINCIPALĂ DE ANALIZĂ
# ==========================================

def run_full_eda():
    print("="*50)
    print("ÎNCEPERE ANALIZĂ COMPLETĂ SET DE DATE COMPANII")
    print("="*50)

    try:
        # ATENȚIE: Aici am modificat calea ca să citească din folderul curent!
        df = pd.read_json("companies.jsonl", lines=True)
    except FileNotFoundError:
        print(" Eroare: Nu s-a găsit fișierul 'companies.jsonl' în acest folder.")
        return

    total_companies = len(df)
    print(f"Total companii procesate: {total_companies}\n")

    # --- A. ANALIZA VALORILOR LIPSĂ (MISSING DATA) ---
    print("--- A. RATA DE COMPLETARE A DATELOR ---")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / total_companies) * 100
    missing_df = pd.DataFrame({'Lipsă (Count)': missing_data, 'Lipsă (%)': missing_percent})
    missing_df = missing_df.sort_values(by='Lipsă (%)', ascending=False)
    print(missing_df.to_string(formatters={'Lipsă (%)': '{:.2f}%'.format}))
    print("\n")

    # --- B. DATE NUMERICE / FINANCIARE ---
    print("--- B. DATE NUMERICE (Dimensiune & Vechime) ---")

    # Curățăm anul
    df['year_founded_clean'] = pd.to_numeric(df['year_founded'], errors='coerce')
    print(f"Anul înființării:")
    print(f"  - Cel mai vechi: {df['year_founded_clean'].min()}")
    print(f"  - Cel mai nou: {df['year_founded_clean'].max()}")
    print(f"  - Fondate după 2018: {len(df[df['year_founded_clean'] > 2018])} companii\n")

    # Angajați și Venituri
    print("Statistici Angajați (employee_count):")
    print(df['employee_count'].describe().to_string(float_format="{:.2f}".format))
    print("\nStatistici Venituri (revenue în USD):")
    rev_stats = df['revenue'].describe()
    print(rev_stats.apply(lambda x: f"${x:,.2f}"))
    print("\n")

    # --- C. GEOGRAFIE, INDUSTRIE ȘI IDENTIFICATORI ---
    print("--- C. GEOGRAFIE & INDUSTRIE ---")

    # Țara (din Address)
    df['address_dict'] = df['address'].apply(clean_dict_string)
    df['country'] = df['address_dict'].apply(lambda x: x.get('country_code') if isinstance(x, dict) else None)
    print(f"Top 5 Țări (din Address):")
    print(df['country'].value_counts().head(5).to_string())

    # Extensii Website (alternativă la locație)
    df['tld'] = df['website'].apply(extract_tld)
    print(f"\nTop 5 Extensii Domeniu (Website):")
    print(df['tld'].value_counts().head(5).to_string())

    # Industrie (NAICS)
    df['naics_dict'] = df['primary_naics'].apply(clean_dict_string)
    df['industry'] = df['naics_dict'].apply(lambda x: x.get('label') if isinstance(x, dict) else None)
    print(f"\nAvem {df['industry'].nunique()} industrii unice. Top 3:")
    print(df['industry'].value_counts().head(3).to_string())

    # Companii publice
    print(f"\nStatut 'is_public':")
    print(df['is_public'].value_counts(dropna=False).to_string())
    print("\n")

    # --- D. CATEGORII LISTATE ---
    print("--- D. ANALIZA LISTELOR (Semantica afacerii) ---")
    list_cols = ['business_model', 'target_markets', 'core_offerings']
    for col in list_cols:
        df[col] = df[col].apply(clean_list)
        all_items = [item for sublist in df[col] for item in sublist if isinstance(item, str)]
        unique_count = len(set(all_items))

        print(f"-> {col.upper()}:")
        print(f"   Valori unice totale: {unique_count}")
        if unique_count > 0:
            top_3 = Counter(all_items).most_common(3)
            print(f"   Top 3 cele mai frecvente: {top_3}")
    print("\n")

    # --- E. TEXT LIBER (DESCRIPTION) ---
    print("--- E. ANALIZA TEXTULUI (Description) ---")
    df['desc_words'] = df['description'].fillna("").apply(lambda x: len(str(x).split()))
    desc_stats = df[df['desc_words'] > 0]['desc_words'].describe()
    print("Statistici lungime 'description' (în cuvinte):")
    print(f"  - Min: {desc_stats['min']:.0f} cuvinte")
    print(f"  - Max: {desc_stats['max']:.0f} cuvinte")
    print(f"  - Medie: {desc_stats['mean']:.1f} cuvinte")
    print("="*50)

if __name__ == "__main__":
    run_full_eda()