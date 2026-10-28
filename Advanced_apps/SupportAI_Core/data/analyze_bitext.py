import pandas as pd
import os

def analyze_bitext():
    print("Loading Bitext dataset for deep analysis...")
    url = "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\n--- 1. Column Overview ---")
    print(df.columns.tolist())

    print("\n--- 2. High-Level Categories (Top 10) ---")
    print(df['category'].value_counts())

    print("\n--- 3. Top 10 Intents ---")
    print(df['intent'].value_counts().head(10))

    print("\n--- 4. Linguistic Flags (The 'Brain' of Bitext) ---")
    # Tags: P (Polite), Q (Colloquial), W (Offensive), Z (Typos), etc.
    all_flags = "".join(df['flags'].fillna("").tolist())
    flag_counts = {
        "P (Polite)": all_flags.count('P'),
        "Q (Colloquial)": all_flags.count('Q'),
        "W (Offensive)": all_flags.count('W'),
        "Z (Typos/Errors)": all_flags.count('Z'),
        "K (Keyword mode)": all_flags.count('K'),
        "B (Basic structure)": all_flags.count('B'),
        "I (Interrogative)": all_flags.count('I'),
        "N (Negation)": all_flags.count('N')
    }
    
    for flag, count in flag_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{flag}: {count} records ({percentage:.1f}%)")

    print("\n--- 5. Complexity Stats ---")
    print(f"Average Instruction Length: {df['instruction'].str.len().mean():.1f} chars")
    print(f"Average Response Length: {df['response'].str.len().mean():.1f} chars")

    print("\n--- 6. Sample of Offensive Language (W) ---")
    offensive_sample = df[df['flags'].str.contains('W', na=False)]['instruction'].head(3).tolist()
    for i, s in enumerate(offensive_sample):
        print(f"{i+1}. {s}")

if __name__ == "__main__":
    analyze_bitext()
