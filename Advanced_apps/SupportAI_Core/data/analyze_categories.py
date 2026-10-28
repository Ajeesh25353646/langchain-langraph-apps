import pandas as pd
import os

def analyze_dataset(file_path: str):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    print("--- Dataset Column Overview ---")
    print(df.columns.tolist())
    
    print("\n--- Unique Query Categories ---")
    print(df['Query_Category'].unique())
    
    print("\n--- Unique Query Subcategories ---")
    print(df['Query_Subcategory'].unique())
    
    print("\n--- Unique Detected Intents ---")
    print(df['Intent_Detected'].unique())
    
    print("\n--- Value Counts for Categories ---")
    print(df['Query_Category'].value_counts())

if __name__ == "__main__":
    analyze_dataset("data/Agentic_AI_Customer_Support_Dataset_v1.csv")
