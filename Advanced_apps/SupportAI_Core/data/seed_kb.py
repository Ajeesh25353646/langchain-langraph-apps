import pandas as pd
import json
import os

# The 14 most relevant E-commerce intents for a comprehensive support agent
TARGET_INTENTS = [
    "recover_password", "registration_problems", "edit_account",
    "change_shipping_address", "delivery_period", "delivery_options",
    "check_refund_policy", "get_refund", "track_refund",
    "track_order", "cancel_order", "get_invoice",
    "payment_issue", "check_cancellation_fee"
]

def seed_kb(limit_per_intent=10):
    print("🚀 Starting Knowledge Base Seeding from Bitext...")
    url = "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 1. Basic Cleaning & Quality Filtering
    # Exclude Offensive (W) and Typos (Z)
    df = df[~df['flags'].str.contains('W|Z', na=False)]
    
    # 2. Filter for Target Intents
    df = df[df['intent'].isin(TARGET_INTENTS)]
    
    kb_entries = []

    for intent in TARGET_INTENTS:
        intent_df = df[df['intent'] == intent]
        
        if intent_df.empty:
            continue
            
        # 3. Ensure Diversity
        # Bitext often uses the same response for many instructions. 
        # We group by unique 'response' first to ensure we aren't repetitive.
        unique_responses = intent_df.groupby('response').first().reset_index()
        
        # 4. Sort by complexity (Response Length) to get the most informative answers
        unique_responses['res_len'] = unique_responses['response'].str.len()
        unique_responses = unique_responses.sort_values('res_len', ascending=False)
        
        # 5. Take the top entries
        selection = unique_responses.head(limit_per_intent)
        
        for _, row in selection.iterrows():
            kb_entries.append({
                "question": row['instruction'],
                "answer": row['response'],
                "metadata": {
                    "intent": intent,
                    "category": row['category']
                }
            })
            
        print(f"✅ Extracted {len(selection)} unique pairs for intent: {intent}")

    # 6. Save to kb.json
    output_path = "kb.json"
    with open(output_path, 'w') as f:
        json.dump(kb_entries, f, indent=2)
    
    print(f"\n✨ Success! New Knowledge Base created with {len(kb_entries)} entries.")
    print(f"📁 Saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    seed_kb(limit_per_intent=10)
