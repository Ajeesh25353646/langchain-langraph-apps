import pandas as pd
import json
import os

# Map Bitext intents to your main.py categories:
# refund, technical_issue, inquiry
INTENT_MAPPING = {
    # REFUND
    "cancel_order": "refund",
    "get_refund": "refund",
    "track_refund": "refund",
    "check_refund_policy": "refund",
    "payment_issue": "refund",
    "check_cancellation_fee": "refund",
    "refund_status": "refund",
    
    # TECHNICAL_ISSUE
    "registration_problems": "technical_issue",
    "recover_password": "technical_issue",
    "edit_account": "technical_issue",
    "bug_report": "technical_issue",
    "troubleshoot_device": "technical_issue",
    "setup_help": "technical_issue",
    "error_message": "technical_issue",
    
    # INQUIRY (Default)
    "delivery_options": "inquiry",
    "delivery_period": "inquiry",
    "track_order": "inquiry",
    "get_invoice": "inquiry",
    "check_invoice": "inquiry",
    "product_info": "inquiry",
    "contact_customer_service": "inquiry",
    "contact_human_agent": "inquiry",
    "change_shipping": "inquiry",
    "change_shipping_address": "inquiry",
    "set_up_shipping_address": "inquiry",
    "newsletter_subscription": "inquiry"
}

def preprocess_bitext_pandas(output_file: str, limit: int = 500):
    """
    Uses pandas to load Bitext directly from HF and processes it for LangSmith.
    """
    print("Loading Bitext dataset via pandas...")
    url = "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading from HF: {e}")
        return

    # Sample for performance
    df_sample = df.head(limit)
    
    processed_examples = []

    for _, row in df_sample.iterrows():
        raw_intent = str(row['intent'])
        mapped_intent = INTENT_MAPPING.get(raw_intent, "inquiry")
        
        # Extract flags/tags for metadata (e.g., 'W' for offensive, 'Z' for typos)
        tags = str(row.get('flags', ''))
        
        example = {
            "inputs": {
                "messages": [{"role": "user", "content": row['instruction']}]
            },
            "outputs": {
                "expected_intent": mapped_intent,
                "expected_response": row['response'],
                "is_offensive": 'W' in tags,
                "has_typos": 'Z' in tags
            },
            "metadata": {
                "raw_intent": raw_intent,
                "category": row['category'],
                "tags": tags
            }
        }
        processed_examples.append(example)

    with open(output_file, 'w') as f:
        json.dump(processed_examples, f, indent=2)
    
    print(f"✅ Successfully preprocessed {len(processed_examples)} records using pandas.")

if __name__ == "__main__":
    preprocess_bitext_pandas("data/bitext_processed.json")

