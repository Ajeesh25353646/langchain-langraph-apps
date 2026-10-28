import json
import os

# Mapping granular banking intents to your support categories:
# refund, technical_issue, inquiry
BANKING_MAPPING = {
    # REFUND / BILLING
    "refund_not_showing_up": "refund",
    "declined_transfer": "refund",
    "pending_transfer": "refund",
    "wrong_amount_of_cash_received": "refund",
    "cancel_transfer": "refund",
    "top_up_failed": "refund",
    
    # TECHNICAL_ISSUE
    "card_not_working": "technical_issue",
    "card_acceptance": "technical_issue",
    "app_reinstall": "technical_issue",
    "lost_or_stolen_card": "technical_issue",
    "pin_blocked": "technical_issue",
    "verify_identity": "technical_issue",
    
    # INQUIRY
    "exchange_rate": "inquiry",
    "card_delivery_estimate": "inquiry",
    "get_physical_card": "inquiry",
    "transfer_fee_charged": "inquiry",
    "balance_not_updated_after_cheque_deposit": "inquiry",
    "atm_support": "inquiry"
}

def preprocess_banking(input_file: str, output_file: str):
    """
    Preprocessing for Banking77 style datasets.
    """
    if not os.path.exists(input_file):
        print(f"Note: {input_file} not found yet. Script is ready for when you add it.")
        return

    processed_data = []
    
    # Assuming standard JSON format from HuggingFace/datasets
    with open(input_file, 'r') as f:
        data = json.load(f)
        for item in data:
            query = item.get("text", "") or item.get("query", "")
            label = item.get("label", "")
            
            mapped_intent = BANKING_MAPPING.get(label, "inquiry")
            
            processed_data.append({
                "inputs": {
                    "messages": [{"role": "user", "content": query}]
                },
                "outputs": {
                    "expected_intent": mapped_intent,
                    "raw_label": label
                }
            })

    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"✅ Processed {len(processed_data)} banking records.")

if __name__ == "__main__":
    preprocess_banking("data/banking_raw.json", "data/banking_processed.json")
