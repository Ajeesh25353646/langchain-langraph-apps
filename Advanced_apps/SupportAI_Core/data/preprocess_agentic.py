import pandas as pd
import json
import os

def preprocess_agentic_csv(input_file: str, output_file: str):
    """
    Processes the Agentic AI Customer Support Dataset CSV.
    Maps categories to: refund, technical_issue, inquiry
    """
    if not os.path.exists(input_file):
        print(f"File {input_file} not found. Please place it in the data/ directory.")
        return

    df = pd.read_csv(input_file)
    
    processed_examples = []

    for _, row in df.iterrows():
        # Mapping Logic
        subcat = str(row['Query_Subcategory'])
        cat = str(row['Query_Category'])
        
        if "Refund" in subcat:
            mapped_intent = "refund"
        elif any(x in subcat for x in ["Payment Failure", "Login Issue", "Bug Report"]) or cat == "Technical":
            mapped_intent = "technical_issue"
        else:
            mapped_intent = "inquiry"

        # Map Priority
        priority = str(row['Priority_Level']).lower()
        if priority not in ["low", "medium", "high", "urgent"]:
            priority = "medium"

        # Format for LangSmith
        example = {
            "inputs": {
                "messages": [{"role": "user", "content": row['Customer_Query_Text']}],
                "customer_id": row['Customer_ID'],
                "ticket_id": row['Ticket_ID']
            },
            "outputs": {
                "expected_intent": mapped_intent,
                "expected_priority": priority,
                "expected_resolution": row['AI_Response_Text'],
                "should_escalate": bool(row['Human_Escalation_Flag'])
            }
        }
        processed_examples.append(example)

    with open(output_file, 'w') as f:
        json.dump(processed_examples, f, indent=2)
    
    print(f"✅ Successfully processed {len(processed_examples)} records from Agentic CSV.")

if __name__ == "__main__":
    preprocess_agentic_csv(
        "data/Agentic_AI_Customer_Support_Dataset_v1.csv", 
        "data/agentic_processed.json"
    )
