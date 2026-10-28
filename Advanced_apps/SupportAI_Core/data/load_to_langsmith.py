import os
import pandas as pd
from langsmith import Client
from dotenv import load_dotenv

# Load environment variables for API Keys
load_dotenv()

def create_ecommerce_eval_set(dataset_name="Customer Support Evaluation Dataset"):
    client = Client()
    
    # 1. Check if dataset already exists
    if client.has_dataset(dataset_name=dataset_name):
        print(f"Dataset '{dataset_name}' already exists. Skipping creation.")
        return

    print(f"📥 Loading Bitext data for {dataset_name}...")
    url = "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    df = pd.read_csv(url)

    # 2. Advanced Data Separation
    test_df = df[df['flags'].str.contains('Z|Q|W', na=False)]
    
    mapping = {
        "get_refund": "refund",
        "track_refund": "refund",
        "recover_password": "technical_issue",
        "registration_problems": "technical_issue",
        "payment_issue": "technical_issue",
        "track_order": "inquiry",
        "delivery_period": "inquiry",
        "get_invoice": "inquiry",
        "contact_customer_service": "inquiry",
        "order_status": "inquiry"
    }
    
    test_df = test_df[test_df['intent'].isin(mapping.keys())]
    
    # 3. Create Small Sample (3 entries)
    sample_df = test_df.sample(n=50, random_state=42)

    # 4. Create the Dataset in LangSmith
    dataset = client.create_dataset(
        dataset_name=dataset_name, 
        description="Extensive testing to verify evaluation schema and tracing."
    )

    # 5. Prepare Examples
    for _, row in sample_df.iterrows():
        is_angry = 'W' in str(row['flags'])

        # Map sentiment to match main.py schema (4 categories)
        # 'W' flag = angry, otherwise = frustrated (annoyed but not enraged)
        expected_sentiment = "angry" if is_angry else "frustrated"

        client.create_example(
            inputs={"question": row['instruction']},
            outputs={
                "expected_intent": mapping[row['intent']],
                "expected_sentiment": expected_sentiment,
                "expected_resolution": row['response']
            },
            metadata={
                "source_intent": row['intent'],
                "flags": row['flags'],
                "is_smoke_test": True
            },
            dataset_id=dataset.id
        )

    print(f"✅ Success! Created '{dataset_name}' with {len(sample_df)} test cases.")

if __name__ == "__main__":
    create_ecommerce_eval_set()
