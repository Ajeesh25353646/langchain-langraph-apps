# Dataset Analysis & Overview

This directory contains datasets and scripts used to evaluate the performance of the agentic customer support system.

## 1. Agentic AI Customer Support Dataset (`Agentic_AI_Customer_Support_Dataset_v1.csv`)
This is a synthetic dataset designed to test end-to-end agentic flows, including triage, sentiment analysis, and escalation.

- **Number of Instances:** 1,200+ records.
- **Columns:** 
    - `Ticket_ID`, `Customer_ID`, `Customer_Name`, `Customer_Email`, `Customer_Country`, `Customer_Language`, `Channel`
    - `Query_Category`: Product, Account, Delivery, Billing, Technical
    - `Query_Subcategory`: Refund Request, Payment Failure, General Inquiry, Login Issue, Bug Report
    - `Customer_Query_Text`: The raw input query.
    - `Intent_Detected`: Cancel Service, Request Info, Get Help, Complaint
    - `Sentiment_Score`, `Priority_Level`, `AI_Response_Text`, `Human_Escalation_Flag`, `Resolution_Status`, `Resolution_Time`, `Customer_Feedback_Rating`, `Follow_Up_Required`, `Timestamp`
- **Key Testing Use Cases:**
    - Testing `triage_agent_node` accuracy.
    - Testing `escalation_agent_node` logic based on `Human_Escalation_Flag`.
    - Analyzing sentiment-to-priority mapping.

## 2. Bitext Customer Support Dataset
A comprehensive tagged training dataset for LLM-based virtual assistants.

- **Number of Instances:** 26,872 question/answer pairs.
- **Columns:** 
    - `flags`: Linguistic variations (Q: Colloquial, W: Offensive, Z: Typos, P: Polite, etc.)
    - `instruction`: User request text.
    - `category`: High-level semantic category (ACCOUNT, ORDER, REFUND, etc.)
    - `intent`: Granular intent (e.g., `get_refund`, `recover_password`).
    - `response`: Gold-standard assistant response.
- **Linguistic Breakdown:**
    - **Colloquial (Q):** ~33% of data.
    - **Typos (Z):** ~20% of data.
    - **Offensive (W):** ~5% of data.
- **Key Testing Use Cases:**
    - Stress-testing the agent with typos and colloquialisms.
    - Benchmarking intent detection against 27 granular categories.

## 3. Banking77 Dataset (Banking Categories)
Granular banking-specific intents used for specialized domain testing.

- **Number of Instances:** ~13,000 (Source: Banking77).
- **Columns:** `text` (query), `label` (intent).
- **Relevant Categories:**

[
  "card_arrival",
  "card_linking",
  "exchange_rate",
  "card_payment_wrong_exchange_rate",
  "extra_charge_on_statement",
  "pending_cash_withdrawal",
  "fiat_currency_support",
  "card_delivery_estimate",
  "automatic_top_up",
  "card_not_working",
  "exchange_via_app",
  "lost_or_stolen_card",
  "age_limit",
  "pin_blocked",
  "contactless_not_working",
  "top_up_by_bank_transfer_charge",
  "pending_top_up",
  "cancel_transfer",
  "top_up_limits",
  "wrong_amount_of_cash_received",
  "card_payment_fee_charged",
  "transfer_not_received_by_recipient",
  "supported_cards_and_currencies",
  "getting_virtual_card",
  "card_acceptance",
  "top_up_reverted",
  "balance_not_updated_after_cheque_or_cash_deposit",
  "card_payment_not_recognised",
  "edit_personal_details",
  "why_verify_identity",
  "unable_to_verify_identity",
  "get_physical_card",
  "visa_or_mastercard",
  "topping_up_by_card",
  "disposable_card_limits",
  "compromised_card",
  "atm_support",
  "direct_debit_payment_not_recognised",
  "passcode_forgotten",
  "declined_cash_withdrawal",
  "pending_card_payment",
  "lost_or_stolen_phone",
  "request_refund",
  "declined_transfer",
  "Refund_not_showing_up",
  "declined_card_payment",
  "pending_transfer",
  "terminate_account",
  "card_swallowed",
  "transaction_charged_twice",
  "verify_source_of_funds",
  "transfer_timing",
  "reverted_card_payment?",
  "change_pin",
  "beneficiary_not_allowed",
  "transfer_fee_charged",
  "receiving_money",
  "failed_transfer",
  "transfer_into_account",
  "verify_top_up",
  "getting_spare_card",
  "top_up_by_cash_or_cheque",
  "order_physical_card",
  "virtual_card_not_working",
  "wrong_exchange_rate_for_cash_withdrawal",
  "get_disposable_virtual_card",
  "top_up_failed",
  "balance_not_updated_after_bank_transfer",
  "cash_withdrawal_not_recognised",
  "exchange_charge",
  "top_up_by_card_charge",
  "activate_my_card",
  "cash_withdrawal_charge",
  "card_about_to_expire",
  "apple_pay_or_google_pay",
  "verify_my_identity",
  "country_support"
]


# Bitext - Customer Service Tagged Training Dataset for LLM-based Virtual Assistants
Overview
This dataset can be used to train Large Language Models such as GPT, Llama2 and Falcon, both for Fine Tuning and Domain Adaptation.

The dataset has the following specs:

Use Case: Intent Detection
Vertical: Customer Service
27 intents assigned to 10 categories
26872 question/answer pairs, around 1000 per intent
30 entity/slot types
12 different types of language generation tags
The categories and intents have been selected from Bitext's collection of 20 vertical-specific datasets, covering the intents that are common across all 20 verticals. The verticals are:

Automotive, Retail Banking, Education, Events & Ticketing, Field Services, Healthcare, Hospitality, Insurance, Legal Services, Manufacturing, Media Streaming, Mortgages & Loans, Moving & Storage, Real Estate/Construction, Restaurant & Bar Chains, Retail/E-commerce, Telecommunications, Travel, Utilities, Wealth Management
For a full list of verticals and its intents see https://www.bitext.com/chatbot-verticals/.

The question/answer pairs have been generated using a hybrid methodology that uses natural texts as source text, NLP technology to extract seeds from these texts, and NLG technology to expand the seed texts. All steps in the process are curated by computational linguists.

Dataset Token Count
The dataset contains an extensive amount of text data across its 'instruction' and 'response' columns. After processing and tokenizing the dataset, we've identified a total of 3.57 million tokens. This rich set of tokens is essential for training advanced LLMs for AI Conversational, AI Generative, and Question and Answering (Q&A) models.

Fields of the Dataset
Each entry in the dataset contains the following fields:

flags: tags (explained below in the Language Generation Tags section)
instruction: a user request from the Customer Service domain
category: the high-level semantic category for the intent
intent: the intent corresponding to the user instruction
response: an example expected response from the virtual assistant
Categories and Intents
The categories and intents covered by the dataset are:

ACCOUNT: create_account, delete_account, edit_account, recover_password, registration_problems, switch_account
CANCELLATION_FEE: check_cancellation_fee
CONTACT: contact_customer_service, contact_human_agent
DELIVERY: delivery_options, delivery_period
FEEDBACK: complaint, review
INVOICE: check_invoice, get_invoice
ORDER: cancel_order, change_order, place_order, track_order
PAYMENT: check_payment_methods, payment_issue
REFUND: check_refund_policy, get_refund, track_refund
SHIPPING_ADDRESS: change_shipping_address, set_up_shipping_address
SUBSCRIPTION: newsletter_subscription
Entities
The entities covered by the dataset are:

{{Order Number}}, typically present in:
Intents: cancel_order, change_order, change_shipping_address, check_invoice, check_refund_policy, complaint, delivery_options, delivery_period, get_invoice, get_refund, place_order, track_order, track_refund
{{Invoice Number}}, typically present in:
Intents: check_invoice, get_invoice
{{Online Order Interaction}}, typically present in:
Intents: cancel_order, change_order, check_refund_policy, delivery_period, get_refund, review, track_order, track_refund
{{Online Payment Interaction}}, typically present in:
Intents: cancel_order, check_payment_methods
{{Online Navigation Step}}, typically present in:
Intents: complaint, delivery_options
{{Online Customer Support Channel}}, typically present in:
Intents: check_refund_policy, complaint, contact_human_agent, delete_account, delivery_options, edit_account, get_refund, payment_issue, registration_problems, switch_account
{{Profile}}, typically present in:
Intent: switch_account
{{Profile Type}}, typically present in:
Intent: switch_account
{{Settings}}, typically present in:
Intents: cancel_order, change_order, change_shipping_address, check_cancellation_fee, check_invoice, check_payment_methods, contact_human_agent, delete_account, delivery_options, edit_account, get_invoice, newsletter_subscription, payment_issue, place_order, recover_password, registration_problems, set_up_shipping_address, switch_account, track_order, track_refund
{{Online Company Portal Info}}, typically present in:
Intents: cancel_order, edit_account
{{Date}}, typically present in:
Intents: check_invoice, check_refund_policy, get_refund, track_order, track_refund
{{Date Range}}, typically present in:
Intents: check_cancellation_fee, check_invoice, get_invoice
{{Shipping Cut-off Time}}, typically present in:
Intent: delivery_options
{{Delivery City}}, typically present in:
Intent: delivery_options
{{Delivery Country}}, typically present in:
Intents: check_payment_methods, check_refund_policy, delivery_options, review, switch_account
{{Salutation}}, typically present in:
Intents: cancel_order, check_payment_methods, check_refund_policy, create_account, delete_account, delivery_options, get_refund, recover_password, review, set_up_shipping_address, switch_account, track_refund
{{Client First Name}}, typically present in:
Intents: check_invoice, get_invoice
{{Client Last Name}}, typically present in:
Intents: check_invoice, create_account, get_invoice
{{Customer Support Phone Number}}, typically present in:
Intents: change_shipping_address, contact_customer_service, contact_human_agent, payment_issue
{{Customer Support Email}}, typically present in:
Intents: cancel_order, change_shipping_address, check_invoice, check_refund_policy, complaint, contact_customer_service, contact_human_agent, get_invoice, get_refund, newsletter_subscription, payment_issue, recover_password, registration_problems, review, set_up_shipping_address, switch_account
{{Live Chat Support}}, typically present in:
Intents: check_refund_policy, complaint, contact_human_agent, delete_account, delivery_options, edit_account, get_refund, payment_issue, recover_password, registration_problems, review, set_up_shipping_address, switch_account, track_order
{{Website URL}}, typically present in:
Intents: check_payment_methods, check_refund_policy, complaint, contact_customer_service, contact_human_agent, create_account, delete_account, delivery_options, get_refund, newsletter_subscription, payment_issue, place_order, recover_password, registration_problems, review, switch_account
{{Upgrade Account}}, typically present in:
Intents: create_account, edit_account, switch_account
{{Account Type}}, typically present in:
Intents: cancel_order, change_order, change_shipping_address, check_cancellation_fee, check_invoice, check_payment_methods, check_refund_policy, complaint, contact_customer_service, contact_human_agent, create_account, delete_account, delivery_options, delivery_period, edit_account, get_invoice, get_refund, newsletter_subscription, payment_issue, place_order, recover_password, registration_problems, review, set_up_shipping_address, switch_account, track_order, track_refund
{{Account Category}}, typically present in:
Intents: cancel_order, change_order, change_shipping_address, check_cancellation_fee, check_invoice, check_payment_methods, check_refund_policy, complaint, contact_customer_service, contact_human_agent, create_account, delete_account, delivery_options, delivery_period, edit_account, get_invoice, get_refund, newsletter_subscription, payment_issue, place_order, recover_password, registration_problems, review, set_up_shipping_address, switch_account, track_order, track_refund
{{Account Change}}, typically present in:
Intent: switch_account
{{Program}}, typically present in:
Intent: place_order
{{Refund Amount}}, typically present in:
Intent: track_refund
{{Money Amount}}, typically present in:
Intents: check_refund_policy, complaint, get_refund, track_refund
{{Store Location}}, typically present in:
Intents: complaint, delivery_options, place_order
Language Generation Tags
The dataset contains tags that reflect how language varies/changes across different linguistic phenomena like colloquial or offensive language. So if an utterance for intent “cancel_order” contains the “COLLOQUIAL” tag, the utterance will express an informal language variation like: “can u cancel my order”.

These tags indicate the type of language variation that the entry expresses. When associated to each entry, they allow Conversational Designers to customize training datasets to different user profiles with different uses of language. Through these tags, many different datasets can be created to make the resulting assistant more accurate and robust. A bot that sells sneakers should be mainly targeted to younger population that use a more colloquial language; while a classical retail banking bot should be able to handle more formal or polite language. The dataset also reflects commonly occurring linguistic phenomena of real-life virtual assistant, such as spelling mistakes, run-on words, punctuation errors…

The dataset contains tagging for all relevant linguistic phenomena that can be used to customize the dataset for different user profiles.

Tags for Lexical variation
M - Morphological variation: inflectional and derivational “is my SIM card active”, “is my SIM card activated”

L - Semantic variations: synonyms, use of hyphens, compounding… “what’s my billing date", “what’s my anniversary date”

Tags for Syntactic structure variation
B - Basic syntactic structure: “activate my SIM card”, “I need to activate my SIM card”

I - Interrogative structure “can you activate my SIM card?”, “how do I activate my SIM card?”

C - Coordinated syntactic structure “I have a new SIM card, what do I need to do to activate it?”

N - Negation “I do not want this item, where to cancel my order?”

Tags for language register variations
P - Politeness variation “could you help me activate my SIM card, please?”

Q - Colloquial variation “can u activ8 my SIM?”

W - Offensive language “I want to talk to a f*&%*g agent”

Tags for stylistic variations
K - Keyword mode "activate SIM", "new SIM"

E - Use of abbreviations: “I'm / I am interested in getting a new SIM”

Z - Errors and Typos: spelling issues, wrong punctuation… “how can i activaet my card”

Other tags not in use in this Dataset
D - Indirect speech “ask my agent to activate my SIM card”

G - Regional variations US English vs UK English: "truck" vs "lorry" France French vs Canadian French: "tchatter" vs "clavarder"

R - Respect structures - Language-dependent variations English: "may" vs "can…" French: "tu" vs "vous..." Spanish: "tú" vs "usted..."

Y - Code switching “activer ma SIM card”



# Model Latency Benchmark

End-to-end latency comparison between the original Gemma 4 31B agent (`main_gemma_31b_v1.py`) and the optimized DeepSeek V4 Flash agent (`main_optimized_v2.py`).

## Methodology

- **Hardware:** RTX 4090 (Qwen3-Embedding-0.6B on CUDA), Intel i9
- **Network:** India-based client → Google AI Studio / OpenCode Zen API (US servers)
- **Queries:** 4 diverse customer support questions (technical + refund intents)
- **Cold start:** First query after fresh app startup (includes retriever initialization + LLM creation)
- **Warm:** Subsequent queries with retriever already in memory

## Latency Comparison

| Scenario | v1 (Gemma 4 31B) | v2 (DeepSeek V4 Flash) | Improvement |
|---|---|---|---|
| **First query (cold start)** | ~51.5s | ~29.1s | **43% faster** |
| **Warm — refund intent** | N/A | **4.2s** | — |
| **Warm — technical (avg)** | ~31s (est.) | **7.9s** | **~75% faster** |

### Optimized v2 — Per-Query Breakdown (warm technical)

| Phase | Latency | Dependency |
|---|---|---|
| Guardrail (Presidio NER + regex) | ~0.1s | CPU |
| PII Detection (Presidio NER) | ~0.1s | CPU |
| **Triage (LLM call)** | **4.3–5.9s** | OpenCode Zen API |
| RAG Retrieval (Qwen3 embed + Pinecone + FlashRank) | 1.7–2.0s | CUDA + Network |
| **Agent Response (LLM call)** | **0.8–1.1s** | OpenCode Zen API |
| **Total** | **6.8–9.0s** | — |

## Optimizations Applied (v2)

1. **Model swap:** Gemma 4 31B (Google AI Studio) → DeepSeek V4 Flash Free (OpenCode Zen). Reduced per-LLM-call latency by ~60%.
2. **Singleton LLM caching:** `_LLM_INSTANCE` module-level singleton eliminates repeated `ChatOpenAI` object creation (~0.4s per node call).
3. **Pinecone index caching:** Sentinel file (`/tmp/.pinecone_index_cache`) skips the `list_indexes()` API call (~1.4–2.6s) on app restarts.
4. **Eager retriever warmup:** Background daemon thread loads Qwen3 embeddings + Pinecone during startup, saving ~16s on the first user query.
5. **Token-level streaming UI:** Combined node-level progress tracking with token-by-token response streaming — the status bar shows execution pipeline progress (guardrail → triage → search → response) while the chat bubble fills in character-by-character with a typing cursor, parallel to what the LLM generates.

## Raw Benchmark Data

See `data/benchmark_v2_results.json` for per-query latency data.
