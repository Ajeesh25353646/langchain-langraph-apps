# Data Inventory: Customer Support Datasets

This document provides a comprehensive analysis of the datasets available in the `/data` directory, specifically tailored for testing and evaluating the `main.py` LangGraph support agent.

---

## 1. Agentic AI Customer Support Dataset
**File:** `Agentic_AI_Customer_Support_Dataset_v1.csv`

### Technical Metadata
- **Total Instances:** 1,201
- **Format:** CSV (comma-separated)
- **Primary Use:** End-to-end flow validation, sentiment-to-priority correlation, and escalation logic testing.

### Column Definitions & Data Types
| Column Name | Description | Example Values |
| :--- | :--- | :--- |
| `Ticket_ID` | Unique identifier for the support ticket | TKT-10000 |
| `Customer_Query_Text` | The raw user input (Input for Triage) | "Customer is facing an issue..." |
| `Query_Category` | High-level categorization | Product, Account, Delivery, Billing, Technical |
| `Query_Subcategory` | Specific issue type | Refund Request, Payment Failure, Login Issue |
| `Intent_Detected` | The expected intent for the agent | Cancel Service, Request Info, Get Help, Complaint |
| `Sentiment_Score` | Numeric sentiment analysis (-1.0 to 1.0) | 0.63, -0.91 |
| `Priority_Level` | Calculated urgency | Low, Medium, High, Critical |
| `Human_Escalation_Flag` | Whether the case was sent to a human | True, False |
| `Resolution_Status` | The final state of the ticket | Resolved, Pending, Escalated |

### Distribution Analysis
- **Top Categories:** Product (21%), Account (20%), Technical (19%), Billing (19%), Delivery (19%).
- **Common Subcategories:** Refund Request, Payment Failure, General Inquiry, Login Issue, Bug Report.

Value counts for Query_Category:
Query_Category
Product      255
Account      240
Technical    238
Billing      235
Delivery     232
Name: count, dtype: int64

Value counts for Query_Subcategory:
Query_Subcategory
Refund Request     270
Payment Failure    241
General Inquiry    234
Bug Report         229
Login Issue        226
Name: count, dtype: int64

Value counts for Customer_Query_Text:
Customer_Query_Text
Customer is facing an issue and needs support.    1200
Name: count, dtype: int64

Value counts for Intent_Detected:
Intent_Detected
Complaint         335
Get Help          305
Cancel Service    291
Request Info      269
Name: count, dtype: int64

Value counts for Priority_Level:
Priority_Level
Low         310
Medium      299
Critical    296
High        295
Name: count, dtype: int64

Value counts for Human_Escalation_Flag:
Human_Escalation_Flag
False    621
True     579
Name: count, dtype: int64

Value counts for Resolution_Status:
Resolution_Status
Resolved     413
Escalated    405
Pending      382
Name: count, dtype: int64

Value counts for Customer_Feedback_Rating:
Customer_Feedback_Rating
2    275
5    238
4    234
3    232
1    221
Name: count, dtype: int64

Value counts for Follow_Up_Required:
Follow_Up_Required
False    606
True     594
---

## 2. Bitext Customer Support Dataset
**Source:** Bitext (Sample via `preprocess_bitext.py`)

### Technical Metadata
- **Total Instances:** 26,872
- **Format:** CSV/JSON (via preprocessing)
- **Primary Use:** Stress testing linguistic variations (slang, typos, offensive language) and intent classification accuracy.

### Column Definitions
- **`instruction`**: The user's query (e.g., "how can i activaet my card").
- **`flags`**: Linguistic markers (the "Brain" of the dataset).
- **`category`**: Semantic bucket (ACCOUNT, REFUND, etc.).
- **`intent`**: Granular intent (e.g., `check_refund_policy`).
- **`response`**: Ground-truth response for RAG benchmarking.

### Linguistic Breakdown (Flags)
- **Colloquial (Q) [33%]:** "can u help me"
- **Typos (Z) [19%]:** "activaet", "cancep"
- **Offensive (W) [4.8%]:** Contains profanity to test `triage` safety/sentiment.
- **Keyword (K) [8%]:** "activate SIM", "new SIM"

--- 1. Column Overview ---
['flags', 'instruction', 'category', 'intent', 'response']

--- 2. High-Level Categories (Top 10) ---
category
ACCOUNT         5986
ORDER           3988
REFUND          2992
CONTACT         1999
INVOICE         1999
PAYMENT         1998
FEEDBACK        1997
DELIVERY        1994
SHIPPING        1970
SUBSCRIPTION     999
CANCEL           950
Name: count, dtype: int64

--- 3. Top 10 Intents ---
intent
contact_customer_service    1000
complaint                   1000
check_invoice               1000
switch_account              1000
edit_account                1000
contact_human_agent          999
check_payment_methods        999
delivery_period              999
newsletter_subscription      999
get_invoice                  999
Name: count, dtype: int64

--- 4. Linguistic Flags (The 'Brain' of Bitext) ---
P (Polite): 1329 records (4.9%)
Q (Colloquial): 8968 records (33.4%)
W (Offensive): 1288 records (4.8%)
Z (Typos/Errors): 5286 records (19.7%)
K (Keyword mode): 2227 records (8.3%)
B (Basic structure): 26872 records (100.0%)
I (Interrogative): 7839 records (29.2%)
N (Negation): 463 records (1.7%)

--- 5. Complexity Stats ---
Average Instruction Length: 46.9 chars
Average Response Length: 634.1 chars

--- 6. Sample of Offensive Language (W) ---
1. help me to cancel my  last goddamn purchase
2. i bought some damn item help to cancep order {{Order Number}}
3. i purchased some fucking item cancel order {{Order Number}}

---

## 3. Banking77 Dataset
**Source:** Banking77 (Domain-Specific)

### Technical Metadata
- **Total Instances:** ~13,000
- **Primary Use:** Specialized RAG testing for financial queries.

### Intent Mapping for `main.py`
The system maps 77 granular banking intents into the three core categories used in `main.py`:
- **`refund`**: `refund_not_showing_up`, `top_up_failed`, `cancel_transfer`.
- **`technical_issue`**: `card_not_working`, `pin_blocked`, `lost_or_stolen_card`.
- **`inquiry`**: `exchange_rate`, `atm_support`, `transfer_fee_charged`.

---

## Testing Relevance Matrix

| Test Scenario | Recommended Dataset | Reason |
| :--- | :--- | :--- |
| **Triage Routing** | Agentic AI / Bitext | High variety of "Expected Intent" vs "Raw Text". |
| **Refund Thresholds** | Agentic AI | Includes explicit "Refund Request" subcategories. |
| **Linguistic Robustness**| Bitext | Large volume of Typos (Z) and Colloquialisms (Q). |
| **RAG Performance** | Banking77 | Domain-specific queries to test KB retrieval accuracy. |
| **Escalation Logic** | Agentic AI | Contains manual escalation flags for comparison. |

---
*Last Updated: March 2026*
