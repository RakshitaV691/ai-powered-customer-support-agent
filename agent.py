from google import genai

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = genai.Client()

historical_data = pd.read_csv("spotify_pairs_1000_fixed.csv")

vectorizer = TfidfVectorizer(stop_words="english")
historical_vectors = vectorizer.fit_transform(
    historical_data["customer_text"]
)

def normalize_intent(raw_intent):
    valid_intents = [
        "Playback",
        "Technical/App Issue",
        "Content Availability",
        "Subscription/Payment",
        "Ads/Promotions",
        "Feature Request",
        "Other/General"
    ]

    raw_intent = raw_intent.strip()

    for intent in valid_intents:
        if intent.lower() == raw_intent.lower():
            return intent

    return "Other/General"


def classify_intent(customer_message):
    prompt = f"""
You are a customer support intent classifier for SpotifyCares.

Classify the customer message into exactly ONE of these intents:

1. Playback
2. Technical/App Issue
3. Content Availability
4. Subscription/Payment
5. Ads/Promotions
6. Feature Request
7. Other/General

Return ONLY the exact intent name. Do not explain your answer.

Customer message:
{customer_message}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return normalize_intent(response.text)

def decide_action(customer_message, intent):
    prompt = f"""
You are deciding whether a Spotify customer support request
should be handled automatically or escalated to a human agent.

Intent: {intent}

Customer message:
{customer_message}

Use these rules:

AUTO-HANDLE:
- Common playback problem
- Common app/device troubleshooting
- General content availability question
- General feature request
- General ads/promotions question

ESCALATE:
- Account-specific issue
- Payment or billing problem
- Refund or charge dispute
- Private account information is required
- The request is too unclear to determine the problem
- High-risk or sensitive situation

Return ONLY one of:
Auto-handle
Escalate
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip()

def get_escalation_reason(customer_message, intent):
    prompt = f"""
You are explaining why a Spotify customer support request
needs to be escalated to a human agent.

Intent: {intent}

Customer message:
{customer_message}

Choose exactly ONE reason:

- Account-specific issue
- Payment or billing problem
- Refund or charge dispute
- Private account information is required
- Unclear customer request
- High-risk or sensitive situation
- Requires human troubleshooting
- Feature request requiring human review
- Unresolved issue with insufficient context

Return ONLY the exact reason.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip()


print("Historical rows:", len(historical_data))
print("Historical columns:", historical_data.columns.tolist())
print("TF-IDF matrix shape:", historical_vectors.shape)

def retrieve_historical_examples(customer_message, top_k=3):
    query_vector = vectorizer.transform([customer_message])

    similarities = cosine_similarity(
        query_vector,
        historical_vectors
    )[0]

    historical_messages = historical_data["customer_text"].tolist()

    for i, message in enumerate(historical_messages):
        if message.strip().lower() == customer_message.strip().lower():
            similarities[i] = -1

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = historical_data.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results




def generate_reply(customer_message):
    examples = retrieve_historical_examples(customer_message, top_k=3)

    historical_context = ""

    for _, row in examples.iterrows():
        historical_context += f"""
Historical customer message:
{row["customer_text"]}

Historical Spotify reply:
{row["spotify_reply"]}

"""

    prompt = f"""
You are a Spotify customer support agent.

Draft a helpful reply to the customer using the historical
Spotify support examples provided below as grounding.

Customer message:
{customer_message}

Historical examples:
{historical_context}

Instructions:
- Use the historical replies as guidance for tone and troubleshooting.
- Do not invent account-specific information.
- Do not claim that you performed an action you cannot perform.
- Do not copy a historical reply word-for-word.
- Adapt the response to the customer's actual message.
- Keep the reply concise and natural.
- Return ONLY the customer-facing reply.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip()


def run_agent(customer_message):
    historical_examples = retrieve_historical_examples(
        customer_message,
        top_k=3
    )

    historical_context = ""

    for _, row in historical_examples.iterrows():
        historical_context += (
            f"Customer: {row['customer_text']}\n"
            f"Support reply: {row['spotify_reply']}\n\n"
        )
 
    prompt = f"""

You are a customer support agent for SpotifyCares.

Customer message:
{customer_message}

Historical support examples:
{historical_context}

Use these historical examples as general grounding and style guidance.
Do not copy them verbatim. Do not invent account-specific information,
request payment details, or claim that an action was performed.

Classify and handle the message.

Return exactly these four lines:

Intent: <one of the seven intents>
Action: <Auto-handle or Escalate>
Reason: <escalation reason, or None if Auto-handle>
Reply: <short customer-support reply>

Seven intents:
- Playback
- Technical/App Issue
- Content Availability
- Subscription/Payment
- Ads/Promotions
- Feature Request
- Other/General

Handling policy:
Auto-handle common playback problems, common app/device troubleshooting,
general content availability questions, general feature requests,
and general ads/promotions questions.

Escalate account-specific issues, payment/billing problems,
refund or charge disputes, requests requiring private account information,
unclear requests, high-risk or sensitive situations,
unresolved issues requiring human troubleshooting,
and feature requests requiring human review.

Allowed escalation reasons:
- Account-specific issue
- Payment or billing problem
- Refund or charge dispute
- Private account information is required
- Unclear customer request
- High-risk or sensitive situation
- Requires human troubleshooting
- Feature request requiring human review
- Unresolved issue with insufficient context

For the reply, use the historical support examples provided by this project
as general style and knowledge. Do not invent account-specific information, request payment details,
or claim that an action was performed.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    text = response.text.strip()

    intent = "Other/General"
    action = "Auto-handle"
    reason = None
    reply = ""

    for line in text.splitlines():
        if line.startswith("Intent:"):
            intent = normalize_intent(line.replace("Intent:", "", 1).strip())

        elif line.startswith("Action:"):
            action = line.replace("Action:", "", 1).strip()

        elif line.startswith("Reason:"):
            reason = line.replace("Reason:", "", 1).strip()
            if reason.lower() == "none":
                reason = None

        elif line.startswith("Reply:"):
            reply = line.replace("Reply:", "", 1).strip()

    return intent, action, reason, reply

if __name__ == "__main__":
    message = "I was charged twice for my Spotify Premium subscription"

    intent, action, reason, reply = run_agent(message)

    print("\nCustomer message:", message)
    print("Predicted intent:", intent)
    print("Expected action:", action)
    print("Escalation reason:", reason)
    print("Draft reply:", reply)