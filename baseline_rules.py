import pandas as pd


# Load the golden set
golden = pd.read_csv("golden_set.csv")

print("Golden set loaded successfully.")
print("Number of examples:", len(golden))

print("\nIntent distribution:")
print(golden["intent"].value_counts())

rules = {
    "Playback": [
        "shuffle", "repeat", "skip", "skipping",
        "playback", "pause", "paused", "stops playing",
        "no sound"
    ],

    "Technical/App Issue": [
        "crash", "crashes", "crashed", "crashing",
        "app", "iphone", "ipad", "android", "ios",
        "windows", "browser", "version", "update",
        "install", "freeze", "freezes"
    ],

    "Content Availability": [
        "missing", "unavailable", "availability",
        "removed", "not available", "can't find",
        "greyed out", "not showing"
    ],

    "Subscription/Payment": [
        "premium", "subscription", "subscribe",
        "payment", "billing", "charged", "charge",
        "refund", "receipt", "paid"
    ],

    "Ads/Promotions": [
        "ad", "ads", "advertisement", "advertising",
        "promotion", "offer", "sale", "eligible"
    ],

    "Feature Request": [
        "feature", "request", "suggestion",
        "add", "option", "ability",
        "would like", "wish", "can you add"
    ]
}

print("\nRules created successfully.")
print("Number of intent rules:", len(rules))

def predict_intent(text):
    text = text.lower()

    scores = {}

    for intent, keywords in rules.items():
        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        scores[intent] = score

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] == 0:
        return "Other/General"

    return best_intent

print("\nSample predictions:")

for text in golden["customer_text"].head(10):
    print("\nMessage:", text)
    print("Predicted:", predict_intent(text))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

predictions = [
    predict_intent(text)
    for text in golden["customer_text"]
]

actual = golden["intent"]

accuracy = accuracy_score(actual, predictions)

precision, recall, f1, _ = precision_recall_fscore_support(
    actual,
    predictions,
    average="macro",
    zero_division=0
)

print("\nRule-Based Baseline Results")
print("---------------------------")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))    

results = golden[["customer_text", "intent"]].copy()
results["predicted_intent"] = predictions

wrong = results[
    results["intent"] != results["predicted_intent"]
]

print("\nNumber of incorrect predictions:", len(wrong))

print("\nFirst 10 incorrect predictions:")
print(
    wrong[
        ["customer_text", "intent", "predicted_intent"]
    ].head(10).to_string(index=False)
)

print("\nMost common confusion pairs:")

confusion = (
    wrong.groupby(["intent", "predicted_intent"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(confusion.to_string(index=False))