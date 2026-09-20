import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from baseline_rules import predict_intent
from agent import run_agent

golden = pd.read_csv("golden_set.csv")

print("Golden set rows:", len(golden))
print("Columns:", list(golden.columns))

print("\nIntent distribution:")
print(golden["intent"].value_counts())

def calculate_metrics(actual, predicted):
    accuracy = accuracy_score(actual, predicted)

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )

    return accuracy, precision, recall, f1

majority_intent = golden["intent"].mode()[0]

predictions = [majority_intent] * len(golden)

accuracy, precision, recall, f1 = calculate_metrics(
    golden["intent"],
    predictions
)

print("\nMajority Baseline Evaluation")
print("----------------------------")
print("Majority intent:", majority_intent)
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

rule_predictions = [
    predict_intent(text)
    for text in golden["customer_text"]
]

accuracy, precision, recall, f1 = calculate_metrics(
    golden["intent"],
    rule_predictions
)

print("\nRule-Based Baseline Evaluation")
print("------------------------------")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

results = golden[["customer_text", "intent"]].copy()

results["predicted_intent"] = rule_predictions

wrong = results[results["intent"] != results["predicted_intent"]]

print("\nTop 5 Failure Patterns")
print("----------------------")

top_failures = (
    wrong.groupby(["intent", "predicted_intent"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(5)
)

print(top_failures.to_string(index=False))

# Agent evaluation results will be stored here
agent_results = []

test_sample = (
    golden.groupby("intent", group_keys=False)
    .sample(n=2, random_state=42)
)

print("\nAgent evaluation sample distribution:")
print(test_sample["intent"].value_counts())

print("\nAgent evaluation test")
print("---------------------")
print("Examples selected for potential evaluation:", len(test_sample))


# Gemini evaluation temporarily skipped because the free-tier quota is exhausted.
print("\nGemini evaluation skipped - quota exhausted.")

agent_results_df = pd.DataFrame(agent_results)

agent_results_df = pd.read_csv("agent_results.csv")

print("Saved agent results:", len(agent_results_df))

agent_accuracy = accuracy_score(
    agent_results_df["actual_intent"],
    agent_results_df["predicted_intent"]
)

agent_precision, agent_recall, agent_f1, _ = precision_recall_fscore_support(
    agent_results_df["actual_intent"],
    agent_results_df["predicted_intent"],
    average="macro",
    zero_division=0
)

print("\nGemini Agent Metrics (saved evaluation sample)")
print("------------------------------------------------")
print(f"Accuracy:  {agent_accuracy:.4f}")
print(f"Precision: {agent_precision:.4f}")
print(f"Recall:    {agent_recall:.4f}")
print(f"F1 Score:  {agent_f1:.4f}")

comparison = pd.DataFrame({
    "Model": [
        "Majority Baseline",
        "Rule-Based Baseline",
        "Gemini Agent"
    ],
    "Accuracy": [
        0.25,
        0.425,
        agent_accuracy
    ],

    "Macro Precision": [
        0.0357,
        0.7361,
        agent_precision
    ],

    "Macro Recall": [
        0.1429,
        0.3976,
        agent_recall
    ],

    "Macro F1": [
        0.0571,
        0.3915,
        agent_f1
    ],
})

print("\nBaseline Comparison")
print("-------------------")
print(comparison.to_string(index=False))

comparison.to_csv("baseline_results.csv", index=False)

print("\nBaseline results saved to baseline_results.csv")

comparison.to_csv("model_comparison.csv", index=False)

print("\nModel comparison saved to model_comparison.csv")

