import pandas as pd

golden = pd.read_csv("golden_set.csv")

print("Unique intent names:")
for intent in sorted(golden["intent"].unique()):
    print(repr(intent))

print("\nCounts:")
print(golden["intent"].value_counts())