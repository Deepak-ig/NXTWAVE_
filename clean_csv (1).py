import pandas as pd

# Load your landmarks CSV
df = pd.read_csv("data/landmarks.csv")

print("Before cleaning:", df.shape)

# Drop any row that has a missing (NaN) value
df = df.dropna()

print("After cleaning:", df.shape)

# Save back to the same file (or a new file if you want to keep original)
df.to_csv("data/landmarks.csv", index=False)
print("Cleaned file saved.")
