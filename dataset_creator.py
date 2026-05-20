import pandas as pd
import random

data = []

# Generate synthetic samples
for _ in range(100000):

    # Random video confidence
    video_confidence = random.randint(30, 100)

    # Random text confidence
    text_confidence = random.randint(30, 100)

    # Simulated final confidence
    final_confidence = int(
        0.7 * video_confidence +
        0.3 * text_confidence +
        random.randint(-5, 5)
    )

    # Clamp values
    final_confidence = max(0, min(100, final_confidence))

    data.append([
        video_confidence,
        text_confidence,
        final_confidence
    ])

# Create dataframe
df = pd.DataFrame(
    data,
    columns=[
        "video_confidence",
        "text_confidence",
        "final_confidence"
    ]
)

# Save CSV
df.to_csv("confidence_dataset.csv", index=False)

print("Dataset Created Successfully")
print(df.head())