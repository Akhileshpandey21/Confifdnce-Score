from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Example labels
actual = [1,1,0,1,0,0,1,0]

# Prediction probabilities
predicted_probabilities = [
    0.95,
    0.85,
    0.20,
    0.60,
    0.30,
    0.40,
    0.90,
    0.10
]

# ROC
fpr, tpr, thresholds = roc_curve(
    actual,
    predicted_probabilities
)

roc_auc = auc(fpr, tpr)

# Plot
plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("roc_curve.png")

plt.show()