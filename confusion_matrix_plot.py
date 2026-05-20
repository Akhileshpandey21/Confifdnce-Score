from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Example data
actual = [1,1,0,1,0,0,1,0]
predicted = [1,1,0,0,0,1,1,0]

# Generate confusion matrix
cm = confusion_matrix(actual, predicted)

# Plot
plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0,1], ["Nervous","Confident"])
plt.yticks([0,1], ["Nervous","Confident"])

# Add values
for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i,j],
            ha='center',
            va='center'
        )

plt.colorbar()

plt.savefig("confusion_matrix.png")

plt.show()