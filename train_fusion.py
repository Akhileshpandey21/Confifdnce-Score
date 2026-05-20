import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("confidence_dataset.csv")

# Features
X = df[[
    "video_confidence",
    "text_confidence"
]].values

# Labels
y = df[[
    "final_confidence"
]].values

# Normalize data
scaler = MinMaxScaler()

X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Fusion model
class FusionNetwork(nn.Module):

    def __init__(self):

        super(FusionNetwork, self).__init__()

        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(16, 8)

        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x

# Initialize model
model = FusionNetwork()

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training loop
epochs = 500

for epoch in range(epochs):

    # Forward pass
    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch}, Loss: {loss.item():.6f}"
        )

# Save model
torch.save(
    model.state_dict(),
    "fusion_model.pth"
)

print("Training Complete")



# A Multimodal AI Framework for Real-Time Confidence and Emotion Analysis Using Facial Behavior and Textual Semantics