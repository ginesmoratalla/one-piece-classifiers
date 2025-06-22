import torch
import torch.nn as nn

# Optimization algorithms, e.g., sgd, adam
import torch.optim as optim
import torch.nn.functional as F

from loader.dataloader import get_data_loaders

device = "mps"


class CNN(nn.Module):

    def __init__(self, num_classes) -> None:
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*34*34, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16*34*34)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


num_classes = 18
lr = 0.001
num_epochs = 20


train_loader, test_loader = get_data_loaders(batch_size=150)

model = CNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)


def train_model():

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        print(f"[TRAINING] epoch {epoch}")
        for i, (data, targets) in enumerate(train_loader):

            data = data.float().to(device)

            # data = data.to(device)
            targets = targets.to(device)

            # forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss {loss.item():.4f}")

    print("Model finished training.\n")


def check_accuracy(loader, model, training):

    if training:
        print("[EVAL] Checking accuracy on training dataset")
    else:
        print("[EVAL] Checking accuracy on test dataset")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            x = x.float().to(device)

            scores = model(x)
            predictions = torch.argmax(scores, dim=1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"[EVAL] Got correct {
              num_correct} / {num_samples} --> Accuracy {(float(num_correct)/float(num_samples))*100:.2f}%"
        )
        print()

    model.train()


if __name__ == "__main__":
    print("[PRE-TRAIN] Training model...")
    train_model()
    check_accuracy(train_loader, model, training=True)
    check_accuracy(test_loader, model, training=False)
