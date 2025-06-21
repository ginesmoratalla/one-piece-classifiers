import torch
import torch.nn as nn

# Optimization algorithms, e.g., sgd, adam
import torch.optim as optim

from loader.dataloader import get_data_loaders

device = "mps"


class FCN(nn.Module):
    def __init__(self, input_size, num_classes) -> None:
        super(FCN, self).__init__()

        self.netw = nn.Sequential(

            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.netw(x)


input_dim = 150 * 150 * 3
num_classes = 18
lr = 0.001
num_epochs = 20


train_loader, test_loader = get_data_loaders(batch_size=150)

model = FCN(input_size=input_dim, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)


def train_model():

    for epoch in range(num_epochs):
        print(f"[TRAINING] epoch {epoch}")
        for batch_idx, (data, targets) in enumerate(train_loader):

            data = data.reshape(data.shape[0], -1).float().to(device)
            # print(f"DEBUG x shape -> {data.shape} {data.dtype}")
            # exit(0)

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

    print("Model finished training.\n")


def check_accuracy(loader, model, train):

    if train:
        print("[EVAL] Checking accuracy on training dataset")
    else:
        print("[EVAL] Checking accuracy on test dataset")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            x = x.reshape(x.shape[0], -1).float().to(device)

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


print(f"{model}\nTraining model...")
train_model()
check_accuracy(train_loader, model, True)
check_accuracy(test_loader, model, False)
