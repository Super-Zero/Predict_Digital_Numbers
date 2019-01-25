import torch
import helper as hp
from torch import nn
from torchvision import datasets, transforms
from torch import optim

# Define a transform to normalize data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Download and load training data
trainset = datasets.MNIST('./data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Implement model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
)

# Set a loss error or Cross-Entropy
criterion = nn.NLLLoss()
# Set a learning reate variable
optimizer = optim.SGD(model.parameters(), lr=0.003)

# Train model
epochs = 1

for e in range(epochs):
    running_loss = 0

    for images, labels in trainloader:

        # Flat MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        # Fordward propagation
        output = model.forward(images)

        loss = criterion(output, labels)

        # Backward propagation
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")

# Test model with predictions
while 1:
    # Get image from dataset
    images, labels = next(iter(trainloader))
    # Flat the image of 28x28 to a vector of 784 values
    img = images[0].view(1, 784)

    # Pass image to model to predict number
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)

    # Visualize results
    hp.view_classify(img.view(1, 28, 28), ps)

    # Ask user to predict other number
    ans = input("Would you like to predict again? (Yes/No): ")
    if (ans == "n"):
        break
