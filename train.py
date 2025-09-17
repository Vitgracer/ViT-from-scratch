import torch
import torch.nn as nn
import torch.optim as optim
from model.cnn.simple_cnn import SimpleCNN
from dataset import get_dataloaders


def train_step(criterion, optimizer, model, device):
    model.train()
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def val_step(criterion, model, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss

def save_model(model):
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    example_input = torch.randn(1, 1, 28, 28).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("digit-predictor-cpu.pt")
    print("Model saved as digit-predictor-cpu.pt")

def train(model, trainloader, valloader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(7):
        running_loss = train_step(criterion, optimizer, model, device)
        val_loss = val_step(criterion, model, device)

        print(f"Epoch {epoch+1}, \
                Train Loss: {running_loss / len(trainloader):.4f}, \
                Val Loss: {val_loss / len(valloader):.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN().to(device)
    trainloader, valloader = get_dataloaders()

    train(model, trainloader, valloader, device)