import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

if __name__ == "__main__":
    # Load datasets
    train_data = datasets.ImageFolder("1000_videos/train", transform=transform_train)
    val_data   = datasets.ImageFolder("1000_videos/validation", transform=transform_train)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 12
    best_val = 0.0
    patience = 2      # allowed epochs where val does NOT improve
    wait = 0          # counter for early stopping

    for epoch in range(EPOCHS):
        # TRAINING
        model.train()
        running_loss = 0
        correct = 0

        for imgs, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()

        train_acc = correct / len(train_data)

        # VALIDATION
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()

        val_acc = val_correct / len(val_data)

        print(f"Epoch {epoch+1}: Train Loss={running_loss:.3f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # CHECK IF VALIDATION IMPROVED
        if val_acc > best_val:
            best_val = val_acc
            wait = 0  # reset patience counter
            torch.save(model.state_dict(), f"best_epoch_{epoch+1}.pth")
            print(f"==== Saved BEST model at Epoch {epoch+1} (Val Acc={val_acc:.4f}) ====")
        else:
            wait += 1
            print(f"No improvement. Patience progress: {wait}/{patience}")

        # EARLY STOPPING TRIGGER
        if wait >= patience:
            print(f"Early stopping activated at Epoch {epoch+1}. Best Val Acc = {best_val:.4f}")
            break

    print("Training finished.")
    print("Best saved model is the epoch with highest Val Acc.")
