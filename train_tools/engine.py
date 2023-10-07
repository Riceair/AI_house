import torch

def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    train_losses = []

    for data, targets in data_loader:
        X = data.to(device)
        y = targets.to(device)

        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

    return sum(train_losses)/len(train_losses)

def valid_one_epoch(model, data_loader, criterion, device):
    model.eval()
    valid_losses = []

    for data, targets in data_loader:
        X = data.to(device)
        y = targets.to(device)

        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y)        
        valid_losses.append(loss.item())

    return sum(valid_losses)/len(valid_losses)

def test_one_epoch(model, data_loader, device):
    model.eval()
    prediction = []

    for data in data_loader:
        X = data.to(device)

        with torch.no_grad():
            outputs = model(X)
        
        outputs = outputs.cpu()
        prediction.append(outputs)

    prediction = torch.vstack(prediction)
    return prediction.numpy()