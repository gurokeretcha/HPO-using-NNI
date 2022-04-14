import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def train(model, device, train_loader,loss_fn, optimizer):
    size = len(train_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(model, device, loss_fn, test_loader):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

def main(args):
    training_data = datasets.FashionMNIST(root="data",train=True,download=True,transform=ToTensor(),)
    test_data = datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor(),)

    train_dataloader = DataLoader(training_data, batch_size=args['batch_size'])
    test_dataloader = DataLoader(test_data, batch_size=64)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self,hidden_size1,hidden_size2 ):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, hidden_size1),
                nn.ReLU(),
                nn.Linear(hidden_size1, hidden_size2),
                nn.ReLU(),
                nn.Linear(hidden_size2, 10)
            )
        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork(hidden_size1=args['hidden_size1'],hidden_size2=args['hidden_size2'] ).to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'])

    for epoch in range(10):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(model, device, train_dataloader, loss_fn,optimizer)
        test_acc = test(model, device, loss_fn, test_dataloader)
        print(test_acc)
    print('final accuracy:', test_acc)

if __name__ == '__main__':
    params = {
    "batch_size": 32,
    "hidden_size1": 128,
    "hidden_size2": 256,
    "lr": 0.04585081360744873,
    "momentum": 0.5363521578821588
}
    main(params)
