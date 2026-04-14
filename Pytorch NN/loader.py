
import torch
from pySmartDL.download import download
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch import nn
import matplotlib.pyplot as plt
import torchvision.models as models

traind = datasets.FashionMNIST(root='data',train=True, download=True,transform=ToTensor())
testd = datasets.FashionMNIST(root='data',train=False,transform=ToTensor(),download=True)
train_dl = DataLoader(traind, batch_size=64, shuffle= True)
test_dl = DataLoader(testd, batch_size=64, shuffle=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)
learning_rate = 1e-3
batch_size = 64

model = models.vgg16()
model.load_state_dict(torch.load('modelweight.pth', weights_only=True))



def test_loop(dataloader,model,loss_fn):
    model.eval() # evaluation mode, no dropping neurons
    size = len(dataloader.dataset)
    num_batches= len(dataloader.dataset)
    test_loss, correct = 0,0
    with torch.no_grad():#no learning or backpropogation so no gradients needed
        for x,y in dataloader:
            pred=model(x)
            test_loss += loss_fn(pred,y).item() # calculate total    loss/cost
            correct+= (pred.argmax(1)== y).type(torch.float).sum().item()# smart way of counting the amount of correct predidctions
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss() # loss functin is the mean of the square differences
optimiser =torch.optim.SGD(model.parameters(),lr =learning_rate) #use stoic GD to minimize loss
epochs =100 #loop through data set 10 times
for test in range(epochs):
    print(f"Epoch {test + 1}\n-------------------------------")
    test_loop(test_dl, model, loss_fn)
