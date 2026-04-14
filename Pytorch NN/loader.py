
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
model.eval()

def train_loop(dataloader, model, loss_fn,optimiser):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x,y) in enumerate(dataloader):  # where X is the image tensor, and y is the ACTUAL Class


        predic = model(x)
        loss = loss_fn(predic,y) # calculates loss of NN, using the prediction and the real valyes

        # backpropogation algorithim - are we isacc chat?
        loss.backward() # computes the gradients of loss backpropogation
        optimiser.step() # update the NN Parametres using the gradients
        optimiser.zero_grad() # zero the gradients

        if batch %100==0:
            loss,current_data = loss.item(), batch* batch_size + len(x)
            img = dataloader.dataset[current_data][0]
            #        plt.imshow(img.permute(1,2,0))
            #        plt.axis('off')
            #        plt.show()
            print(f"loss: {loss:>7f}  [{current_data:>5d}/{size:>5d}]")

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
    train_loop(train_dl, model, loss_fn, optimiser)
    test_loop(test_dl, model, loss_fn)
