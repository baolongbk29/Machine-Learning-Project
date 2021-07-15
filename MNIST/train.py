import torch
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from model import Model
from torch import nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#Load Dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

mnist_valset, mnist_testset = torch.utils.data.random_split(mnist_testset, [int(0.9 * len(mnist_testset)), int(0.1 * len(mnist_testset))])

train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(mnist_valset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=False)

print("Training dataset size: ", len(mnist_trainset))
print("Validation dataset size: ", len(mnist_valset))
print("Testing dataset size: ", len(mnist_testset))


# visualize data
fig=plt.figure(figsize=(20, 10))
for i in range(1, 6):
    img = transforms.ToPILImage(mode='L')(mnist_trainset[i][0])
    fig.add_subplot(1, 6, i)
    plt.title(mnist_trainset[i][1])
    plt.imshow(img)
plt.show()



model = Model()
model.to(device)
critetion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr =0.001)

if (torch.cuda.is_available()):
    model.cuda()

epochs = 100
train_loss = list()
val_loss = list()
best_val_loss = 0.99
for epoch in range(epochs):
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    #Trainning
    for iter, (image, label) in enumerate(train_dataloader):
        if (torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()

        optimizer.zero_grad()

        pred = model(image)

        loss = critetion(pred,label)
        total_train_loss +=loss.item()

        loss.backward()
        optimizer.step()

    total_train_loss = total_train_loss/(iter+1)
    train_loss.append(total_train_loss)


    #Validation
    model.eval()
    total = 0
    for iter, (image,label) in enumerate(val_dataloader):
        if (torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()

        pred = model(image)


        loss = critetion(pred,label)
        total_val_loss +=loss.item()


        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1

    accuracy = total / len(mnist_valset)

    total_val_loss = total_val_loss / (iter + 1)
    val_loss.append(total_val_loss)

    print('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch + 1, epochs,
                                                                                                  total_train_loss,
                                                                                                  total_val_loss,
                                                                                                  accuracy))

    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        print("Saving the model state dictionary for Epoch: {} with Validation loss: {:.8f}".format(epoch + 1,
                                                                                                        total_val_loss))
        torch.save(model.state_dict(), "model.dth")

fig=plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, epochs+1), train_loss, label="Train loss")
plt.plot(np.arange(1, epochs+1), val_loss, label="Validation loss")
plt.xlabel('Loss')
plt.ylabel('Epochs')
plt.title("Loss Plots")
plt.legend(loc='upper right')
plt.show()