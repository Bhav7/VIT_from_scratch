import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from tqdm import tqdm
import numpy as np

from models import *
from utils import *


mps_device = torch.device("mps")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])


trainset = torchvision.datasets.FashionMNIST(root='./fashion_mnist', train=True,
                                        download=True, transform=transform)


testset = torchvision.datasets.FashionMNIST(root='./fashion_mnist', train=False,
                                        download=True, transform=transform)


num_classes = len(trainset.classes)

def trainer(model, epochs, trainset, testset):

    patchified_set = []
    labels = []

    for i in tqdm(trainset):
        patches = patch_generator(i[0], 14)
        labels.append(i[1])
        patchified_set.append(patches)

    patchified_set = torch.tensor(patchified_set)
    labels = torch.tensor(labels)

    total_instances = len(patchified_set)
    patchified_trainset = patchified_set[:int(total_instances*0.7)]
    patchified_valset = patchified_set[int(total_instances*0.7):]

    train_labels = labels[:int(total_instances*0.7)]
    val_labels = labels[int(total_instances*0.7):]

    epochs = epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3) 
    bs = 512
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(0, epochs):
        model.train()
        epoch_perm = torch.randperm(len(patchified_trainset))
        patchified_trainset = patchified_trainset[epoch_perm]
        train_labels = train_labels[epoch_perm]
        train_loss = []
        train_average = []
        for batch_idx in tqdm(range(0, len(patchified_trainset), bs)):
            x_batch = patchified_trainset[batch_idx: batch_idx + bs].to(mps_device)
            y_batch = train_labels[batch_idx: batch_idx + bs].to(mps_device)


            optimizer.zero_grad()
            scores = model(x_batch)
            loss = loss_fn(scores, y_batch)
            
            loss.backward()
            optimizer.step()
            

            with torch.no_grad():
                preds = torch.argmax(scores, axis = 1)
                train_average.append(torch.sum(preds == y_batch).item())
                train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_acc, val_loss = compute_accuracy_and_loss(model, patchified_valset, val_labels, bs, mps_device)
            print(f"Epoch: {epoch}, Train Loss: {np.mean(train_loss)}, Train Acc: {np.sum(train_average)/len(patchified_trainset)}, Val Loss: {val_loss}, Val Acc: {val_acc}")

    patchified_testset = []
    test_labels = []

    for i in tqdm(testset):
        patches = patch_generator(i[0], 14)
        test_labels.append(i[1])
        patchified_testset.append(patches)

    patchified_testset = torch.tensor(patchified_testset)
    test_labels = torch.tensor(test_labels)

    test_acc, test_loss = compute_accuracy_and_loss(model, patchified_testset, test_labels, bs, mps_device)
    return test_acc



my_vit = VisionTransformer(4, 196, 64, num_classes, 6, 4)
my_vit.to(mps_device)
my_vit_test_acc = trainer(my_vit, 100, trainset, testset)

print(f"Performance of my_vit is : {my_vit_test_acc}")

other_vit = Other_VisionTransformer(64, 64*4, 4, 6, num_classes, 4)
other_vit.to(mps_device)

other_vit_test_acc = trainer(other_vit, 100, trainset, testset)

print(f"Performance of other_vit is : {other_vit_test_acc}")