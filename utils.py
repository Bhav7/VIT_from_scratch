import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_generator(input_tensor, size, viz_mode = False):
    patchified = []
    for row_i in range(0, input_tensor.shape[1], size):
        for col_i in range(0, input_tensor.shape[1], size):
            if viz_mode == True:
                patch = input_tensor[:, row_i:row_i+size, col_i:col_i+size].tolist()
            else:
                patch = torch.flatten(input_tensor[:, row_i:row_i+size, col_i:col_i+size]).tolist()
            patchified.append(patch)
    return patchified

def compute_accuracy_and_loss(model, features, labels, bs, device):
    total_accuracy = 0
    total_loss = []
    for batch_idx in range(0, len(features), bs):
        x_batch = features[batch_idx: batch_idx + bs].to(device)
        y_batch = labels[batch_idx: batch_idx + bs].to(device)
        scores = model(x_batch)
        loss = F.cross_entropy(scores, y_batch)
        preds = torch.argmax(scores, axis = 1)
        total_accuracy += torch.sum(preds == y_batch)
        total_loss.append(loss.item())
    return (total_accuracy/len(labels), torch.mean(torch.tensor(total_loss)))