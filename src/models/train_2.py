import torch
import torch.nn.functional as F


def train(train_loader, valid_loader, epochs, optimizer, model, print_every=1, device="cpu"):
    traindata_len = 356
    bs = 89
    model.train()
    total_correctness, total_sensitivity, total_specificity, _total_loss = [], [], [], []
    for epoch in range(epochs):
        total_loss = 0
        epoch_correctness, epoch_sensitivity, epoch_specificity, epoch_loss = [], [], [], []
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            print(outputs.type())
            print(labels.type())
            loss = F.binary_cross_entropy(outputs.view(-1), labels.double())
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            target_rounded = torch.round(outputs)
            correctness, sensitivity, specificity = profile_results(target_rounded, labels)
            epoch_correctness.append(correctness), epoch_sensitivity.append(sensitivity), epoch_specificity.append(specificity), epoch_loss.append(total_loss)

            if (i + 1) % print_every == 0:
                print("epoch: {:d}, iter {:d}/{:d}, loss {:.4f}, Correct: {:.2f}%, Sensitivty {:.2f}%, Specificity {:.2f}%".format(
                epoch + 1, i + 1, traindata_len // bs, total_loss / print_every, correctness*100, sensitivity*100, specificity*100))
                total_loss = 0

        total_correctness.append(epoch_correctness), total_sensitivity.append(epoch_sensitivity), total_specificity.append(epoch_specificity), _total_loss.append(epoch_loss)

    model.eval()
    return total_correctness, total_sensitivity, total_specificity, _total_loss


def profile_results(outputs, targets):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(outputs)):
        output = outputs[i][0]
        target = targets[i]

        if output and target:
            tp = tp+1
        elif not output and not target:
            tn = tn+1
        elif not output and target:
            fn = fn+1
        elif output and not target:
            fp = fp + 1
        else:
            print("Could not identify")

    total_correctnes = (tp + tn)/(fp + fn + tp + tn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    return total_correctnes, sensitivity, specificity
