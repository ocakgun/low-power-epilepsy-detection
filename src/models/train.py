import torch
import pandas as pd


def train(net, train_loader, epochs, criterion, optimizer, print_every=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.train()
    columns = ["epoch", "iteration", "size", "loss", "fn", "fp", "tn", "tp"]
    results = []
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            running_loss = 0
            seizures, labels, filenames = batch[0].to(device), batch[1].to(device), batch[2]
            optimizer.zero_grad()
            outputs = net(seizures)
            loss = criterion(outputs.view(-1), labels.double())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            fn, fp, tn, tp = profile_results(torch.round(outputs), labels)
            results.append([epoch + 1, i + 1, len(seizures), running_loss, fn, fp, tn, tp])
            print_last_results(epoch + 1, i + 1, running_loss, fn, fp, tn, tp) if (i + 1) % print_every == 0 else False
    net.eval()
    return pd.DataFrame(results, columns=columns)


def print_last_results(epoch, iter, loss, fn, fp, tn, tp):
    correctness, sensitivity, specificity = profile_to_measure(tn, tp, fp, fn)

    print("epoch: {:d}, iter {:d}, loss {:.4f}, Correct: {:.2f}%, Sensitivty {:.2f}%, Specificity {:.2f}%".format(
    epoch, iter, loss, correctness*100, sensitivity*100, specificity*100))


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

    return fn, fp, tn, tp


def profile_to_measure(tn, tp, fp, fn):
    correctness = (tn + tp) / (tp + tn + fp + fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    return correctness, sensitivity, specificity
