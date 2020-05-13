import torch
import torch.nn.functional as F

def train(data_loader, epochs, optimizer, model, print_every=1, device="cpu"):
    traindata_len = 356
    bs = 48
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        epoch_loss = []
        epoch_correct = []
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = F.binary_cross_entropy(outputs.view(-1), labels.double())
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            # print(torch.eq(torch.round(outputs).view(-1), labels))
            correct = torch.sum(torch.eq(torch.round(outputs).view(-1), labels) == True)


            if (i + 1) % print_every == 0:
                print("epoch: {:d}, iter {:d}/{:d}, loss {:.4f}, Correct: {:.2f}%".format(
                epoch + 1, i + 1, traindata_len // bs + 1, total_loss / print_every, correct.item()/len(inputs)*100))
                total_loss = 0

            epoch_loss.append(total_loss)
            epoch_correct.append(correct.item())

        print(sum(epoch_correct))
        print("Correct: {:.2f}%".format(sum(epoch_correct)/traindata_len*100))
