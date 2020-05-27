from matplotlib import pyplot as plt
from src.models.learn_rate import find_lr

def visualize_train_results(results):
    epochs = int(results.iloc[-1]["epoch"])
    iterations = int(results.iloc[-1]["iteration"])
    tr, tco, tse, tsp = [], [], [], []

    print(results.to_string)

    for epoch in range(epochs):
        fn = results.iloc[epoch * iterations:(epoch + 1) * iterations]["fn"].sum()
        fp = results.iloc[epoch * iterations:(epoch + 1) * iterations]["fp"].sum()
        tn = results.iloc[epoch * iterations:(epoch + 1) * iterations]["tn"].sum()
        tp = results.iloc[epoch * iterations:(epoch + 1) * iterations]["tp"].sum()
        tr.append(results.iloc[((epoch + 1) * iterations)-1]["loss"])
        tco.append((tn + tp) / (tp + tn + fp + fn))
        tse.append(tp / (tp + fn))
        tsp.append(tn / (tn + fp))

    plt.plot(tco, label="Correctness")
    plt.plot(tse, label="Sensitivity")
    plt.plot(tsp, label="Specificity")
    plt.legend()
    # plt.show()

    plt.plot(tr, label="Loss")
    plt.legend()
    plt.show()


def find_learning_rate():
    log, losses = find_lr(model, F.binary_cross_entropy, optimizer, train_loader, init_value=1e-8, final_value=10e-4, device="cpu")
    total_correctness, total_sensitivity, total_specificity, total_loss = train(train_loader, 1, optimizer_1, epilepsy_model_1)


def profile_per_file():
    pass