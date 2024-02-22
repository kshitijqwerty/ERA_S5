from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchsummary import summary


class ModelHelper:
    def __init__(self, model, device, train_loader, test_loader):
        # Data to plot accuracy and loss graphs
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        if device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def GetCorrectPredCount(self, pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

    def train(self, optimizer, scheduler, criterion, num_epochs):
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}")
            self._train_step(optimizer, criterion)
            self._test_step(criterion)
            scheduler.step()

    def _train_step(self, optimizer, criterion):
        self.model.train()  # train mode set
        pbar = tqdm(self.train_loader)  # wrap loader iterater in tqdm

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):  # enumerate the pbar iterator
            data, target = data.to(self.device), target.to(
                self.device
            )  # transfer each value to cuda device
            optimizer.zero_grad()  # make the gradiants zero before optimizing

            # Predict
            pred = self.model(data)  # pred

            # Calculate loss
            loss = criterion(pred, target)  # calculate loss
            train_loss += loss.item()  # add to total loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            correct += self.GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f"Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
            )

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(self.train_loader))

    def _test_step(self, criterion):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += criterion(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss

                correct += self.GetCorrectPredCount(output, target)

        test_loss /= len(self.test_loader.dataset)
        self.test_acc.append(100.0 * correct / len(self.test_loader.dataset))
        self.test_losses.append(test_loss)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(self.test_loader.dataset),
                100.0 * correct / len(self.test_loader.dataset),
            )
        )

    def plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")

    def get_summary(self, input_size):
        temp_model = self.model.to('cpu')
        summary(temp_model, input_size=input_size)
