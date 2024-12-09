from matplotlib import pyplot as plt
import pandas as pd
from sklearn.calibration import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNetwork:
    def __init__(self, input_size, num_classes, device):
        self.device = device
        self.model = self._build_model(input_size, num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)
        self.train_losses = []

    def _build_model(self, input_size, num_classes):
        class NeuralNet(nn.Module):
            def __init__(self, input_size, num_classes):
                super(NeuralNet, self).__init__()
                self.fc1 = nn.Linear(input_size, 128)
                self.bn1 = nn.BatchNorm1d(128)
                self.fc2 = nn.Linear(128, 64)
                self.bn2 = nn.BatchNorm1d(64)
                self.fc3 = nn.Linear(64, 32)
                self.bn3 = nn.BatchNorm1d(32)
                self.fc4 = nn.Linear(32, num_classes)
                self.dropout = nn.Dropout(0.4)

            def forward(self, x):
                x = F.relu(self.bn1(self.fc1(x)))
                x = self.dropout(x)
                x = F.relu(self.bn2(self.fc2(x))) 
                x = self.dropout(x)
                x = F.relu(self.bn3(self.fc3(x))) 
                x = self.fc4(x)
                return x
        return NeuralNet(input_size, num_classes)
    
    def distillation_loss(self, student_logits, teacher_logits, labels, temperature, alpha):
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
        soft_student = F.log_softmax(student_logits / temperature, dim=1)

        distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

        ce_loss = self.criterion(student_logits, labels)

        return alpha * distillation_loss + (1 - alpha) * ce_loss

    def train_with_distillation(self, train_loader, teacher_logits, num_epochs=5, temperature=3.0, alpha=0.5):
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                start_idx = i * len(y_batch)
                end_idx = (i + 1) * len(y_batch)
                batch_teacher_logits = teacher_logits[start_idx:end_idx].to(self.device)

                student_logits = self.model(X_batch)

                loss = self.distillation_loss(student_logits, batch_teacher_logits, y_batch, temperature, alpha)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            self.scheduler.step(avg_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def train(self, train_loader, num_epochs=5):
        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            self.train_losses.append(avg_loss)

            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break

            self.scheduler.step(avg_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        accuracy = 100 * correct / total
        
        return accuracy

    def get_teacher_logits(self, data_loader):
        self.model.eval()
        logits_list = []
        total_samples = 0
        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch)
                logits_list.append(logits)
                total_samples += X_batch.size(0)
        return torch.cat(logits_list), total_samples
    
    def plot_training_loss(self):
        df_metrics = pd.DataFrame({'Epoch': range(1, len(self.train_losses) + 1), 'Loss': self.train_losses})
        df_metrics.plot(x='Epoch', y='Loss', kind='line', title='Training Loss', legend=True)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        return plt


    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, label_encoder):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        return plt

    
    @staticmethod
    def plot_roc_curve(y_true, y_scores, num_classes, classes=[]):
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        if(len(classes) != num_classes):
            classes = []
        plt.figure(figsize=(10, 6))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            if (len(classes) > 0):
                plt.plot(fpr, tpr, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')
            else:
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        return plt


    @staticmethod
    def plot_class_distribution(y_true, y_pred, label_encoder):
        results = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
        results['Correct'] = results['True'] == results['Predicted']

        summary = results.groupby('True')['Correct'].value_counts().unstack(fill_value=0)
        summary.columns = ['Incorrect', 'Correct']

        summary.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Prediction Accuracy by Class')
        plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
        plt.legend(title='Prediction')
        return plt
