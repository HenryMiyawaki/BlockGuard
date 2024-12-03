import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np


class Preprocessor:
    def __init__(self, file_path, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_labels(self, path=None, labels=[]):
        if labels:
            self.labels = labels
        elif path:
            try:
                self.labels = pd.read_csv(path).iloc[:, 0].tolist()
            except Exception as e:
                raise ValueError(f"Cannot load label in {path}: {e}")
        else:
            raise ValueError(f"No label provided. Please, prove an array or an path for a CSV containing the labels")

    def process(self):
        if not self.labels:
            raise ValueError("Labels not loaded, please load labels with method 'load_labels' and providing an label array or a path for CSV file.")
        
        df = pd.read_csv(self.file_path)
        if len(df.columns) != len(self.labels):
            raise ValueError("The number of columns does not match with the provided labels")
        df.columns = self.labels
        
        df['label_encoded'] = self.label_encoder.fit_transform(df['label'])
        X = df.drop(columns=['label', 'label_encoded'])
        y = df['label_encoded']
        X = self.scaler.fit_transform(X.select_dtypes(include=['float64', 'int64']))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, 
                                                            stratify=y, random_state=self.random_state)
        return X_train, X_test, y_train, y_test
    
    def process_another_datasets(self, file_paths):
        if not self.labels:
            raise ValueError("The labels need to be proccesed before adding another datasets.")
        datasets = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            if len(df.columns) != len(self.labels):
                raise ValueError(f"The number of columns in the file {file_path} does not match with the provided labels.")
            df.columns = self.labels

            df['label_encoded'] = self.label_encoder.transform(df['label'])
            X = df.drop(columns=['label', 'label_encoded'])
            y = df['label_encoded']
            X = self.scaler.transform(X.select_dtypes(include=['float64', 'int64']))  # Usa o scaler ajustado
            datasets.append((torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)))
        
        return datasets


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

    def train(self, train_loader, num_epochs=20):
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
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def plot_training_loss(self):
        df_metrics = pd.DataFrame({'Epoch': range(1, len(self.train_losses) + 1), 'Loss': self.train_losses})
        df_metrics.plot(x='Epoch', y='Loss', kind='line', title='Training Loss', legend=True)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, label_encoder):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_scores, num_classes):
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        plt.figure(figsize=(10, 6))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

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
        plt.show()

def main():
    file_path = 'datasets/Merged_dataset/w_all.csv'
    preprocessor = Preprocessor(file_path)
    preprocessor.load_labels(labels=[
            "duration", "protocol_type", "service", "src_bytes", "dst_bytes", "flag", "count", "srv_count", "serror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_serror_rate", "srv_diff_host_rate", "dst_host_count", 
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_serror_rate", "dst_host_srv_diff_host_rate", "dst_host_srv_serror_rate", "label"
        ])
    X_train, X_test, y_train, y_test = preprocessor.process()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = X_train.shape[1]
    num_classes = len(preprocessor.label_encoder.classes_)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    network = NeuralNetwork(input_size, num_classes, device)

    network.train(train_loader, num_epochs=20)

    network.evaluate(test_loader)

    network.plot_training_loss()

    y_pred = []
    y_scores = []
    network.model.eval()
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = network.model(X_batch)
            y_scores.extend(outputs.cpu().numpy())
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())

    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    network.plot_confusion_matrix(y_test, y_pred, preprocessor.label_encoder)

    network.plot_roc_curve(y_test, y_scores, num_classes)

    network.plot_class_distribution(y_test, y_pred, preprocessor.label_encoder)


if __name__ == '__main__':
    main()