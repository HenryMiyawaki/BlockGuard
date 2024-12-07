from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from neural_network import NeuralNetwork
from preprocessor import Preprocessor

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

    network.plot_roc_curve(y_test, y_scores, num_classes, classes=preprocessor.label_encoder.classes_)

    network.plot_class_distribution(y_test, y_pred, preprocessor.label_encoder)


if __name__ == '__main__':
    main()