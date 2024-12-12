
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.calibration import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from neural_network import NeuralNetwork


class Metrics():
    def __init__(self, model=None, graph=None):
        if (model == None or not isinstance(model, NeuralNetwork)) and graph == None:
            print("Error: there is no neuralnetwork provided.")

        if model != None:
            self.train_losses = model.train_losses

        if graph != None:
            self.graph = graph
    
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

