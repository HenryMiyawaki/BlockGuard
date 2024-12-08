import os
import networkx as nx
import random as rd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from neural_network import NeuralNetwork
from preprocessor import Preprocessor

class PeerNetwork:
    def __init__(self):
        self.labels = [
                "duration", "protocol_type", "service", "src_bytes", "dst_bytes", "flag", "count", "srv_count", "serror_rate",
                "same_srv_rate", "diff_srv_rate", "srv_serror_rate", "srv_diff_host_rate", "dst_host_count", 
                "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_serror_rate", "dst_host_srv_diff_host_rate", "dst_host_srv_serror_rate", "label"
            ]
        self.num_classes = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed_model, self.test_loader = self.initialize_seed_model()
        self.graph = self.create_graph()
        
        
    def initialize_seed_model(self):
        file_path = 'datasets/Worker_1_+_2/FL/test_w2.csv'
        
        preprocessor = Preprocessor(file_path)
        preprocessor.load_labels(labels=self.labels)
        
        X_train, X_test, y_train, y_test = preprocessor.process()
        self.num_classes = len(preprocessor.label_encoder.classes_)

        train_loader, test_loader = self._create_loaders(X_train, y_train, X_test, y_test)

        input_size = X_train.shape[1]
        seed_model = NeuralNetwork(input_size, self.num_classes, self.device)
        seed_model.train(train_loader, num_epochs=5)

        self.plot_metrics(seed_model, preprocessor, test_loader, y_test)

        return seed_model, test_loader

    def create_graph(self):
        graph = nx.Graph()
        paths = [
            "datasets/Merged_dataset/w_all.csv", 
            "datasets/Merged_dataset/w1.csv", 
            "datasets/Worker_1_+_2/DL/DL_2_train.csv"
        ]

        for i in range(5):
            dataset_path = paths[rd.randrange(0, len(paths))]

            preprocessor = Preprocessor(dataset_path)
            preprocessor.load_labels(labels=self.labels)
            X_train, X_test, y_train, y_test = preprocessor.process()
            train_loader, test_loader = self._create_loaders(X_train, y_train, X_test, y_test)

            graph.add_node(
                i,
                model=self.seed_model,
                train_loader=train_loader,
                local_test=test_loader,
                seed_test_loader=self.test_loader
            )


        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        graph.add_edges_from(edges)

        return graph
    
    def _create_loaders(self, X_train, y_train, X_test, y_test):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader
        
    def distill_knowledge(self, num_rounds=5, temperature=2.0, alpha=0.5):
        nodes = list(self.graph.nodes())

        for round_num in range(num_rounds): 
            print(f"\n--- Knowledge Distillation Round {round_num + 1} ---")
            
            for i in range(len(nodes)):
                student_node = nodes[i] 
                teacher_node = nodes[(i + 1) % len(nodes)] 

                student_model = self.graph.nodes[student_node]['model']
                teacher_model = self.graph.nodes[teacher_node]['model']

                student_accuracy = (student_model.evaluate(self.graph.nodes[student_node]['local_test']) + student_model.evaluate(self.test_loader))/2
                teacher_accuracy = (teacher_model.evaluate(self.graph.nodes[teacher_node]['local_test']) + teacher_model.evaluate(self.test_loader))/2

                if student_accuracy > teacher_accuracy:
                    print("skipped training")
                    continue

                print(f"Evaluating student model BEFORE distillation (Node {student_node}):")
                accuracy = student_model.evaluate(self.graph.nodes[student_node]['local_test'])
                print(f"Accuracy is {accuracy:.2f}%")

                teacher_logits = []
                teacher_model.model.eval()
                with torch.no_grad():
                    for X_batch, _ in self.graph.nodes[student_node]['train_loader']:
                        X_batch = X_batch.to(self.device)
                        logits = teacher_model.model(X_batch)
                        teacher_logits.append(logits)
                teacher_logits = torch.cat(teacher_logits)

                student_model.model.train()

                print(f"Training student model via distillation (Node {student_node}):")
                student_model.train_with_distillation(
                    self.graph.nodes[student_node]['train_loader'],
                    teacher_logits,
                    num_epochs=5,
                    temperature=temperature,
                    alpha=alpha
                )

                print(f"Evaluating student model AFTER distillation (Node {student_node}):")
                accuracy = student_model.evaluate(self.graph.nodes[student_node]['local_test'])
                print(f"Accuracy is {accuracy:.2f}%")
        
    def plot_metrics(self, seed_model, preprocessor, test_loader, y_test):
        seed_model.plot_training_loss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        y_pred = []
        y_scores = []
        seed_model.model.eval()
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = seed_model.model(X_batch)
                y_scores.extend(outputs.cpu().numpy())
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())

        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        plt_cf = seed_model.plot_confusion_matrix(y_test, y_pred, preprocessor.label_encoder)

        self.save_plot(plt_cf, "confusion_matrix")

        plt_roc = seed_model.plot_roc_curve(y_test, y_scores, self.num_classes, classes=preprocessor.label_encoder.classes_)

        self.save_plot(plt_roc, "roc_curve")

        plt_class = seed_model.plot_class_distribution(y_test, y_pred, preprocessor.label_encoder)

        self.save_plot(plt_class, "class_distribution")
    
    def save_plot(self, plt, name):
        save_path = "results"
    
        os.makedirs(save_path, exist_ok=True)
        
        file_path = os.path.join(save_path, name + '.png')
        
        plt.savefig(file_path)

        plt.close()