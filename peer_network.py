import os
import networkx as nx
import random as rd
import numpy as np
import torch
from neural_network import NeuralNetwork
from utils.metrics import Metrics
from utils.preprocessor import Preprocessor

class PeerNetwork:
    def __init__(self):
        
        self.num_classes = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed_model, self.test_loader, self.seed_accuracy = self.initialize_seed_model()
        self.graph = self.create_graph()
        
        
    def initialize_seed_model(self):
        file_path = 'datasets/Worker_1_+_2/FL/test_w2.csv'
        
        preprocessor = Preprocessor(file_path)
        preprocessor.load_labels()
        
        X_train, X_test, y_train, y_test = preprocessor.process()
        self.num_classes = len(preprocessor.label_encoder.classes_)

        train_loader, test_loader = preprocessor.create_loaders(X_train, y_train, X_test, y_test)

        input_size = X_train.shape[1]
        seed_model = NeuralNetwork(input_size, self.num_classes, self.device)
        seed_model.train(train_loader, num_epochs=5)
        self.seed_preprocessor = preprocessor

        self.plot_metrics(seed_model, test_loader, y_test)

        seed_accuracy = seed_model.evaluate(train_loader)
        print(f"Accuracy of models at nodes (have the same model) for seed test: {seed_accuracy:.2f}% \n -----------------------------------\n")
        print("\n--- Plotting for each node ---")

        return seed_model, test_loader, seed_accuracy

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
            preprocessor.load_labels()
            X_train, X_test, y_train, y_test = preprocessor.process()
            train_loader, test_loader = preprocessor.create_loaders(X_train, y_train, X_test, y_test)

            graph.add_node(
                i,
                model=self.seed_model,
                train_loader=train_loader,
                local_test=test_loader,
                seed_test_loader=self.test_loader,
                y_test_local=y_test,
                local_accuracy=self.seed_accuracy
            )


        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        graph.add_edges_from(edges)

        return graph
        
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
                    print(f"\n---Skipped training for Node {student_node}---\n")
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
        
    def plot_metrics(self, model, test_loader, y_test, path="results"):
        metrics = Metrics(model)
        metrics.plot_training_loss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        y_pred = []
        y_scores = []
        model.model.eval()
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model.model(X_batch)
                y_scores.extend(outputs.cpu().numpy())
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())

        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        plt_cf = metrics.plot_confusion_matrix(y_test, y_pred, self.seed_preprocessor.label_encoder)

        self.seed_preprocessor.save_plot(plt_cf, path, "confusion_matrix")

        plt_roc = metrics.plot_roc_curve(y_test, y_scores, self.num_classes, classes=self.seed_preprocessor.label_encoder.classes_)

        self.seed_preprocessor.save_plot(plt_roc, path, "roc_curve")

        plt_class = metrics.plot_class_distribution(y_test, y_pred, self.seed_preprocessor.label_encoder)

        self.seed_preprocessor.save_plot(plt_class, path, "class_distribution")

    def plot_graph_metrics(self, metric_key=None, title=None):
        metrics = Metrics(graph=self.graph)

        if metric_key == None and title == None:
            plt_graph = metrics.plot_graph_evolution()
        else:
            plt_graph = metrics.plot_graph_evolution(metric_key=metric_key or None, title=title or None)

        self.seed_preprocessor.save_plot(plt_graph, save_path="results/graph", name="class_distribution")
