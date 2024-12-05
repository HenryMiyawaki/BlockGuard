import networkx as nx
import random as rd
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
        self.neural_network_seed, self.test_loader = self.start()
        self.graph = self.create_graph()
        self.num_classes = 0
        
        
    def start(self):
        file_path = 'datasets/Worker_1_+_2/FL/test_w2.csv'
        
        preprocessor = Preprocessor(file_path)
        preprocessor.load_labels(labels=self.labels)
        
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
        self.num_classes = len(preprocessor.label_encoder.classes_)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        network = NeuralNetwork(input_size, self.num_classes, device)

        network.train(train_loader, num_epochs=5)
        
        return network, test_loader

    def create_graph(self):
        grap = nx.Graph()
        
        paths = ["datasets\Merged_dataset\w_all.csv", "datasets\Merged_dataset\w_all.csv", "datasets\Merged_dataset\w_all.csv", "datasets\Merged_dataset\w1.csv", "datasets\Worker_1_+_2\DL\DL_2_train.csv", "datasets\Worker_1_+_2\FL\train_w2.csv"]
        
        for i in range(5):
            random_path = paths[rd.randrange(0, 3)]
            
            preprocessor = Preprocessor(random_path)
            preprocessor.load_labels(labels=self.labels)
            X_train, X_test, y_train, y_test = preprocessor.process()
            
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            unique_model = self.neural_network_seed
            
            grap.add_node(i, 
                    local_data=train_loader,
                    model=unique_model,
                    train=train_loader,
                    train_x=X_train,
                    tester=test_loader)

            edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
            grap.add_edges_from(edges)
        
        nodes = list(grap.nodes())
        for i in range(len(nodes)):
            node_data = grap.nodes[nodes[i]]
            input_size = node_data['train_x'].shape[1] 
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            network = NeuralNetwork(input_size, self.num_classes, device)
            network.train(node_data['train'], num_epochs=5)
            
            grap.nodes[nodes[i]]['model'] = network
            
            network.evaluate(self.test_loader)
            network.evaluate(grap.nodes[nodes[i]]['tester'])
            
            grap.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])
            
        print("Network graph with peers and connections:")
        print(grap.nodes(data=True))

if __name__ == '__main__':
    PeerNetwork()