from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from neural_network import NeuralNetwork
from peer_network import PeerNetwork
from utils.preprocessor import Preprocessor

def main():
    print("Starting Peer Network tests...\n")
    network = PeerNetwork()

    print("\n--- Test 1: Training and Evaluation of Seed Model ---")
    try:
        seed_accuracy = network.seed_model.evaluate(network.test_loader)
        print(f"Seed Model accuracy on the test set: {seed_accuracy:.2f}% \n -----------------------------------")
    except Exception as e:
        print(f"Error during Seed Model test: {e}")

    print("\n--- Test 2: Graph Creation and Local Model Training ---")
    try:
        print(f"Number of nodes in the graph: {len(network.graph.nodes())}")
        print(f"Number of edges in the graph: {len(network.graph.edges())}")

        for node in network.graph.nodes(data=True):
            local_model = node[1]['model']

            network.plot_metrics(local_model, node[1]['local_test'], node[1]['y_test_local'], path="results/node_before/plot_node" + str(node[0]))
            local_accuracy = local_model.evaluate(node[1]['local_test'])
            print(f"Accuracy of model at node {node[0]} for local test (local datasets): {local_accuracy:.2f}%")
                    
    except Exception as e:
        print(f"Error during graph test and local models: {e}")

    print("\n--- Test 3: Knowledge Distillation Round ---")
    try:
        network.distill_knowledge(num_rounds=5, temperature=3.0, alpha=0.3)
        for node in network.graph.nodes(data=True):
            local_accuracy = node[1]['model'].evaluate(node[1]['local_test'])
            print(f"Accuracy of model at node {node[0]} after distillation: {local_accuracy:.2f}%")
    except Exception as e:
        print(f"Error during knowledge distillation: {e}")

    print("\n--- Tests Completed ---")

    for node in network.graph.nodes(data=True):
        local_model = node[1]['model']

        local_accuracy = local_model.evaluate(node[1]['local_test'])
        print(f"Final accuracy for {node[0]} in local test dataset: {local_accuracy:.2f}%")
        
if __name__ == '__main__':
    main()
