import networkx as nx
import random as rd

class PeerNetwork:
    def __init__(self):
        print("init")
        self.graph = self.create_graph()

    def create_graph(self):
        grap = nx.Graph()

        for i in range(5):
            grap.add_node(i, 
                    local_data=None,
                    model=None)

        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        grap.add_edges_from(edges)

        print("Network graph with peers and connections:")
        print(grap.nodes(data=True))

if __name__ == '__main__':
    PeerNetwork()