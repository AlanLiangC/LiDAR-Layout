from pyvis.network import Network
import networkx as nx

def generate_graph(dataset, graph_dict):

    G = nx.DiGraph()
    box_names = [dataset.classes_r[idx] for idx in graph_dict['decoder']['objs'].numpy()]
    for i in range(len(box_names)):
        G.add_node(i, label=box_names[i])
    triples = graph_dict['decoder']['triples']
    for i in range(len(triples)):
        r = triples[i]
        G.add_edge(int(r[0]), int(r[2]), title=dataset.relationships[r[1]])

    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.from_nx(G)

    net.show_buttons(filter_=['physics'])
    net.show("dynamic_graph.html",notebook=False)
