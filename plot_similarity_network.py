import networkx as nx
import matplotlib.pyplot as plt
import os
import json 
import click 
from enum import Enum 
import glob
import matplotlib.image as mpimg


beir_map = {'nfcorpus': 6, 'msmarco': 8, 'fiqa': 5, 'scidocs': 2, 'fever': 1, 'arguana': 4, 'scifact': 1, 'trec-covid': 6, 
            'climate-fever': 1, 'hotpotqa': 5, 'nq': 5, 'quora': 3, 'webis-touche2020': 4, 'dbpedia-entity': 7, 
            'physics': 3, 'stats': 3, 'webmasters': 3, 'wordpress': 3, 'programmers': 3, 'english': 3, 'mathematica': 3, 'gaming': 3, 'gis': 3, 'unix': 3, 'tex': 3, 'android': 3}
lotte_map = {'Lifestyle search': 1, 'Writing search':1, 'Lifestyle forum': 2, 'Pooled forum': 2, 'Writing forum': 2, 
             'Recreation forum': 2, 'Science search': 1, 'Pooled search': 1, 'Recreation search': 1, 'Technology search': 1, 
             'Science forum': 2, 'Technology forum': 2, 'msmarco': 3}
def get_node_colors(G, data):
    if "beir" in data: 
        return [beir_map.get(node, 0) for node in G.nodes()]

    else:
        return [lotte_map.get(node, 0) for node in G.nodes()]
      
    
class Tasks(Enum):
    QUERY_DISTRUBITON = "query_type_distribution"
    QUERY_VOCAB_OVERLAP = "query_overlap"
    QUERY_LEXICAL_OVERLAP = "query_answer_lexical_overlap"
    CORPUS_VOCAB_OVERLAP = "corpus_vocab_overlap"
    def get_title(self):
        return " ".join(self.value.split("_")).capitalize()
    

def get_column_names(out, data_name: str):
    names = out.keys()
    data_names = [os.path.split(name)[1] for name in names]
    if "lotte" in data_name:
        return [" ".join(data.split("_")[:-1]).capitalize() if "msmarco" not in data else "msmarco" for data in data_names]
    else:
        return data_names
    
def get_save(task: Tasks, data_name: str, extension: str) -> str:
    return f"{task.value}_{data_name}.{extension}"   
      
def get_title(task: Tasks, data_name: str) -> str:
    title = task.get_title()
    data_name = " ".join(data_name.split("_")).capitalize()
    return f"{title} for {data_name}"

def get_data_task_from_file(json_):
    _, file_name = os.path.split(json_)
    # Find data name
    data_names = ["lotte_dev", "lotte_test", "beir"]
    data_ = None
    for data in data_names:
        if data in json_:
            data_ = data 
    
    task_ = None 
    for task in Tasks: 
        if task.value in json_:
            task_ = task
    
    return data_, task_
@click.command()
@click.option("--json_file", type = str, help = "Json similarity data to create graph representation")
def create_spring_network(json_file):
    # Create a sample weighted graph (you can replace this with your own graph data)
    G = nx.Graph()
    with open(os.path.join(json_file), 'r') as writer:
        similarity_matrix = json.load(writer)
    data, task = get_data_task_from_file(json_file)
    row_column_names = get_column_names(similarity_matrix, data)
    edges = []
    for row, row_ in zip(similarity_matrix.keys(), row_column_names):
        for col, col_ in zip(similarity_matrix.keys(), row_column_names):
            
            if "lotte" in data:
                row_name, col_name = row_.split(" ")[0], col_.split(" ")[0]
                if row_name != col_name:
                    edge =  (row_, col_, {"weight" : round([dict_val[col] for dict_val in similarity_matrix[row] if col in dict_val][0],2)})
                    edges.append(edge)

            elif row != col:
                edge =  (row_, col_, {"weight" : round([dict_val[col] for dict_val in similarity_matrix[row] if col in dict_val][0],2)})
                edges.append(edge)

    G.add_edges_from(edges)

    plt.figure(figsize = (15,15))
    pos = nx.spring_layout(G, scale = None, dim=2, k = 0.01, seed = 42)
    nx.draw(G, pos, with_labels=True, node_size= 6000, font_weight='bold', font_color='black', 
            edge_color=[G[u][v]['weight'] for u, v in G.edges],
            edge_cmap= plt.cm.Greys,
            alpha=.8,
            edge_vmax= 1,
            edge_vmin = 0,
            node_color = get_node_colors(G,data))

    plt.title(get_title(task, data))
    plt.savefig("generated/png/network_plot_{}".format(get_save(task, data, ".png")))
if __name__ == '__main__':
    create_spring_network()