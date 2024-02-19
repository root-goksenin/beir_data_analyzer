from comparators import CorpusComparator, QRelsComparator
from utils import get_lotte, get_beir, plot_similarity_matrix, plot_normal_heatmap
from enum import Enum
import click 
from typing import Dict, List
import os 
from functools import lru_cache
import pandas as pd 


class Tasks(Enum):
    QUERY_DISTRUBITON = "query_type_distribution"
    QUERY_VOCAB_OVERLAP = "query_overlap"
    QUERY_LEXICAL_OVERLAP = "query_answer_lexical_overlap"
    CORPUS_VOCAB_OVERLAP = "corpus_vocab_overlap"
    def get_tool(self, data_loaders):
        return CorpusComparator(data_loaders) if "corpus" in self.value else QRelsComparator(self, data_loaders)
    def get_title(self):
        return " ".join(self.value.split("_")).capitalize()
    
class Analyze():
    def __init__(self, task, data_loaders):
        self.tool = task.get_tool(data_loaders)
    def run(self, output_file):
        return self.tool.run(output_file)      

class DataGettr:
    
    @property
    @lru_cache()
    def beir(self):
        return get_beir()
    
    @property
    @lru_cache()
    def lotte_dev(self):
        return get_lotte("dev")
    
    @property
    @lru_cache()   
    def lotte_test(self):
        return get_lotte("test")

def get_column_names(out: Dict, data_name: str) -> List[str]:
    names = out.keys()
    data_names = [os.path.split(name)[1] for name in names]
    if "lotte" in data_name:
        return [" ".join(data.split("_")[:-1]).capitalize() if "msmarco" not in data else "msmarco" for data in data_names]
    else:
        return data_names

def get_title(task: Tasks, data_name: str) -> str:
    title = task.get_title()
    data_name = " ".join(data_name.split("_")).capitalize()
    return f"{title} for {data_name}"
               
def get_save(task: Tasks, data_name: str, extension: str) -> str:
    return f"{task.value}_{data_name}.{extension}"    

      
@click.command()
@click.option('--data_name', type=click.Choice(["lotte_dev", "lotte_test", "beir"]), help='The name of the data')
@click.option("--task", type=click.Choice(["query_type_distribution", "query_overlap", "query_answer_lexical_overlap","corpus_vocab_overlap"]), help="Task for zero-shot analyze")
def main(data_name, task):
    '''
    When the task is query_answer_lexical_overlap, create a new folder, and put them into the folder.
    '''
    task = Tasks(task)
    # Get the data_name and split from the data_getter
    data_loaders = getattr(DataGettr(), data_name)
    # Runs the analysis on the task and data_names, then returns a dictionary file. Thiis is also saved.
    analyze = Analyze(task, data_loaders)
    out = analyze.run(output_file= get_save(task, data_name, "json"))
    if task in [Tasks.CORPUS_VOCAB_OVERLAP, Tasks.QUERY_VOCAB_OVERLAP]:
        plot_similarity_matrix(out, get_title(task, data_name), get_save(task, data_name, "png"), get_column_names(out, data_name), out.keys())
    
if __name__ == "__main__":
    main()