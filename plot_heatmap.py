
import json
import click 
from utils import plot_similarity_matrix
from typing import List, Dict
from enum import Enum
import os 


class Tasks(Enum):
    QUERY_DISTRUBITON = "query_type_distribution"
    QUERY_VOCAB_OVERLAP = "query_overlap"
    QUERY_LEXICAL_OVERLAP = "query_answer_lexical_overlap"
    CORPUS_VOCAB_OVERLAP = "corpus_vocab_overlap"
    def get_title(self):
        return " ".join(self.value.split("_")).capitalize()
def get_column_names(out: Dict, data_name: str) -> List[str]:
    names = out.keys()
    data_names = [os.path.split(name)[1] for name in names]
    if "lotte" in data_name:
        return [" ".join(data.split("_")[:-1]).capitalize() for data in data_names]
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
    json_file = get_save(task, data_name, "json")
    with open(json_file, 'r') as f:
        out = json.load(f)
    print(get_column_names(out, data_name))
    if task in [Tasks.CORPUS_VOCAB_OVERLAP, Tasks.QUERY_VOCAB_OVERLAP]:
        plot_similarity_matrix(out, get_title(task, data_name), get_save(task, data_name, "png"), get_column_names(out, data_name), out.keys())
    
if __name__ == "__main__":
    main()