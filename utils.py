
import os 
from typing import List 
import json 
from nltk.tokenize import RegexpTokenizer
import tqdm 
import os 
import json
import seaborn as sns 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter 
from typing import Dict
import json
import os
from beir.datasets.data_loader import GenericDataLoader
from collections import namedtuple

vocab_tuple = namedtuple('Vocab', 'name vocab')


sns.set(font_scale=2)
memo = {}


new_beir_index =  [
    "Msmarco",
    "Arguana",
    "Climate-fever",
    "Dbpedia-entity",
    "Fever",
    "Fiqa",
    "Hotpotqa",
    "Nfcorpus",
    "Nq",
    "Quora",
    "Scidocs",
    "Scifact",
    "Trec-covid",
    "Webis-touche2020"
    ]
new_lotte_index =  [
    "Msmarco"
    "Lifestyle forum",
    "Writing forum",
    "Technology forum",
    "Recreation forum",
    "Science forum",
    "Pooled forum",
    "Lifestyle search",
    "Writing search",
    "Technology search",
    "Recreation search",
    "Science search",
    "Pooled search",
    ]
    

def json_to_pd(json_file: Dict):
    def my_function(value):
        return value
    df =  pd.DataFrame.from_dict(json_file).transpose()
    df.replace(np.nan, 0, inplace= True)
    df = df[["What", "When", "Who", "How", "Where", "Why", "Which", "Y/N", "Declerative"]]
    df = df.rename(columns = {"Declerative" : "Declarative"})
    return df.applymap(my_function)
    
def process_df_for_beir(df):
    names = list(map(lambda x: os.path.join("./beir_data", x),os.listdir("./beir_data")))
    names_ = {a : a.split("/")[-1].capitalize() for a in names}
    df = df.rename(index = names_) 
    df = df.reindex(new_beir_index)
    return df

def process_df_for_lotte(df):
    names = list(map(lambda x: os.path.join("./lotte_beir_format", x),os.listdir("./lotte_beir_format")))
    names_ = {a : " ".join(a.split("/")[-1].split("_")[:2]).capitalize() for a in names if "msmarco" not in a}
    names_["./beir_data/msmarco"] = "Msmarco"
    df = df.rename(index = names_) 
    df = df.reindex(new_lotte_index)
    return df

class Search:
    def __init__(self):
        self.beir =  process_df_for_beir
        self.lotte = process_df_for_lotte
        
        
def analyze_query_type(json_data):
    '''
    This is the function to analyze query types from the given json data
    '''
    search = Search()
    def find_match(json_data):
        if "lotte_dev" in json_data:
            return "Query Type Distribution for Lotte DEV", "lotte_query_dev.png", getattr(search, "lotte")
        elif "lotte_test" in json_data:
            return "Query Type Distribution for Lotte TEST", "lotte_query_test.png",  getattr(search, "lotte")
        else:
            return "Query Type Distribution for BEIR", "beir_query.png",  getattr(search, "beir")
            
    with open(json_data, "r") as f:
        data = json.load(f)
        
    title, save, func = find_match(json_data)
    plot_normal_heatmap(func(json_to_pd(data)), title = title, save = save)
    print(f"Saved as {save}")

def get_lotte(split = "test"):
    '''
    Returns lotte data split with msmarco in it. 
    '''
    data_paths = list(map(lambda x: os.path.join("../master_thesis_ai/lotte_beir_format", x),os.listdir("../master_thesis_ai/lotte_beir_format")))
    def get_lotte_dev():
        data_names = [
            d for d in data_paths if d.split("/")[-1].split("_")[-1] == "dev"
        ] + ["../master_thesis_ai/beir_data/msmarco"]
        return {data_name: GenericDataLoader(data_name).load("dev" if data_name != "../master_thesis_ai/beir_data/msmarco" else "train") for data_name in data_names}
            
    def get_lotte_test():
        data_names = [
            d for d in data_paths if d.split("/")[-1].split("_")[-1] == "test"
        ] + ["../master_thesis_ai/beir_data/msmarco"]
        return {data_name: GenericDataLoader(data_name).load("test" if data_name != "../master_thesis_ai/beir_data/msmarco" else "train") for data_name in data_names}
    
    return get_lotte_test() if split == "test" else get_lotte_dev()

def get_beir():
    data_list = os.listdir("../master_thesis_ai/beir_data")
    data_list.remove("cqadupstack")
    data_paths = list(
        map(
            lambda x: os.path.join("../master_thesis_ai/beir_data", x), data_list
        )
    )
    data_paths += list(
        map(
            lambda x: os.path.join("../master_thesis_ai/beir_data/cqadupstack", x), os.listdir("../master_thesis_ai/beir_data/cqadupstack")
        )
    )
    return {data_name: GenericDataLoader(data_name).load("test" if data_name != "../master_thesis_ai/beir_data/msmarco" else "train") for data_name in data_paths}

def normalize(counter):
    total_count = sum(counter.values())
    return {key: value / total_count for key, value in counter.items()}

def get_word_freq(text_iter, queries = False):
    # Put all the words in the corpus into a list
    words = Counter()
    tokenizer = RegexpTokenizer(r'\w+')
    for text in tqdm.tqdm(text_iter.values()):
        if not queries:
            text = text['title'] + text['text'] if text['title'] != "" else text['text']
        tokenized = tokenizer.tokenize(text)
        tokenized = [w.lower() for w in tokenized]
        words.update(tokenized)

    return normalize(words)

def normalized_jaccard_similarity(vocab_1: vocab_tuple, vocab_2: vocab_tuple):
    if vocab_1.name not in memo:
        memo[vocab_1.name] = {}
    # If we already calculate the jaccard similarity for these datasets, return from memo
    if memo[vocab_1.name].get(vocab_2.name, None) is not None:
        return memo[vocab_1.name][vocab_2.name]
    words = set(vocab_1.vocab.keys()).union(set(vocab_2.vocab.keys()))
    up = 0
    down = 0
    for k in words:
        word_freq_1, word_freq_2 = vocab_1.vocab.get(k, 0), vocab_2.vocab.get(k,0)
        up += min(word_freq_1, word_freq_2)
        down += max(word_freq_1, word_freq_2)
    memo[vocab_1.name][vocab_2.name] = up/down
    return up/down

def plot_heatmap(df, title, save):
    df = df.fillna(0)
    fig = plt.figure(figsize = (20,20))
    heatmap = sns.heatmap(df, vmin=0, annot=True, cmap='Reds', fmt = ".2f", square = True, annot_kws={"fontsize":15})
    heatmap.set_title(title, fontdict={'fontsize':24}, pad=16)
    heatmap.figure.savefig(save, bbox_inches='tight')

def plot_normal_heatmap(df, title, save):
    df = df.fillna(0)
    heatmap = sns.heatmap(df, vmin=0, annot=True, cmap='Reds', fmt=".1%", square = False, annot_kws={"fontsize":12})
    heatmap.set_title(title, fontdict={'fontsize':24}, pad=16)
    heatmap.figure.savefig(save, bbox_inches='tight')


def return_query_type(query: str) -> str: 
    first_word = query.split(" ")[0].lower().strip()
    yes_and_no = ["is", "was", "are", "were", "do", "does", "did", "have", "has", "had", "should", "can", "would", "could", "am", "shall"]
    if first_word in ["what", "what\'s"]:
        return "What"
    elif first_word in ["how", "how\'s"]:
        return "How"
    elif first_word in ["why", "why\'s"]:
        return "Why"
    elif first_word in ["when", "when\'s"]:
        return "When"
    elif first_word in ["where", "where\'s"]:
        return "Where"
    elif first_word in ["which", "which\'s"]:
        return "Which"
    elif first_word in ["who", "who\'s"]:
        return "Who"
    elif first_word in yes_and_no:
        return "Y/N"
    else:
        return "Declerative"
    
    
def get_word_freq_from_text(text):
    words = Counter()
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text)
    words = Counter(tokenized)
    return normalize(words)
      
def get_avg_overlap(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    m = 0
    total = 0
    for v in data.values():
        for v_ in v.values():
            m += v_
            total += 1
    return m / total


def calculate_overlap(text_1, key_1, text_2, key_2):
    vocab_1 = get_word_freq_from_text(text_1)
    vocab_2 = get_word_freq_from_text(text_2) 
    data_1, data_2 = vocab_tuple(key_1, vocab_1), vocab_tuple(key_2, vocab_2)
    return normalized_jaccard_similarity(data_1, data_2)


def plot_similarity_matrix(matrix: Dict, title: str, save: str, column_names: List[str], raw_column_names: List[str]):
    # From the dictionary of similarities, creatas a square dataframe, and then plots a heatmap
    # Supply pre-processed column names.
    square_df = pd.DataFrame(index=column_names, columns=column_names)
    for row, row_ in zip(raw_column_names, column_names):
        for col, col_ in zip(raw_column_names, column_names):
            if row == col:
                square_df.loc[row_, col_] = 1.0
            else:
                square_df.loc[row_, col_] = [dict_val[col] for dict_val in matrix[row] if col in dict_val][0]
    
    plot_heatmap(square_df, title, save)
