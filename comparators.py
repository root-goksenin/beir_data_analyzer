
from beir.datasets.data_loader import GenericDataLoader
from typing import List, Dict
import os 
import json 
import tqdm
from utils import get_word_freq, normalized_jaccard_similarity, plot_heatmap, return_query_type, normalize, calculate_overlap, vocab_tuple
from collections import defaultdict

class CorpusComparator():
    def __init__(self, data_loaders):
        self.corpus_loaders: Dict[str, GenericDataLoader] = data_loaders
    
    def run(self, output_file: str):
        return self.compare_corpuses(output_file)
        
    def compare_corpuses(self, output_file: str):
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                self.similarities = json.load(f)
        else:
            self._create_corpus_similarity_matrix(output_file)
        return self.similarities

    def _create_corpus_similarity_matrix(self, output_file):
        vocabs = {
            data: get_word_freq(corpus)
            for data, (corpus, _, _) in self.corpus_loaders.items()
        }
        self.similarities = {}
        for key_1, vocab_1 in vocabs.items():
            for key_2, vocab_2 in vocabs.items():
                if key_1 != key_2: 
                    if key_1 not in self.similarities:
                        self.similarities[key_1] = []
                    data_1, data_2 = vocab_tuple(key_1, vocab_1), vocab_tuple(key_2, vocab_2)
                    self.similarities[key_1].append({key_2: normalized_jaccard_similarity(data_1, data_2)})
        
        with open(output_file, 'w') as writer:
            json.dump(self.similarities, writer)
        

class QRelsComparator():
    def __init__(self,task, corpus_loaders):
        self.corpus_loaders: Dict[str, GenericDataLoader] = corpus_loaders
        self.task = task
    
    
    def run(self, output_file : str):
        func = getattr(self, f"check_{self.task.value}")
        return func(output_file)
    
    def check_query_overlap(self, output_file: str = None):
        vocabs = {
            key: get_word_freq(queries, queries = True)
            for key, (_, queries, _) in self.corpus_loaders.items()
        }
        self.similarities = {}
        for key_1, vocab_1 in vocabs.items():
            for key_2, vocab_2 in vocabs.items():
                # Ignore the same datasrt
                if key_1 != key_2: 
                    if key_1 not in self.similarities:
                        self.similarities[key_1] = []
                    data_1, data_2 = vocab_tuple(key_1, vocab_1), vocab_tuple(key_2, vocab_2)
                    # Compute the normalized jaccard similarity between dataset 1 and dataset 2
                    self.similarities[key_1].append({key_2: normalized_jaccard_similarity(data_1, data_2)})
        
        with open(output_file, 'w') as writer:
            json.dump(self.similarities, writer)
        
        return self.similarities
    
    
    def check_query_type_distribution(self, output_file: str = None, split = "test"):
        types = {}
        for key, (_, queries, _) in self.corpus_loaders.items():
            c = defaultdict(int)
            for query in queries.values():
                c[return_query_type(query)] += 1
            types[key] = normalize(c)
        
        with open(output_file, "w") as f:
            json.dump(types, f)
        
        return types
        
            
    
    def check_query_answer_lexical_overlap(self, output_file: str = None, split = "test"):
        self.query_overlaps = {} 
        for key, (corpus, queries, qrels) in self.corpus_loaders.items():
            query_overlap = {}
            for key,val in tqdm.tqdm(qrels.items()):
                if key not in query_overlap:
                    query_overlap[key] = {}
                query = queries[key]
                for doc_id in val.keys():
                    try:
                        doc = corpus[doc_id]['text']
                        if corpus[doc_id]['title'] != '':
                            doc = corpus[doc_id]['title'] + " " + doc
                        # Calculate overlap between the query, and answer doc.
                        query_overlap[key][doc_id] = calculate_overlap(query, key, doc, doc_id)
                    except KeyError as e:
                        print(f'Got {e}, ignoring')
            self.query_overlaps[key] = query_overlap
        
        with open(output_file, 'w') as writer:
            json.dump(self.similarities, writer)
        
        return self.similarities
    
    