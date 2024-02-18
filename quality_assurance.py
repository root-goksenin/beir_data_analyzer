from utils import get_lotte, get_beir
from beir.datasets.data_loader import GenericDataLoader
import json 
from multiprocessing import Process, Manager
import pandas as pd 


lotte_test = get_lotte()
lotte_dev = get_lotte("dev")
beir = get_beir()

def check_answers(data_path, split):
    corpus, queries, qrels = GenericDataLoader(data_path).load(split)
    total_missing_queries = 0
    total_missing_docs = 0
    query_overlap = {}
    missing_queries = []
    for key,val in qrels.items():
        docs = []
        if key not in queries:
            total_missing_queries += 1
            missing_queries.append(key)
        for doc_id in val.keys():
            if doc_id not in corpus:
                total_missing_docs += 1
                docs.append(doc_id)
        if len(docs) > 0:
            query_overlap[key] = docs
    return {"total_missing_queries" : total_missing_queries, "total_missing_documents": total_missing_docs, "missing_document_numbers" : query_overlap, "missing_query_numbers": missing_queries }
                 
                 
def populate_summary(summary, data, split):
    summary[data] = check_answers(data, split)  
    
    
    
def analyse_summary():
    with open("summary.json", "r") as file:
        data = json.load(file)
   
    missing_documents = {} 
    for k,v in data.items():
        if v['total_missing_documents'] > 0:
            missing_documents[k] = {'missed' : v['missing_document_numbers'], 'total' : len(v['missing_document_numbers'])}
     
    df = pd.DataFrame.from_dict(missing_documents).transpose().sort_values(by = 'total', ascending=False)
    print(df)
                    
if __name__ == "__main__":
    # file = "summary.json"
    
    with Manager() as manager:
        summary = manager.dict()
        processes = [] 
        
        for val in lotte_test:
            processes.append(Process(target = populate_summary, args=(summary, val, "test")))
        for val in lotte_dev:
            processes.append(Process(target = populate_summary, args=(summary, val, "dev")))
        # for val in beir:
        #     processes.append(Process(target = populate_summary, args=(summary, val, "test")))
        
        [p.start() for p in processes]
        [p.join() for p in processes]
        print(summary)
        # with open(file, 'w') as f:
        #     json.dump(summary.copy(), f)
    
    
    # print(check_answers("lotte_beir_format_new/pooled_search_dev", "dev"))
    # print(check_answers("lotte_beir_format_new/pooled_forum_dev", "dev"))
    # print(check_answers("lotte_beir_format_new/pooled_search_test", "test"))
    # print(check_answers("lotte_beir_format_new/pooled_forum_test", "test"))

    queries, documents = analyse_summary()
    # print(documents)
        