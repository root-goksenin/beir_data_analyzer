

# Plot heatmap from the given json file



      
@click.command()
@click.option('--json', type=click.Choice(["lotte_dev", "lotte_test", "beir"]), help='The json file for plotting the heatmap')
def main(data_name, task):
    '''
    When the task is query_answer_lexical_overlap, create a new folder, and put them into the folder.
    '''
    task = Tasks(task)
    data_names, split = getattr(DataGettr(), data_name)
    analyze = Analyze(task, *data_names)
    out = analyze.run(output_file= get_save(task, data_name, "json"), split = split)
    if task in [Tasks.CORPUS_VOCAB_OVERLAP, Tasks.QUERY_VOCAB_OVERLAP]:
        plot_similarity_matrix(out, get_title(task, data_name), get_save(task, data_name, "png"), get_column_names(out, data_name))
    
if __name__ == "__main__":
    for task in ["query_type_distribution", "query_overlap", "query_answer_lexical_overlap","corpus_vocab_overlap"]:
        for data_name in ["lotte_dev", "lotte_test", "beir"]:
            print(get_title(Tasks(task),data_name))
            print(get_save(Tasks(task), data_name, "json"))