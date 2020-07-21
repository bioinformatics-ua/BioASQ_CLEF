import os


def requirements_satisfied():
    """
    return true if all the requirements are satisfied, false otherwise
    """
    
    
    url = ""
    
    files_required = ["download_folder/tokenizers/bioasq_2020_RegexTokenizer.json",
                       "download_folder/embeddings/WORD2VEC_embedding_bioasq_gensim_iter_15_freq0_200_Regex_word2vec_bioasq_2020_RegexTokenizer",
                       "download_folder/pre-trained-word2vec/bioasq_gensim_iter_15_freq0_200_Regex_word2vec.bin",
                       "download_folder/pre-trained-word2vec/bioasq_gensim_iter_15_freq0_200_Regex_word2vec.bin.vectors.npy"]
    
    not_found = list(filter(lambda x: not os.path.exists(x), files_required))
    
    if len(not_found)>0:
        print("Please download the zip file available here: ")
        print("In more detail, the following files are missing")
        for file in not_found:
            print(f"{file} is missing")
            
        return False
    
    return True