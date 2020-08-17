import argparse
import json
from nir.utils import create_filter_query_function, change_bm25_parameters
from mmnrm.utils import set_random_seed, load_model_weights, load_model
from mmnrm.training import TrainCollection, TestCollection
from mmnrm.dataset import TrainCollectionV2, TestCollectionV2, sentence_splitter_builderV2
from mmnrm.evaluation import BioASQ_Evaluator
from mmnrm.modelsv2 import deep_rank
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import backend as K

print("Tensorflow version: ",tf.__version__)


import math
import os
import sys
import pickle
import numpy as np
import time

import subprocess
import tempfile
import shutil

from nir.embeddings import FastText, Word2Vec

import io
from nir.tokenizers import Regex, BioCleanTokenizer, BioCleanTokenizer2

def load_neural_model(path_to_weights):
    
    rank_model = load_model(path_to_weights, change_config={"return_snippets_score":True})
    tk = rank_model.tokenizer
    
    model_cfg = rank_model.savable_config["model"]
    
    max_input_query = model_cfg["max_q_length"]
    max_input_sentence = model_cfg["max_s_length"]
    max_s_per_q_term = model_cfg["max_s_per_q_term"]
    
    # redundant code... replace
    max_sentences_per_query = model_cfg["max_s_per_q_term"]

    pad_query = lambda x, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                       maxlen=max_input_query,
                                                                                       dtype=dtype, 
                                                                                       padding='post', 
                                                                                       truncating='post', 
                                                                                       value=0)

    pad_sentences = lambda x, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                           maxlen=max_input_sentence,
                                                                                           dtype=dtype, 
                                                                                           padding='post', 
                                                                                           truncating='post', 
                                                                                           value=0)

    pad_docs = lambda x, max_lim, dtype='int32': x[:max_lim] + [[]]*(max_lim-len(x))

    idf_from_id_token = lambda x: math.log(tk.document_count/tk.word_docs[tk.index_word[x]])

    train_sentence_generator, test_sentence_generator = sentence_splitter_builderV2(tk, 
                                                                                      max_sentence_size=max_input_sentence,
                                                                                      mode=4)
    
    def test_input_generator(data_generator):

        data_generator = test_sentence_generator(data_generator)

        for _id, query, docs in data_generator:

            #tokenization
            query_idf = list(map(lambda x: idf_from_id_token(x), query))

            tokenized_docs = []
            ids_docs = []
            offsets_docs = []

            for doc in docs:

                padded_doc = pad_docs(doc["text"], max_lim=max_input_query)
                for q in range(len(padded_doc)):
                    padded_doc[q] = pad_docs(padded_doc[q], max_lim=max_sentences_per_query)
                    padded_doc[q] = pad_sentences(padded_doc[q])
                tokenized_docs.append(padded_doc)
                ids_docs.append(doc["id"])
                offsets_docs.append(doc["offset"])

            # padding
            query = pad_query([query])[0]
            query = [query] * len(tokenized_docs)
            query_idf = pad_query([query_idf], dtype="float32")[0]
            query_idf = [query_idf] * len(tokenized_docs)

            yield _id, [np.array(query), np.array(tokenized_docs), np.array(query_idf)], ids_docs, offsets_docs

    return rank_model, test_input_generator

def rerank(model, t_collection):
    
    generator_Y = t_collection.generator()
                
    q_scores = defaultdict(list)

    for i, _out in enumerate(generator_Y):
        query_id, Y, docs_info, offsets_docs = _out
        s_time = time.time()
        
        scores, q_sentence_attention = model.predict(Y)
        scores = scores[:,0].tolist()
            
        print("\rEvaluation {} | time {}".format(i, time.time()-s_time), end="\r")
        #q_scores[query_id].extend(list(zip(docs_ids,scores)))
        for i in range(len(docs_info)):
            q_scores[query_id].append((docs_info[i], scores[i], q_sentence_attention[i], offsets_docs[i]))

    # sort the rankings
    for query_id in q_scores.keys():
        q_scores[query_id].sort(key=lambda x:-x[1])
        q_scores[query_id] = q_scores[query_id][:10]
    
    return q_scores


def get_snippets(query_data, threashold):
    final_snippets = []
    
    # remove duplicated during the algorithm
    _cache = set()
    
    for d, scored_docs in enumerate(query_data):
        #print("number of docs visited", d+1)
        if len(final_snippets)>10:
            break

        snippet_score = scored_docs[2]
        answer_shape = snippet_score.shape
        snippet_score_1d = snippet_score.ravel()
        
        snippets_by_query = organize_snippets(scored_docs[3], answer_shape)
        
        permutation = (-snippet_score_1d).argsort()
        snippet_score_1d = snippet_score_1d[permutation]
        _filter = snippet_score_1d>threashold
        indices = np.arange(0, snippet_score_1d.shape[0])

        top_indices = indices[permutation][_filter]

        for _top_indice in top_indices:

            q_index = int(_top_indice / answer_shape[1])
            s_index = int(_top_indice % answer_shape[1])
            try:
                snippet = snippets_by_query[q_index][s_index]
                
                # entry for duplicates
                entry = (scored_docs[0], snippet[0], snippet[1][0], snippet[1][1])

                if entry in _cache:
                    continue 
                _cache.add(entry)
                    
                final_snippets.append((scored_docs[0], snippet))
            except Exception as e:
                """
                print(_top_indice)
                print(q_index, s_index)
                print(indices[permutation])
                v = np.arange(0, snippet_score_1d.shape[0])
                print( np.reshape(v, (30,5)) )
                print(snippet_score)
                print(Y[1][0][2])
                print(Y[0])
                print(organize_snippets(scored_docs[3], answer_shape))
                print(snippets_by_query[q_index])"""
                pass
                #raise e
    
    return final_snippets[:10]

def organize_snippets(offsets, answer_shape):
    queries = [[] for _ in range(answer_shape[0])]

    for offset in offsets:
        for q_ids in offset[-1]:
            if len(queries[q_ids])<5:
                queries[q_ids].append((offset[0], offset[1], offset[2]))

    return queries

def snippet_retrieval(query_scores, threashold):
    
    answers = []

    for j, data in enumerate(query_scores.items()):
        print("evaluating",j,end="\r")
        q_id, q_data = data 

        snippets = []
        for _snippet in get_snippets(q_data, threashold):
            _snippet_pmid = _snippet[0]
            _snippet_section = _snippet[1][0]
            _snippet_start = _snippet[1][1][0] if _snippet_section == "title" else _snippet[1][1][0]-1
            _snippet_end = _snippet[1][1][1]
            _snippet_text = _snippet[1][2]

            snippets.append({"offsetInBeginSection":_snippet_start,
                             "offsetInEndSection":_snippet_end,
                             "beginSection":_snippet_section,
                             "endSection":_snippet_section,
                             "text":_snippet_text,
                             "document":"http://www.ncbi.nlm.nih.gov/pubmed/"+_snippet_pmid})
        answers.append({"id":q_id,
                      "snippets":snippets,
                      "documents":list(map(lambda x:"http://www.ncbi.nlm.nih.gov/pubmed/"+x[0], q_data))})
        
    return answers

def save_answers_to_file(answers, name):
    _name = name+"_answer.json"
    
    with open(_name,"w", encoding="utf-8") as f:
        json.dump({"questions":answers},f)
        
    return _name

if __name__ == "__main__":
    
    # argparsing
    parser = argparse.ArgumentParser(description='This is program to make evaluation over the bioASQ test data')
    parser.add_argument('model_weight_path', type=str)
    parser.add_argument('test_set', type=str)
    parser.add_argument('candidate_set', type=str)
    parser.add_argument('-threashold', dest='threashold', default=0.1)


    args = parser.parse_args()
    
    threashold = 0.1    
    
    # load queries
    with open(args.test_set, "r", encoding="utf-8") as f:
        queries = json.load(f)["questions"]
    
    evaluate = all(map(lambda x: "documents" in x, queries))
    print("Running in the evaluation mode, since the test set has positive documents")
    
    # load candidates set
    with open(args.candidate_set, "r", encoding="utf-8") as f:
        candidate_set = json.load(f)
        
    # validation, since candidate_set must have all the queries from test_set
    if not all([x["id"] in candidate_set for x in queries]):
        print("test set and candidate set has unmatchable ids")
        sys.exit(1)
    
    # build collection dataset for the neural model
    # create the test set
    query_list = list(map(lambda x:{"id":x["id"], "query":x["body"]}, queries))
    test_collection = TestCollectionV2(query_list, candidate_set).batch_size(100)
    
    # load the model
    rank_model, test_input_generator = load_neural_model(args.model_weight_path)
    
    # add the input generator to the collection
    test_collection.set_transform_inputs_fn(test_input_generator)
    
    # perform the neural rerank
    query_scores = rerank(rank_model, test_collection)
    
    # run snippet retrieval algorithm
    answers = snippet_retrieval(query_scores, args.threashold)
    
    # output metrics if possible
    if evaluate:
        temp_dir = tempfile.mkdtemp()
        try:
            save_answers_to_file(answers, os.path.basename(args.test_set))
            answers_file = save_answers_to_file(answers, temp_dir)
            
            # evaluate
            bioasq_results = subprocess.Popen(
                ['java',
                 '-Xmx10G',
                 '-cp', '$CLASSPATH:./download_folder/BioASQEvaluation/BioASQEvaluation.jar',
                 'evaluation.EvaluatorTask1b', 
                 '-phaseA', 
                 '-e', "8",
                 args.test_set,
                 answers_file],
                stdout=subprocess.PIPE, shell=False).communicate()[0]
            
            print(bioasq_results)

        except Exception as e:
            raise e # maybe handle the exception in the future
        finally:
            # always remove the temp directory
            print("Remove {}".format(temp_dir))
            shutil.rmtree(temp_dir)
    
    
    else:
        save_answers_to_file(answers, os.path.basename(args.test_set))
