import pandas as pd
import os 
import sys
import csv
import time
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
import scipy.stats
import spacy
from spacy.lang.en import English
import nltk
from nltk import pos_tag
import spicy
import pickle
import pandas
import random
import math
import multiprocessing
import tensorflow as tf
from spellchecker import SpellChecker
import json
from pathlib import Path
from process_data import check_path
stopwords = set(stopwords.words('english'))
nlp = spacy.load('en')
'''MD  Modal verb (can, could, may, must)
VB  Base verb (take)
VBC Future tense, conditional
VBD Past tense (took)
VBF Future tense
VBG Gerund, present participle (taking)
VBN Past participle (taken)
VBP Present tense (take)
VBZ Present 3rd person singular (takes)'''
tenses = {"future":["MD", "VBC", "VBF"], "past":["VBD", "VBN"], "present":["VBP", "VBZ", "VBG", "VB"]}
tense_dict = {pos:tense for tense in tenses for pos in tenses[tense]}
print("tense_dict:", tense_dict)
print(stopwords)

import numpy as np
import scipy.stats
from scipy.stats import entropy
def js(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
   # normalize
    p = p/p.sum()
    q = q/q.sum()
    #return scipy.stats.entropy(t_list, p_list)        
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2

def mse(l):
    ave= np.mean(l)
    msd = 0
    for x in l:
        msd += (x-ave)**2/ave
    return msd

def extract_features(infile, outfile, out_dir, features, part="hypothesis"):
    spell = SpellChecker()
    features = [x.lower() for x in features]
    f_handles={}
    par_file = os.path.join(out_dir, f"{part}_nlp.json")
    ext_features = defaultdict(set)
    with open(infile, "r") as f, \
        open(outfile, "wb") as f_out, \
        open(par_file, "wb") as f_par:
        f_reader = csv.DictReader(f)
        
        for f in features:
            f_handles[f]= open (os.path.join(out_dir, f"{part}.{f}.features"), "w")
            #print(os.path.join(out_dir, f+".features"))
        p_list  = []
        for line in f_reader:
            guid = line["guid"]
            par_dict = {}
            parsed_line = nlp(line[part])  
            par_dict[guid] = parsed_line
            #line = json.dumps(par_dict, ensure_ascii=False)
            #f_par.write(line + "\n")
            #p_list.append(parsed_line)
            #print("line:", line)
            l = []
            for f in features:
                if f == "word":
                    #l = detect_word(line[part])
                    l = detect_word(parsed_line)
                elif f == "overlap":
                    l = detect_overlap(line["premise"], line["hypothesis"])
                elif f == "sentiment":
                    l = detect_sent(line[part])
                elif f == "negation":
                    l = detect_neg(parsed_line)
                elif f == "ner":
                    l = detect_ner(parsed_line)
                elif f == "tense":
                    l = detect_tense(parsed_line)
                elif f == "typo":
                    l = detect_typo(spell, line[part])
                elif f == "pronoun":
                    l = detect_pronoun(parsed_line)
                else:
                    print(f"There is no feature named {f}")
                    continue

                w_dict = {}
                w_dict[line["guid"]] = l
                json_line = json.dumps(w_dict, ensure_ascii=False)
                f_handles[f].write(json_line + "\n")
                f_handles[f].flush()
                if len(f) != 0:
                    ext_features[guid].update([f+"\t"+x for x in l])
        pickle.dump(p_list,f_par)
        pickle.dump(ext_features, f_out)
        print("ext_features:", ext_features)
        #pickle.load(out_parsed_file, p_list)
        for f in features:
            f_handles[f].close()
        return ext_features
        

'''def processor(file_name, out_parsed_file, features, out_dir, flag, part="hypothesis"):
    spell = SpellChecker()
    features = [x.lower() for x in features]
    f_handles={}
    with open(file_name, "r") as f, \
        open(out_parsed_file, "wb") as f_par:
        f_reader = csv.DictReader(f)
        for f in features:
            f_handles[f]= open (os.path.join(out_dir, f"{part}.{flag}.{f}.features"), "w")
            #print(os.path.join(out_dir, f+".features"))

        p_list  = []
        for line in f_reader:
            par_dict = {}
            parsed_line = nlp(line[part])  
            par_dict[line["guid"]] = parsed_line
            #line = json.dumps(par_dict, ensure_ascii=False)
            #f_par.write(line + "\n")
            p_list.append(parsed_line)
            #print("line:", line)
            l = []
            for f in features:
                if f == "word":
                    l = detect_word(line[part])
                elif f == "overlap":
                    l = detect_overlap(line["premise"], line["hypothesis"])
                elif f == "sentiment":
                    l = detect_sent(line[part])
                elif f == "negation":
                    l = detect_neg(parsed_line)
                elif f == "ner":
                    l = detect_ner(parsed_line)
                elif f == "tense":
                    l = detect_tense(parsed_line)
                elif f == "typo":
                    l = detect_typo(spell, line[part])
                elif f == "pronoun":
                    l = detect_pronoun(parsed_line)

                w_dict = {}
                w_dict[line["guid"]] = l
                json_line = json.dumps(w_dict, ensure_ascii=False)
                f_handles[f].write(json_line + "\n")
                f_handles[f].flush()
        pickle.dump(p_list,f_par)
        #pickle.load(out_parsed_file, p_list)
        for f in features:
            f_handles[f].close() '''

def detect_word(line):
    
    tokens = [token.lemma_ for token in line]
    #tokens = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(line)]
    return list(set(tokens))

def detect_overlap(premise, hypothesis):
        
    premise_tokens = word_tokenize(premise)
    hypothesis_tokens = word_tokenize(hypothesis)
    for p in premise_tokens:
        if p not in stopwords and p in hypothesis_tokens:
            return ["overlap"]
    return []

def detect_sent(line):

        pos_list = set(opinion_lexicon.positive())
        neg_list = set(opinion_lexicon.negative())
        
        positives = list()
        negatives = list()
        hypothesis_token = word_tokenize(line)

        for p in hypothesis_token:
            if p in pos_list:
                positives.append(p)
            elif p in neg_list:
                negatives.append(p)
            
        if len(positives)-len(negatives) > 0:
            return ["sent_any", "positive"]
        elif len(positives)-len(negatives) < 0:
            return ["sent_any", "negative"]
        else:
            return []


def detect_neg(p_line):
    hypothesis_dep = [token.dep_ for token in p_line]
    if "neg" in hypothesis_dep:
        return ["negation"]
    else:
        return []

'''       return ["any", \
            "PERSON",  #People, including fictional. 
            "NORP",  #Nationalities or religious or political groups.
            "FAC",   #Buildings, airports, highways, bridges, etc.
            "ORG", #Companies, agencies, institutions, etc.
            "GPE", #Countries, cities, states.
            "LOC", #Non-GPE locations, mountain ranges, bodies of water.
            "PRODUCT", #Objects, vehicles, foods, etc. (Not services.)
            "EVENT", #Named hurricanes, battles, wars, sports events, etc.
            "WORK_OF_ART", #Titles of books, songs, etc.
            "LAW", #Named documents made into laws.
            "LANGUAGE", #Any named language.
            "DATE", #Absolute or relative dates or periods.
            "TIME", #Times smaller than a day.
            "PERCENT", #Percentage, including ”%“.
            "MONEY", #  Monetary values, including unit.
            "QUANTITY", #Measurements, as of weight or distance.
            "ORDINAL", #"first", "second", etc.
            "CARDINAL" #Numerals that do not fall under another type
            ]'''

def detect_ner(p_line):
    ner = [token.label_ for token in p_line.ents]
    return list(set(ner))

def detect_tense(p_line):
    hypothesis_tag = [token.tag_ for token in p_line]
    tense = [tense_dict[h] for h in hypothesis_tag if h in tense_dict]
    return list(set(tense))

def detect_typo(spell, line):
    hypothesis_token = word_tokenize(line)
    for h in hypothesis_token:
        if spell.correction(h) != h:
            return ["typo"]
    return []

def detect_pronoun(p_line):
    hypothesis_dep = [token.pos_ for token in p_line]
    if "PRON" in hypothesis_dep:
        return ["pronoun"]
    else:
        return []
#def extract_features(features, in_file, out_file, log_dir, part="hypothesis"):
#    #part should be "premise" or "hypothesis" and "hypothesis" in default 
#    assert Path(in_file).isfile()
#    assert check_path(out_file)
#    
#    feature_dict = processor(in_file, out_file, log_dir, features, part)
    


if __name__ == "__main__":
    extract_features()
    '''tasks = ["SNLI", "QNLI", "MNLI","ARCT","ARCT2", "ROC", "RECLOR", "SWAG", "COPA", "COMMON_QA","RACE"]
    features = ["sentiment", "overlap", "negation", "typo", "tense", "ner", "word"]
    task = sys.argv[1]
    in_dir = "/home/shanshan/generate_noise/ESIM/data/dataset/"
    out_dir = f"/home/shanshan/traning_data/data/{task}"
    if os.path.exists(out_dir) ==  False:
        os.makedirs(out_dir)
    train_file = os.path.join(in_dir, task, "train", "original_train.csv")
    test_file = os.path.join(in_dir, task, "test", "original_test.csv")
    train_out_file = os.path.join(out_dir, "train_parse.json")
    test_out_file = os.path.join(out_dir, "test_parse.json")
    
    #for part in ["premise", "hypothesis"]:
    for part in ["hypothesis"]:
        processor(train_file, train_out_file, features, out_dir, "train", part)
        processor(test_file, test_out_file, features, out_dir, "test", part)'''

 
