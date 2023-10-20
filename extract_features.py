import pandas as pd
import os 
import sys
import csv
import time
import string
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
punctuation = string.punctuation
stopwords.update(set(punctuation))
nlp = spacy.load('en_core_web_sm')
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

def extract_features(infile, outfile, out_dir, features, part="hypothesis", file_type="train"):
    spell = SpellChecker()
    features = [x.lower() for x in features]
    f_handles={}
    par_file = os.path.join(out_dir, f"{part}_nlp.json")
    ext_features = defaultdict(dict)
    with open(infile, "r") as f, \
        open(outfile, "wb") as f_out, \
        open(par_file, "wb") as f_par:
        f_reader = csv.DictReader(f)
        
        for f in features:
            f_handles[f]= open (os.path.join(out_dir, f"{file_type}.{part}.{f}.features"), "w")
            #print(os.path.join(out_dir, f+".features"))
        p_list  = []
        i = 0
        ner_dict = defaultdict(set)
        if "ner" in features:
            ner_f = open(os.path.join(out_dir, f"{file_type}_ner_dict"), "wb")
        for line in f_reader:
            i += 1
            #if i < 100:
                #print(ext_features)
                #print(i)
            #else:
            #    exit()
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
                    l = detect_overlap(nlp(line["premise"]), parsed_line)
                elif f == "sentiment":
                    l = detect_sent(parsed_line)
                elif f == "negation":
                    l = detect_neg(parsed_line)
                elif f == "ner":
                    l = detect_ner(parsed_line)
                    for (t, ner_label) in l[0]:
                        ner_dict[ner_label].add(t)
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
                w_dict[line["guid"]] = l[1]
                json_line = json.dumps(w_dict, ensure_ascii=False)
                f_handles[f].write(json_line + "\n")
                f_handles[f].flush()
                if len(f) != 0 and len(l[1]) != 0:
                    #for x, indexes in l[1].items():
                    #    print(x, indexes)
                    for x, indexes in l[1].items():
                        feature = f+"\t"+str(x)
                        #print("feature:", feature)
                        index_list = indexes
                        ext_features[guid][feature] = index_list
                    #print("ext_dict:", ext_features)
                    #ext_features[guid].append({f+"\t"+str(x): indexes for x, indexes in l[1].items()})
        if "ner" in features:
            #print(ner_dict)
            pickle.dump(ner_dict, ner_f)
            ner_f.close
        pickle.dump(p_list,f_par)
        pickle.dump(ext_features, f_out)
        #print("ext_features:", ext_features)
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

def detect_word(p_line):
    tokens = []
    for token in p_line:
        if token.lemma_ == "-PRON-":
            tokens.append(token.text)
        else:
            tokens.append(token.lemma_)
    #tokens = [token.lemma_ for token in line]
    tuples = defaultdict(list)
    for i, t in enumerate(tokens):
        tuples[t.lower()].append(i) 
    #print(tuples) 
    #tokens = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(line)]
    return list(set(tokens)),  tuples

def detect_overlap(premise, hypothesis):
        
    #premise_tokens = word_tokenize(premise)
    #hypothesis_tokens = word_tokenize(hypothesis)
    premise_tokens = []
    hypothesis_tokens = []
    for token in premise:
        if token.lemma_ == "-PRON-":
            premise_tokens.append(token.text)
        else:
            premise_tokens.append(token.lemma_)

    for token in hypothesis:
        if token.lemma_ == "-PRON-":
            hypothesis_tokens.append(token.text)
        else:
            hypothesis_tokens.append(token.lemma_)
    tuples = defaultdict(list)
    for i, h in enumerate(hypothesis_tokens):
        if h.lower() not in stopwords and h in premise_tokens:
            tuples["overlap"].append(i)
    if len(tuples["overlap"]) > 0:       
        return ["overlap"], tuples
    else:
        return [], {}

def detect_sent(p_line):

        pos_list = set(opinion_lexicon.positive())
        neg_list = set(opinion_lexicon.negative())
        
        positives = list()
        negatives = list()
        #hypothesis_token = word_tokenize(line)
        #hypothesis_tokens = [x for x in line]
        tuples = defaultdict(list)
        for i, p in enumerate(p_line):
            if p.lemma_ in pos_list and p.pos_ in ["ADJ", "ADV"]:
                positives.append(p)
                tuples["positive"].append(i)
                #tuples["sent-any"].append(i)
            elif p.lemma_ in neg_list and p.pos_ in ["ADJ", "ADV"]:
                negatives.append(p)
                tuples["negative"].append(i)
                #tuples["sent-any"].append(i)
            
        if len(positives)-len(negatives) > 0:
            #return ["sent_any", "positive"], tuples
            return ["positive"], tuples
        elif len(positives)-len(negatives) < 0:
            return ["negative"], tuples
        else:
            return [], {}


def detect_neg(p_line):
    hypothesis_dep = [token.dep_ for token in p_line]
    tuples = defaultdict(list)
    for i, dep in enumerate(hypothesis_dep):
        if dep == "neg":
            tuples["negation"].append(i)

    if len(tuples) > 0:
        return ["negation"], tuples
    else:
        return [], {}

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
    tokens = [token.text for token in p_line]
    ner = [(token.text, token.label_) for token in p_line.ents]
    tuples = defaultdict(list)
    for e in list(p_line.ents):
        #print(tokens)
        #print(e)
        indexes = [tokens.index(x) for x in e.text.split() if x in tokens]
        tuples[e.label_].extend(indexes)
    return list(set(ner)), tuples

def detect_tense(p_line):
    tuples = defaultdict(list)
    tense = defaultdict(list)
    hypothesis_tag = [token.tag_ for token in p_line]
    #tense = [tense_dict[h] for i, h in enumerate(hypothesis_tag) if h in tense_dict]
    for i, h in enumerate(hypothesis_tag):
        if h in tense_dict:
            tense[tense_dict[h]].append(i)
    #print("tense_dict:", tense)        
    if "past" in tense:
        tuples["past"] = tense["past"]
    elif "future" in tense:
        tuples["future"] = tense["future"]
    elif "present" in tense:
        tuples["present"] =  tense["present"]
        
    return list(set(tense)), tuples

def detect_typo(spell, line):
    hypothesis_token = word_tokenize(line)
    tuples = defaultdict(list)
    for i, h in enumerate(hypothesis_token):
        if spell.correction(h) != h:
            tuples["typo"].append(i)
    
    if len(tuples) > 0:
        return ["typo"], tuples
    else:
        return [], {}

def detect_pronoun(p_line):
    #hypothesis_pos = [token.pos_ for token in p_line]
    tuples = defaultdict(list)
    for i, x in enumerate(p_line):
        if x.pos_ == "PRON" and x.tag_ == "PRP":
            tuples["pronoun"].append(i)

    if len(tuples) > 0:
        return ["pronoun"], tuples
    else:
        return [], {}
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

 
