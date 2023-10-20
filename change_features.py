import pandas as pd
import os 
import sys
import string 
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
nlp = spacy.load('en_core_web_sm')
spell = SpellChecker()
import torch
#from transformers import BertTokenizer, BertModel, BertForMaskedLM
#from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)# OPTIONAL
import csv
from itertools import chain
from nltk.corpus import wordnet
import numpy as np
import scipy.stats
from scipy.stats import entropy
from tenseflow import change_tense as ct
import neuralcoref
neuralcoref.add_to_pipe(nlp)
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
punctuation = string.punctuation
stopwords.update(set(punctuation))
#print("tense_dict:", tense_dict)
#print(stopwords)
pos_list = set(opinion_lexicon.positive())
neg_list = set(opinion_lexicon.negative())
#print(pos_list)

#ner_dict = pickle.load(ner_file)
#tokenizer = BertTokenizer.from_pretrained('roberta-large')
#model = BertForMaskedLM.from_pretrained('roberta-large')
#tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
#model = RobertaForMaskedLM.from_pretrained('roberta-large')
#model.eval()
from transformers import pipeline
unmasker = pipeline('fill-mask', model='roberta-large')

def predict_masked_tokens(text):
    predict_list = [x["token_str"].strip() for x in unmasker(text)]
    return predict_list


'''def predict_masked_sent(text, top_k=100):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    print(tokenized_text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
    res_tokens = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        res_tokens.append(predicted_token)

        #print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))
    return res_tokens'''

def change_delete(questions, feature_caseids, guid_features, feature, change_type, ner_dict):
    print("feature:", feature)
    f =  feature.split("\t")[1]
    feature_type = feature.split("\t")[0]
    caseids = feature_caseids[feature]
    return_cases = []
    for ids in caseids:
        if ids in questions:
            cases = questions[ids]
            for case in cases:
                id = case["guid"]
                premise = case["premise"]
                hypothesis = case["hypothesis"]
                if feature_type in ["word", "overlap", "negation", "typo", "pronoun", "ner", "sentiment", "tense"]:
                    #print("feature:", feature, guid_features[id], id)
                    if feature in guid_features[id]:
                        indexes = guid_features[id][feature]
                        if change_type == "delete":
                            changed_hyp = delete_word(hypothesis, feature, indexes)
                        else:
                            changed_hyp = substitute_word(premise, hypothesis, feature, indexes, ner_dict=ner_dict)

                        r_dict ={}
                        r_dict["guid"] = case["guid"]
                        r_dict["premise"] = case["premise"]
                        r_dict["hypothesis"] = changed_hyp
                        r_dict["label"] = case["label"]
                        r_dict["exist"] = "yes"
                    else:
                        r_dict ={}
                        r_dict["guid"] = case["guid"]
                        r_dict["premise"] = case["premise"]
                        r_dict["hypothesis"] = case["hypothesis"]
                        r_dict["label"] = case["label"]
                        r_dict["exist"] = "no"
                    return_cases.append(r_dict)
    return return_cases

#def delete_features(premise, hypothesis, feature, indexes, delete=False):
    # sentiment feature will not be changed. and ner should be deleted
#            hyp_tokens = delete_word(hypothesis, feature, feature_type)
    
def delete_word(hypothesis, feature, indexes):
    if feature.split("\t")[0] in ["ner", "negation", "word"]:
        h_tokens = [x.text for x in nlp(hypothesis)]
    else:
        h_tokens = word_tokenize(hypothesis)
    f = feature[0]
    for i in indexes:
        #print("hypothesis, feature, indexes:", hypothesis, feature, indexes)
        #h_tokens.pop(i)
        h_tokens[i] = "UNK"
    changed_hyp = " ".join(h_tokens)
    return changed_hyp

def substitute_word(premise, hypothesis, feature, indexes, ner_dict = None):
    f = feature.split("\t")[0]
    fe = feature.split("\t")[1]
    parsed_hypothesis = nlp(hypothesis)  
    l = []
    if f == "word":
        #l = detect_word(line[part])
        l = change_word(parsed_hypothesis, indexes)
    elif f == "overlap":
        l = change_overlap(nlp(premise), parsed_hypothesis)
    elif f == "sentiment":
        l = change_sent(parsed_hypothesis, fe)
    elif f == "negation":
        l = change_word(parsed_hypothesis, indexes)
    elif f == "ner":
        l = change_ner(parsed_hypothesis, fe, indexes, ner_dict)
    elif f == "tense":
        l = change_tenses(hypothesis, fe)
    elif f == "typo":
        l = change_typo(hypothesis, indexes)
    elif f == "pronoun":
        l = change_pronoun(premise, hypothesis, parsed_hypothesis, indexes)
    else:
        print(f"There is no feature named {f}")

    return l

#def delete_word(line, indexes):
#    h_tokens = [x.text for x in line]
#    for i in indexes:
#        h_tokens[i] = "UNK"
#    changed_hyp = " ".join(h_tokens)
#    return changed_hyp

def change_word(p_hypothesis, indexes):
    h_tokens = [x.text for x in p_hypothesis]
    for i in indexes:
        h_tokens[i] = "UNK"
    changed_hyp = " ".join(h_tokens)
    #print(changed_hyp)
    return changed_hyp

def get_synonyms(w):
    synonyms = wordnet.synsets(w)
    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
    return lemmas

def change_overlap(p_premise, p_hypothesis):
        
    #premise_tokens = word_tokenize(premise)
    #hypothesis_tokens = word_tokenize(hypothesis)
    premise_tokens = [x.text for x in p_premise]
    hypothesis_tokens = [x.text for x in p_hypothesis]
    tuples = defaultdict(list)
    for i, h in enumerate(hypothesis_tokens):
        change = False
        h = h.lower()
        if h not in stopwords and h in premise_tokens:
            mask_tokens = [x for x in hypothesis_tokens]
            mask_tokens[i] = "<mask>"
            mask_sen = " ".join(mask_tokens)
            print("mask_sen:", mask_sen)
            #alters = predict_masked_sent(mask_sen)
            alters = predict_masked_tokens(mask_sen)
            synonyms = get_synonyms(h)
            print(alters)
            print(synonyms)
            for a in alters:
                if a not in premise_tokens and a not in stopwords and a in synonyms:
                    hypothesis_tokens[i] = a
                    change = True
                    break
            if change == False:
                    hypothesis_tokens[i] = "UNK"
                    
    changed_hyp = " ".join(hypothesis_tokens)
    return changed_hyp


def change_sent(p_hypothesis, feature):

    positives = list()
    negatives = list()
    #hypothesis_tokens = word_tokenize(hypothesis)
    hypothesis_tokens = [x.text for x in p_hypothesis]

    tuples = defaultdict(list)
    for i, p in enumerate(p_hypothesis):
        change = False
        if p.lemma_ in pos_list and p.pos_ in ["ADJ", "ADV"] and feature == "positive":
            #print("p:", p)
            #print("hypothesis_tokens:", hypothesis_tokens)
            mask_tokens = [x for x in hypothesis_tokens]
            mask_tokens[i] = "<mask>"
            mask_sen = " ".join(mask_tokens)
            alters = predict_masked_tokens(mask_sen)

            #print("alters:", alters)
            for s in alters:
                if s not in pos_list:
                    hypothesis_tokens[i] = s
                    change = True
                    break
            if change == False:
                hypothesis_tokens[i] = "UNK"

        elif p.lemma_ in neg_list and p.pos_ in ["ADJ", "ADV"] and feature == "negative":
            mask_tokens = [x for x in hypothesis_tokens]
            mask_tokens[i] = "<mask>"
            mask_sen = " ".join(mask_tokens)
            alters = predict_masked_tokens(mask_sen)
            #print("alters:", alters)
            for s in alters:
                if s not in neg_list:
                    hypothesis_tokens[i] = s
                    change = True
                    break
            if change == False:
                hypothesis_tokens[i] = "UNK"

    changed_hyp = " ".join(hypothesis_tokens)
    return changed_hyp

'''def detect_neg(hypothesis, feature, indexes):
    hypothesis_dep = [token.dep_ for token in p_line]
    tuples = defaultdict(list)
    for i, dep in enumerate(hypothesis_dep):
        if dep == "neg":
            tuples["negation"].append(i)

    if len(tuples) > 0:
        return ["negation"], tuples
    else:
        return [], {}'''

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

def change_ner(p_line, fe, indexes, ner_dict):
    tokens = [token.text for token in p_line]
    i_old = 0
    for i_num, i in enumerate(indexes):
        if (i_num != 0 and i != (i_old + 1)) or i_num==0:

            # give an extra placeholder in case there exist the same ner token as before.
            if len(ner_dict[fe])>2:
                alters = [x for x in list(random.sample(ner_dict[fe], 2)) if x != tokens[i]]
                tokens[i] = alters[0]
            else:
                tokens[i] = "UNK"
        else:
            tokens[i] = None
        i_old = i
    changed_hyp = " ".join([t for t in tokens if t != None])
    return changed_hyp

def change_tenses(hypothesis, feature):
    alters = [x for x in list(tenses.keys()) if x != feature]
    al = random.sample(alters,1) 
    try:
        changed_hyp = ct(hypothesis, al[0])
    except BaseException:
        changed_hyp = ct(hypothesis, al[0])

    return changed_hyp

def change_typo(hypothesis, indexes):
    hypothesis_token = word_tokenize(hypothesis)
    tuples = defaultdict(list)
    for i in indexes:
        hypothesis_token[i] = spell.correction(hypothesis_token[i])

    changed_hyp = " ".join(hypothesis_token)
    return changed_hyp

def change_pronoun(premise, hypothesis, parsed_hypothesis, indexes):
    ph = " ".join([premise, hypothesis])
    parsed_ph = nlp(ph)
    h_size = len(parsed_hypothesis)
    real_indexes = [h_size-x for x in indexes]
    atokens = [(x,parsed_ph[-x]) for x in real_indexes]
    tokens = [x.text for x in parsed_hypothesis]
    for (i, token) in atokens:
        #print(token._.coref_clusters)
        if len(token._.coref_clusters) != 0:
            c_text = str(token._.coref_clusters[0][0])
        else:
            c_text = "UNK"
        tokens[h_size-i] = c_text

    changed_hyp = " ".join(tokens)
    return changed_hyp

if __name__ == "__main__":
    premise = "I think lucy is a pretty girl."
    hypothesis = "I agree with your thought that lucy is beauty."
    #premise = "My sister has a dog."
    #hypothesis = "She loves him." 
    parsed_h = nlp(hypothesis)
    indexes = [0,3]
    #print(change_overlap(premise, hypothesis))
    #print(change_sent(hypothesis, "positive"))
    print(change_tenses(hypothesis, "present"))
    #print(change_pronoun(premise, hypothesis, parsed_h, indexes))




