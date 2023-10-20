# this file is used to prepare data, split data and check model result
import csv
import os 
from process_data import process_data
from extract_features import extract_features
from sp_test import test_case_split
from collections import defaultdict
import argparse


def read_model_res(res_file):
    
    res = defaultdict()
    with open(res_file, "r") as res_file:
        #res_reader = csv.DictReader(res_file)
        for line in res_file:
            r = line.strip("\n").split("\t")
            res[r[0]] = r[1]
    
    return res

def read_file_res(test_file):
    
    res = defaultdict()
    with open(test_file, "w") as res_file:
        res_reader = csv.DictReader(res_file)
        for line in res_reader:
            guid = line["guid"]
            label = line["label"]
            res[guid] = label
    
    return res

def test_acc(ground_truth, model_res):
    acc = 0.0
    right = 0
    model_answers = defaultdict()
    
    assert len(ground_truth) == len(model_res)
    
    for guid, g_l in ground_truth.items():
        m_l = model_res[guid]
        if g_l == m_l:
            right += 1
            model_answers[guid] = True
        else:
            model_answers[guid] = False

    acc = right/float(len(ground_truth))
    
    return acc, model_answers

def features_to_caseids(guid_features):
    
    feature_case_ids = defaultdict(list)
    for tid, feature_list in guid_features.items():
        for f in feature_list:
            features_to_caseids[f].append(tid)
    
    return features_to_caseids

def main():
    
    parser = argparse.ArgumentParser(description='manual to this script')
    
    parser.add_argument('--task', type=str, default = None)
    parser.add_argument('--log_dir', type=str, default = None)
    parser.add_argument('--train_inf', type=str, default = None)
    parser.add_argument('--train_outf', type=str, default = None)
    parser.add_argument('--test_inf', type=str, default = None)
    parser.add_argument('--test_outf', type=str, default = None)
    parser.add_argument('--features', type=list, default = None)
    parser.add_argument('--lemmatize', type=bool, default = None)
    parser.add_argument('--model_res_file', type=list, default = None)
    
    args = parser.parse_args()

    # First step: preprocessing data 
    # file type mains "train", "dev" or "test"
    print("1. step one: process data.")
    process_data(task, "train", train_outf, train_inf, lemmatize=lemmatize)
    process_data(task, "test", test_outf, test_inf, lemmatize=lemmatize)
    print("done") 
    # Second step: extract features

    print("2. step two: extract_feature.")
    train_guid_features = extract_features(infile, outfile, log_dir, features, part="hypothesis")
    test_guid_features = extract_features(infile, outfile, log_dir, features, part="hypothesis")
    print("done") 
    
    # Third step: cal feature score:
    
    # Forth step: split cases
    print("3. step three: split case")
    test_cases = test_case_split(features, guid_features)
     
    print("done") 
    # Fifth step: check result
    # the rule is to test the cases
    print("4. fourth step: test case")
    test_ground_truth = read_file_res(test_inf)  

    test_model_result = read_model_res(args.model_res_file)

    original_acc, model_answers = test_acc(test_ground_truth, test_model_result) 
    
    #feature_caseids = trans_features(train_guid_features)
    test_feature_caseids = features_to_caseids(test_guid_features)
    result_dict = {}
    for feature in features:

        exist_case_list = test_feature_caseids[feature]
        result_list = [model_answer[x] for x in exist_case_list]
        acc = sum(result_list)/float(len(result_list))
        result_dict[feature] = acc
        print(sorted(result_dict, lambda x:x[1], reverse=True))

if __name__ == "__main__":
    main()




