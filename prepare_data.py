# this file is used to prepare data, split data and check model result
import csv
import os 
from process_data import process_data
from extract_features import extract_features
from sp_test import case_split
from feature_score import get_feature_score, cal_mixed_feature_score
from collections import defaultdict
from change_features import change_delete
from merge import merge_data
import argparse
import pickle
import math
import spacy
from nltk.tokenize import word_tokenize
nlp = spacy.load('en')

def read_model_res(res_file):
    
    res = defaultdict()
    with open(res_file, "r") as f:
        #res_reader = csv.DictReader(res_file)
        for line in f:
            r = line.strip("\n").split("\t")
            if "-" in r[0]:
                res[r[0]] = r[1]
    
    return res

def read_file_res(test_file):
    
    res = defaultdict()
    with open(test_file, "r") as res_file:
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
    print("grounds:", len(ground_truth))
    print("model_res:", len(model_res))
    assert len(ground_truth) == len(model_res)
    
    for guid, g_l in ground_truth.items():
        m_l = model_res[guid]
        gid = guid.split("-")[0]
        if g_l == m_l:
            right += 1
            model_answers[gid] = True
        else:
            model_answers[gid] = False

    acc = right/float(len(ground_truth))
    
    return acc, model_answers

def features_to_caseids(guid_features):
    
    feature_case_ids = defaultdict(set)
    feature_choice_ids = defaultdict(set)
    gid_list = defaultdict(list)
    for tid, feature_list in guid_features.items():
        gid = tid.split("-")[0]
        gid_list[gid].append(tid)
    
    for gid, tids in gid_list.items():
        counter = defaultdict(set)
        for tid in tids:
            for f in guid_features[tid]:
                counter[f].add(tid)
                feature_choice_ids[f].add(tid)
        for feature, c in counter.items():
            if len(c) < len(tids):
                feature_case_ids[feature].add(gid)
                #feature_choice_ids[feature].update(tids)
            #features_to_caseids[f].add(tid.split("-")[0])
    
    return feature_case_ids, feature_choice_ids

def file_exist(feature_file):

    if os.path.exists(feature_file):
        if os.path.getsize(feature_file):
            return True
        else:
            print(f"feature file: {feature_file} is empty.")
            return False
    else:
        print(f"feature file: {feature_file} is not exist.")
        return False

def read_csv(csv_file):
    
    csv_dict = defaultdict(list)
    with open(csv_file, "r") as f:
        f_r = csv.DictReader(f)
        for line in f_r:
            gid = line["guid"].split("-")[0]
            csv_dict[gid].append(line)

    
    return csv_dict

def write_feature_cases(features, f_to_caseids, guid_features, questions, subtest_dir):
    for i, f in enumerate(features):
        f_name = "_".join(f.split("\t"))
        print(f"feature {f} has {len(f_to_caseids[f])} cases.")
        feature_file = os.path.join(subtest_dir, f"{i}_{f_name}.csv")
        with open(feature_file, "w") as f_w:
            f_writer = csv.DictWriter(f_w, fieldnames=["guid", "premise", "hypothesis", "label", "significant_or_not", "change_to"])
            f_writer.writeheader()
            ids = f_to_caseids[f]
            lines = [questions[x] for x in ids]
            for question in lines:
                w_dict = {}
                for line in question:
                    w_dict["guid"] = line["guid"]
                    w_dict["label"] = line["label"]
                    w_dict["premise"] = line["premise"]
                    #print(guid_features[line["guid"]])
                    #print(line)
                    hypothesis = line["hypothesis"]
                    if f in guid_features[line["guid"]]:
                        indexes = guid_features[line["guid"]][f]
                        if f.split("\t")[0] in ["ner", "negation", "word", "tense", "pronoun", "overlap", "sentiment"]:
                            h_tokens = [x.text for x in nlp(hypothesis)]
                        else:
                            h_tokens = word_tokenize(hypothesis)
                        for ind in indexes:
                            h_tokens[ind] = "*"+h_tokens[ind]
                        w_dict["hypothesis"] = " ".join(h_tokens) 
                        f_writer.writerow(w_dict)
                    else:
                        f_writer.writerow(line)


def write_changed_cases(features, f_to_caseids, guid_features, questions, subtest_dir, change_type, ner_dict):
    print("ner_dict:", ner_dict)
    for i, f in enumerate(features):
        f_name = "_".join(f.split("\t"))
        feature_file = os.path.join(subtest_dir, f"{i}_{f_name}.csv")
        lines = change_delete(questions, f_to_caseids, guid_features, f, change_type=change_type, ner_dict=ner_dict)
        with open(feature_file, "w") as f_w:
            f_writer = csv.DictWriter(f_w, fieldnames=["guid", "premise", "hypothesis", "chypothesis", "label", "exist"])
            f_writer.writeheader()
            for line in lines:
                w = {}
                guid = line["guid"]
                w["guid"] = guid 
                w["label"] = line["label"]
                w["premise"] = line["premise"]
                w["exist"] = line["exist"]
                w["chypothesis"] = line["hypothesis"]
                for x in questions[guid.split("-")[0]]:
                    if x["guid"] == guid:
                        w["hypothesis"] = x["hypothesis"] 

                f_writer.writerow(w)
        
def main():
    
    parser = argparse.ArgumentParser(description='manual to this script')
    
    parser.add_argument('--task', type=str, default = None)
    parser.add_argument('--log_dir', type=str, default = None)
    parser.add_argument('--train_inf', type=str, default = None)
    parser.add_argument('--train_outf', type=str, default = None)
    parser.add_argument('--train_features_file', type=str, default = None)
    parser.add_argument('--test_features_file', type=str, default = None)
    parser.add_argument('--test_inf', type=str, default = None)
    parser.add_argument('--test_outf', type=str, default = None)
    parser.add_argument('--change_type', type=str, default = None)
    parser.add_argument('--merge_changed_file', type=str, default=None)
    parser.add_argument('--features', nargs='+', default = None)
    parser.add_argument('--cueness_types', nargs='+', default = None)
    parser.add_argument('--feature_split', type=str, default=None)
    #parser.add_argument('--lemmatize', type=bool, default = False)
    
    args = parser.parse_args()
    
    task = args.task
    
    log_dir = args.log_dir

    
    features = args.features

    print("features:", features)
    #exit()
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    
    train_inf = args.train_inf
    train_outf = args.train_outf

    test_inf = args.test_inf
    test_outf = args.test_outf



    train_out_dir = os.path.abspath(os.path.dirname(train_outf))
    test_out_dir = os.path.abspath(os.path.dirname(test_outf))
    if os.path.exists(train_out_dir) == False:
        os.makedirs(train_out_dir)

    if os.path.exists(test_out_dir) == False:
        os.makedirs(test_out_dir)
    
    #change_dir = os.path.join(log_dir, args.model_name, args.change_type)
    change_dir = os.path.join(log_dir, "dataf", args.change_type)
    subdata_dir = os.path.join(log_dir, "dataf")
    
    cueness_dir = os.path.join(log_dir, "cueness", "all")

    train_subdir = os.path.join(subdata_dir, "train")
    test_subdir = os.path.join(subdata_dir, "test")
    
    train_changedir = os.path.join(change_dir, "train")
    test_changedir = os.path.join(change_dir, "test")
    
    if os.path.exists(subdata_dir) == False:
        os.makedirs(subdata_dir)
    if os.path.exists(change_dir) == False:
        os.makedirs(change_dir)
    
    if os.path.exists(cueness_dir) == False:
        os.makedirs(cueness_dir)
    
    if os.path.exists(train_subdir) == False:
        os.makedirs(train_subdir)
    if os.path.exists(test_subdir) == False:
        os.makedirs(test_subdir)
    
    if os.path.exists(train_changedir) == False:
        os.makedirs(train_changedir)
    if os.path.exists(test_changedir) == False:
        os.makedirs(test_changedir)
    lemmatize = False
    #lemmatize = args.lemmatize
    #print("lemmatize:", lemmatize)

    # First step: preprocessing data 
    # file type mains "train", "dev" or "test"
    print("1. step one: process data.")
    is_train_outf = file_exist(train_outf)
    is_test_outf = file_exist(test_outf)
    if is_train_outf == False:
        process_data(task, "train", train_outf, train_inf, lemmatize=lemmatize)
    if is_test_outf == False:
        process_data(task, "test", test_outf, test_inf, lemmatize=lemmatize)
    print("done") 
    # Second step: extract features

    print("2. step two: extract_feature.")
    trainf_feature = args.train_features_file
    testf_feature = args.test_features_file
    is_trainf_features = file_exist(trainf_feature)
    is_testf_features = file_exist(testf_feature)
    
    if is_trainf_features:
        train_guid_features = pickle.load(open(trainf_feature, "rb"))
        #print("train_guid_features", train_guid_features)
        #exit()
    else:
        train_guid_features = extract_features(train_outf, trainf_feature, log_dir, features, part="hypothesis", file_type = "train")

    if is_testf_features:
        test_guid_features = pickle.load(open(testf_feature, "rb"))
    else:
        test_guid_features = extract_features(test_outf, testf_feature, log_dir, features, part="hypothesis", file_type = "test")
    print("done") 
    
    ner_train_dict = pickle.load(open(os.path.join(log_dir, "train_ner_dict"), "rb"))
    ner_test_dict = pickle.load(open(os.path.join(log_dir, "test_ner_dict"), "rb"))
    
    ner_dict = defaultdict(set)
    for ner, n_set in ner_train_dict.items():
        if ner in ner_test_dict:
            ner_dict[ner] = n_set.union(ner_test_dict[ner])
        else:
            ner_dict[ner] = n_set
    # Third step: cal feature score:
    
    # Forth step: split cases
    #print("3. step three: split case")
    train_cases = case_split(features, test_guid_features)
    test_cases = case_split(features, test_guid_features)
     
    print("done") 
    print("4. fourth step: test case")
    test_ground_truth = read_file_res(test_outf)  
    train_ground_truth = read_file_res(train_outf)  
    
    train_feature_caseids, train_feature_choiceids = features_to_caseids(train_guid_features)
    test_feature_caseids, test_feature_choiceids = features_to_caseids(test_guid_features)
    
    tr_features = train_feature_caseids.keys()
    t_features = test_feature_caseids.keys()
    
    print("5. fifth step: calculate cueness score")
    # cuenesscan only be calculated from training data
    dataset_score_dicts = defaultdict(defaultdict)
    for score_type in args.cueness_types:
        out_cue_file = os.path.join(cueness_dir, score_type+".txt")
        word_label_score, word_label_freq = get_feature_score(train_guid_features, train_ground_truth, score_type=score_type, device_n=2)
        dataset_score_dicts[score_type], distribution = cal_mixed_feature_score(word_label_score, word_label_freq) 
        with open(out_cue_file, "w") as c_w:
            c_writer = csv.DictWriter(c_w, fieldnames=["cue", "score", "distribution"])
            c_writer.writeheader()
            for fe, score in dataset_score_dicts[score_type].items():
                w = {}
                w["cue"] = fe
                w["score"] = score
                w["distribution"] = distribution[fe]
                c_writer.writerow(w)
    print("6. fifth step: split subtestcase")
    # the question should be right answered
    train_questions = read_csv(train_outf)
    test_questions = read_csv(test_outf)
    thresh = 10 
    train_features = [x for x, x_list in train_feature_caseids.items() if len(x_list) > thresh ]
    test_features = [x for x, x_list in test_feature_caseids.items() if len(x_list) > thresh ]
    features  = list(set(train_features).intersection(set(test_features)))
    if args.feature_split == True or args.feature_split == "True":
        write_feature_cases(features, test_feature_caseids, test_guid_features, test_questions, test_subdir) 
        write_feature_cases(features, train_feature_caseids, train_guid_features, train_questions, train_subdir) 
    if args.change_type in ["delete", "substitute"]:
        write_changed_cases(features, test_feature_caseids, test_guid_features, test_questions, test_changedir, change_type=args.change_type, ner_dict=ner_dict)
        write_changed_cases(features, train_feature_caseids, train_guid_features, train_questions, train_changedir, change_type=args.change_type, ner_dict=ner_dict)
    else:
        print(f"the value for --change should be in 'delete' or 'substitute.' ")


    print("7. seventh step: merge changed data to a file")

    merge_data(test_changedir, args.merge_changed_file)

   #for question in questions:
    #    for line in question:

if __name__ == "__main__":
    main()
