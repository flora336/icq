# this file is used to merge data for model evaluation
# this file is also used for split features to check how many data can be used for flip testing
# this file is also used to split features to check the flip rate for each cue.

# the guid is used to data merge and split merged guid have the structure: feature_name@@guid@@original/deleted
import csv
import os 
import numpy as np
from process_data import process_data
from extract_features import extract_features
from sp_test import case_split
from feature_score import get_feature_score
from collections import defaultdict
from change_features import change_delete
import argparse
import pickle
import math
import spacy
from nltk.tokenize import word_tokenize
nlp = spacy.load('en_core_web_sm')

def features_to_caseids(guid_features):
    
    feature_case_ids = defaultdict(set)
    feature_choice_ids = defaultdict(set)
    gid_list = defaultdict(list)
    for tid, feature_list in guid_features.items():
        gid = tid.split("-")[0]
        gid_list[gid].append(tid)
    
    for gid, tids in gid_list.items():
        counter = defaultdict(int)
        for tid in tids:
            for f in guid_features[tid]:
                counter[f] += 1
                feature_choice_ids[f].add(tid)
        for feature, c in counter.items():
            if c < len(tids):
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

 
def read_labeled_file(file_name):
    cases = defaultdict(list)
    chooses = defaultdict()
    # sig_num_
    feature = "_".join(file_name.split("_")[2:])
    with open(file_name, "r") as f:
        f_reader = csv.DictReader(f)
        for line in f_reader:
            case_id = line["guid"].split("-")[0]
            cases[case_id].append(line["significant_or_not"])
    for cid in cases:
        choose_list = []
        c_list = cases[cid]
        for c in c_list:
            if "y" in c.lower():
                choose_list.append(False)
            elif "n" in c.lower():
                choose_list.append(True)
            else:
                choose_list.append(None)
        if False in choose_list:
            chooses[cid] = False
        elif True in choose_list:
            chooses[cid] = True
        else:
            chooses[cid] = False

    return chooses

def read_exist(file_name):
    exists = defaultdict(defaultdict)
    with open(file_name, "r") as f:
        f_reader = csv.DictReader(f)
        for line in f_reader:
            uid = line["guid"].split("@@")
            feature = uid[0]
            guid = uid[1]
            exists[feature][guid] = line["exist"]

    return exists


def read_original_file(file_name):
    cases = defaultdict(list)
    labels = defaultdict()
    with open(file_name, "r") as f:
        f_reader = csv.DictReader(f)
        for line in f_reader:
            #guid = "@@".join([feature, line[guid], file_type])
            cid = line["guid"].split("-")[0]
            cases[cid].append(line)
            labels[line["guid"]] = line["label"]
    return cases, labels
def read_delete_file(file_name):
    cases = defaultdict(list)
    labels = defaultdict()
    with open(file_name, "r") as f:
        f_reader = csv.DictReader(f)
        for line in f_reader:
            guid = line["guid"].split("@@")[1]
            cid = guid.split("-")[0]
            cases[cid].append(line)
            labels[line["guid"]] = line["label"]
    return cases, labels
            
def read_result(file_name):
    res = defaultdict()
    with open(file_name, "r") as f:
        count = 0
        for line in f:
            if count != 0:
                r = line.strip("\n").split("\t")
                if len(r) == 2:
                    res[r[0]] = r[1]
            count += 1
    return res

def check_data(original_file, all_deleted_file, prompt_file, labeled_dir, original_res_file, deleted_res_file, out_file):
    original_cases, ground_labels = read_original_file(original_file)
    _, delete_ground_labels = read_delete_file(all_deleted_file)
    exist_res = read_exist(all_deleted_file)
    original_pred = read_result(original_res_file) 
    deleted_pred = read_result(deleted_res_file) 
    prompt_pred = read_result(prompt_file) 
    deleted_features_pred = defaultdict(defaultdict) # "right" or "wrong" and don't know which one exist feature
    
    all_cases = defaultdict(int)
    alter_cases = defaultdict(int)

    deleted_flip = defaultdict(int)
    
    pred_res = defaultdict(defaultdict)
    original_res = defaultdict()
    
    for unique_id, pred in deleted_pred.items():
        
        feature = unique_id.split("@@")[0]
        guid = unique_id.split("@@")[1]
        cid = guid.split("-")[0]
        #print(unique_id, )
        #print(exist_res[feature])
        exist = exist_res[feature][guid]
        if exist == "yes":
            deleted_features_pred[feature][guid] = pred
        if pred == delete_ground_labels[unique_id]:
            pred_res[feature][cid] = True
        else:
            pred_res[feature][cid] = False
    print("deleted_pred:", deleted_features_pred)
    print("what:", pred_res["word_what.csv"]) 
    
    for uid, pred in original_pred.items():
        cid = uid.split("-")[0]
        if pred == ground_labels[uid]:
            original_res[cid] = True
        else:
            original_res[cid] = False
    if os.path.exists(labeled_dir):
        labeled_files = os.listdir(labeled_dir)

    if os.path.exists(deleted_dir):
        deleted_files = os.listdir(deleted_dir)

    with open(out_file, "w") as f:
        f_w = csv.DictWriter(f, fieldnames=["feature", "all_cases", "alter_cases", "all_acc", "no_change_acc", "change_acc", "flip_rate", "prompt_rate", "sig_right_flip_rate", "sig_wrong_flip_rate", "right_rate", "delete_right_rate"])
        f_w.writeheader()
        for labeled_name in labeled_files:
            w = {}
            l_f = os.path.join(labeled_dir, labeled_name)
            choose_res = read_labeled_file(l_f)
            print("choose_res:", choose_res)
            feature = "_".join(labeled_name.split("_")[1:])
            rank = labeled_name.split("_")[0]
            all_cases = len(choose_res)
            alter_cases = 0
            all_acc_list = []
            no_change_acc_list = []
            change_acc_list = []
            change_pred_list = []
            no_change_pred_list = []
            flip_list = []
            sig_original_pred_list = []
            sig_deleted_pred_list = []
            sig_right_flip = 0
            sig_wrong_flip = 0
            sig_flip = 0 
            for cid, res in choose_res.items():
                print("res:", cid, res)
                all_acc_list.append(original_res[cid])
                if res:
                    alter_cases += 1
                    no_change_acc_list.append(original_res[cid])
                    #print(pred_res[feature]) 
                    #print(choose_res)
                    #print(feature)
                    change_acc_list.append(pred_res[feature][cid])
                    for x in original_cases[cid]:
                        #gid = x["guid"].split("@@")[1]
                        gid = x["guid"]
                        #print("gid:", gid)
                        if gid in exist_res[feature]:
                            #print(exist_res[feature])
                            #print(exist_res[feature][gid])
                            if exist_res[feature][gid] == "yes":
                                print("yes_gid:", gid)
                                sig_deleted_pred_list.append(deleted_features_pred[feature][gid])
                                sig_original_pred_list.append(original_pred[gid])

                    #no_change_pred_list.append(original_features_pred[cid])
                    #change_pred_list.append(deleted_features_pred[feature][cid])
                #else:
                #    print(choose_res)
            for i, x in enumerate(sig_original_pred_list):
                if sig_deleted_pred_list[i] != x:
                    sig_flip += 1
                    if x == "right":
                        sig_right_flip += 1
                    else:
                        sig_wrong_flip += 1
            print("sig_right_flip", sig_right_flip)
            print("sig_wrong_flip", sig_wrong_flip)
            print("sig_flip", sig_flip)
            print("sum(np.array(no_change_acc_list) != np.array(change_acc_list))", sum(np.array(no_change_acc_list) != np.array(change_acc_list)))
            print("len(no_change_acc_list):", no_change_acc_list)
            print("len(change_acc_list):", change_acc_list)
            #print(change_pred_list)
            #exit()
            if len(all_acc_list) != 0 and alter_cases >0 :
                w["feature"] = feature
                w["all_cases"] = all_cases
                w["alter_cases"] = alter_cases
                w["all_acc"] = sum(all_acc_list)/float(len(all_acc_list))
                w["no_change_acc"] = sum(no_change_acc_list)/float(len(no_change_acc_list))
                w["change_acc"] = sum(change_acc_list)/float(len(change_acc_list))
                w["flip_rate"] = sum(np.array(no_change_acc_list) != np.array(change_acc_list))/float(len(no_change_acc_list))
                w["sig_right_flip_rate"] = float(sig_right_flip)/sum(np.array(no_change_acc_list) != np.array(change_acc_list))
                w["sig_wrong_flip_rate"] = float(sig_wrong_flip)/sum(np.array(no_change_acc_list) != np.array(change_acc_list))
                w["right_rate"] = len([x for x in sig_original_pred_list if x == "right"])/float(len(sig_original_pred_list))
                # right rate is the case to predict "right" with a specific feature on original test case
                w["delete_right_rate"] = len([x for x in sig_deleted_pred_list if x == "right"])/float(len(sig_original_pred_list))
                # delete right rate is the case to predict "right" with a specific feature on deleted test case
                f = feature.split(".csv")[0]
                if f in prompt_pred:
                    w["prompt_rate"] = prompt_pred[feature.split(".csv")[0]]
                else:
                    w["prompt_rate"] = ""

                f_w.writerow(w)
            else:
                print(f"{feature} is not exist")

       
def main2():
    
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
        train_guid_features = extract_features(train_outf, trainf_feature, log_dir, features, part="hypothesis")

    if is_testf_features:
        test_guid_features = pickle.load(open(testf_feature, "rb"))
    else:
        test_guid_features = extract_features(test_outf, testf_feature, log_dir, features, part="hypothesis")
    print("done") 
    
    # Third step: cal feature score:
    
    # Forth step: split cases
    print("3. step three: split case")
    train_cases = case_split(features, test_guid_features)
    test_cases = case_split(features, test_guid_features)
     
    print("done") 
    # Fifth step: check result
    # the rule is to test the cases
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
        dataset_score_dicts[score_type] = get_feature_score(train_guid_features, train_ground_truth, score_type=score_type, device_n=2)
        with open(out_cue_file, "w") as c_w:
            for fe, score in dataset_score_dicts[score_type].items():
                c_w.write(f"{fe}\t{score}\n")
    
    print("6. fifth step: split subtestcase")
    # the question should be right answered
    train_questions = read_csv(train_outf)
    test_questions = read_csv(test_outf)
    thresh = 10 
    train_features = [x for x, x_list in train_feature_caseids.items() if len(x_list) > thresh ]
    test_features = [x for x, x_list in test_feature_caseids.items() if len(x_list) > thresh ]
    features  = list(set(train_features).intersection(set(test_features)))
    #print(len(features), features)
    if args.feature_split == True or args.feature_split == "True":
        write_feature_cases(features, test_feature_caseids, test_guid_features, test_questions, test_subdir) 
        write_feature_cases(features, train_feature_caseids, train_guid_features, train_questions, train_subdir) 
    if args.change_type in ["delete", "substitute"]:
        write_changed_cases(features, test_feature_caseids, test_guid_features, test_questions, test_changedir, change_type=args.change_type)
        write_changed_cases(features, train_feature_caseids, train_guid_features, train_questions, train_changedir, change_type=args.change_type)
    else:
        print(f"the value for --change should be in 'delete' or 'substitute.' ")


   #for question in questions:
    #    for line in question:


if __name__ == "__main__":
    task = "roc"
    deleted_dir = f"/home/shanshan/icq/prob/log/{task}/dataf/delete/test"
    all_deleted_file = f"/home/shanshan/icq/prob/data/{task}/delete.csv"
    labeled_dir = f"/home/shanshan/icq/prob/log/{task}/dataf/significance/"
    original_file = f"/home/shanshan/icq/prob/data/{task}/test.csv"
    #for m in ["bert", "xlnet", "roberta"]:
    for m in ["xlnet", "roberta"]:
        original_res_file = f"/home/shanshan/generate_noise/transformers/output_dir/{task.upper()}/original_1_{m}_base/test_results_original.txt"
        deleted_res_file = f"/home/shanshan/generate_noise/transformers/output_dir/{task.upper()}/original_1_{m}_base/test_results_delete.txt"
        prompt_file = f"/home/shanshan/icq/prob/log/{task}/{m}/prompt/flip_rate.txt"
        out_file = f"/home/shanshan/icq/prob/result/{task}_{m}_res.csv"
        check_data(original_file, 
            all_deleted_file, 
            prompt_file, 
            labeled_dir, 
            original_res_file, 
            deleted_res_file, 
            out_file)
