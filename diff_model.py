# this file is used to prepare data, split data and check model result
import csv
import os 
from collections import defaultdict
from process_data import process_data
import argparse
import pickle

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

def diff(m1_res, m2_res, grounds):
    ids = set()
    for guid, g_l in grounds.items():
        m1_l = m1_res[guid]
        m2_l = m2_res[guid]
        if m1_l != m2_l and m2_l == g_l:
            ids.add(guid.split("-")[0])
    return ids
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
    gid_list = defaultdict(list)
    for tid, feature_list in guid_features.items():
        gid = tid.split("-")[0]
        gid_list[gid].append(tid)
    
    for gid, tids in gid_list.items():
        counter = defaultdict(int)
        for tid in tids:
            for f in guid_features[tid]:
                counter[f] += 1
        for feature, c in counter.items():
            if c < len(tids):
                feature_case_ids[feature].add(gid)
            #features_to_caseids[f].add(tid.split("-")[0])
    
    return feature_case_ids

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

def main():
    
    parser = argparse.ArgumentParser(description='manual to this script')
    
    parser.add_argument('--task', type=str, default = None)
    parser.add_argument('--log_dir', type=str, default = None)
    parser.add_argument('--train_inf', type=str, default = None)
    parser.add_argument('--train_outf', type=str, default = None)
    parser.add_argument('--test_inf', type=str, default = None)
    parser.add_argument('--test_outf', type=str, default = None)
    parser.add_argument('--lemmatize', action = "store_true")
    parser.add_argument('--model1_res_file', type=str, default = None)
    parser.add_argument('--model2_res_file', type=str, default = None)
    parser.add_argument('--augment', type=str, default = None)
    parser.add_argument('--model_name', type=str, default = None)
    parser.add_argument('--end', action = "store_true")
    
    args = parser.parse_args()
    
    task = args.task
    
    log_dir = args.log_dir
    
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
    
   
    subtest_dir = os.path.join(log_dir, args.model_name)
    if os.path.exists(subtest_dir) == False:
        os.makedirs(subtest_dir)

    lemmatize = args.lemmatize
    print("lemmatize:", lemmatize)
    model1_file = args.model1_res_file
    model2_file = args.model2_res_file
    
    model = args.model_name
    augment = args.augment

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

    print("2. step two: read model results.")

    test_ground_truth = read_file_res(test_outf)  
    
    model1_res = read_model_res(model1_file)
    model2_res = read_model_res(model2_file)
    
    diff_case_ids = diff(model1_res, model2_res, test_ground_truth)
    case_dict = read_csv(test_outf)

    write_file = os.path.join(log_dir, f"test_{model}_{augment}.csv")
    with open(write_file, "w") as f_w:
        f_writer = csv.DictWriter(f_w, fieldnames=["guid", "premise", "hypothesis", "label"])
        f_writer.writeheader()
        lines = [case_dict[x] for x in diff_case_ids]
        for question in lines:
            for line in question:
                f_writer.writerow(line)
        
    #for question in questions:
    #    for line in question:

if __name__ == "__main__":
    main()
