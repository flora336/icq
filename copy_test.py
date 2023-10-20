# this file is used to prepare data, split data and check model result
import csv
import os 
from process_data import process_data
from extract_features import extract_features
from sp_test import test_case_split
from collections import defaultdict
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
    parser.add_argument('--features', nargs='+', default = None)
    parser.add_argument('--lemmatize', type=bool, default = False)
    parser.add_argument('--model_res_file', type=str, default = None)
    parser.add_argument('--model_name', type=str, default = None)
    
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
    
    subtest_dir = os.path.join(log_dir, args.model_name)
    if os.path.exists(subtest_dir) == False:
        os.makedirs(subtest_dir)

    lemmatize = args.lemmatize
    print("lemmatize:", lemmatize)
    model_res_file = args.model_res_file

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
    test_cases = test_case_split(features, test_guid_features)
     
    print("done") 
    # Fifth step: check result
    # the rule is to test the cases
    print("4. fourth step: test case")
    test_ground_truth = read_file_res(test_outf)  

    test_model_result = read_model_res(args.model_res_file)

    original_acc, model_answers = test_acc(test_ground_truth, test_model_result) 
    
    #feature_caseids = trans_features(train_guid_features)
    test_feature_caseids, test_feature_choiceids = features_to_caseids(test_guid_features)
    result_dict = {}
    thresh = 10
    t_features = test_feature_caseids.keys()
    print("len_t_features:", len(t_features))
    for feature in t_features:

        exist_case_list = test_feature_caseids[feature]
        exist_choice_list = test_feature_choiceids[feature]
        #print(sorted(exist_choice_list))
        #exit()
        counter = 0

        right_list = []
        wrong_list = []
        #print("exist_case_list:", exist_case_list)
        if len(exist_case_list) > thresh and len(model_answers) - len(exist_case_list) > thresh:
            result_list = [model_answers[x] for x in exist_case_list]
            for tid in exist_choice_list:
                if test_ground_truth[tid] == "right":
                    right_list.append(model_answers[tid.split("-")[0]])
                else:
                    wrong_list.append(model_answers[tid.split("-")[0]])
            #right_list = [model_answers[x] for x in exist_case_list]
            #wrong_list = [model_answers[x] for x in exist_case_list if ]
            #acc = sum(result_list)/float(len(result_list))
            #result_dict[feature] = acc

            #right_list = split_dict["right"][feature] = [model_answer[x] for x in right_label] 
            #wrong_list = split_dict["wrong"][feature] = [model_answer[x] for x in wrong_label] 
            delta = (sum(result_list)*len(model_answers)-sum(model_answers.values())*len(result_list))/ \
                    (len(result_list)*(len(model_answers)-len(result_list)))
            if len(right_list) == 0 and len(wrong_list) != 0:
                right_delta = 0.0
                wrong_delta = (sum(wrong_list)*len(model_answers)-sum(model_answers.values())*len(wrong_list))/ \
                        (len(wrong_list)*(len(model_answers)-len(wrong_list)))
            elif len(wrong_list) == 0 and len(right_list) != 0: 
                right_delta = (sum(right_list)*len(model_answers)-sum(model_answers.values())*len(right_list))/ \
                        (len(right_list)*(len(model_answers)-len(right_list)))
                wrong_delta = 0.0
            else:
                right_delta = (sum(right_list)*len(model_answers)-sum(model_answers.values())*len(right_list))/ \
                        (len(right_list)*(len(model_answers)-len(right_list)))
                wrong_delta = (sum(wrong_list)*len(model_answers)-sum(model_answers.values())*len(wrong_list))/ \
                        (len(wrong_list)*(len(model_answers)-len(wrong_list)))
            result_dict[feature] = delta
            #result_dict[feature+"-"+"right"] = right_delta
            #result_dict[feature+"-"+"wrong"] = wrong_delta
            #result_right_dict[feature] = right_delta
            #result_wrong_dict[feature] = wrong_delta

    sort_features = list(sorted(result_dict.items(), key = lambda x:x[1], reverse=True))[:100]
    print(sort_features)
    print("len_result_dict:", len(result_dict))
    print("done")
   
    
    print("5. fifth step: split subtestcase")
    # the question should be right answered
    questions = read_csv(test_outf)
    print("len_questions:", len(questions))
    print("sorted questions:", sort_features)
    print("sorted questions:", result_dict)
    for i, f_s in enumerate(sort_features):
        if i < 20:
            f = f_s[0]
            #f = [0]
            #label = feature[1]

            f_name = "_".join(f.split("\t"))
            print(f"feature {f} has {len(test_feature_caseids[f.split('-')[0]])} cases and the acc score is {result_dict[f]}")
            feature_file = os.path.join(subtest_dir, f"{i}_{f_name}.csv")
            with open(feature_file, "w") as f_w:
                f_writer = csv.DictWriter(f_w, fieldnames=["guid", "premise", "hypothesis", "label"])
                f_writer.writeheader()
                ids = test_feature_caseids[f.split("-")[0]]
                lines = [questions[x] for x in ids]
                for question in lines:
                    for line in question:
                        f_writer.writerow(line)
        
    #for question in questions:
    #    for line in question:

if __name__ == "__main__":
    main()
