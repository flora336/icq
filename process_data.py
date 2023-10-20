#encoding=utf-8
import sys
import csv
import json
import string
import os
import nltk
csv.field_size_limit(sys.maxsize)
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import multiprocessing
import math
import time
from pathlib import Path
# 1. Init Lemmatizer
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

#file_w_name = "./train_original.csv"
#file_w_name = "./original_dev_matched.csv"

'''with open(file_name,'r') as f, open(file_w_name, 'w') as f_w:
    header = ["guid", "premise", "hypothesis", "label"]
    f_reader = csv.reader(f, delimiter="\t")
    f_writer = csv.DictWriter(f_w, header)
    f_writer.writeheader()
    index = 0
  
    # Translation tables to remove parentheses and punctuation from
    # strings.
    parentheses_table = str.maketrans({"(": None, ")": None})
    punct_table = str.maketrans({key: " "
                               for key in string.punctuation})'''
def get_values(line, line_num, task_name, file_type, lemma=False):
    value_list = []
    premise = []
    hypothesis_list = []
    if task_name in [ 'snli','mnli_matched','mnli_mismatched']:
        line_dic = json.loads(line)
        ids = line_num
        premise = [line_dic["sentence1"]]
        hypothesis = line_dic["sentence2"]
        label = line_dic["gold_label"]
        #value_list.append(((ids, premise, [hypothesis], label), lemma))
        value_list.append(((ids, premise, [hypothesis], [label]), lemma))
    elif task_name in [ 'anli']:
        label_map={"e":"entailment", "n":"neutral", "c":"contradiction"}
        line_dic = json.loads(line)
        ids = line_num
        premise = [line_dic["context"]]
        hypothesis = line_dic["hypothesis"]
        label = label_map[line_dic["label"]]
        print("label:", label)
        value_list.append(((ids, premise, [hypothesis], [label]), lemma))
    elif task_name == "common_qa":
        line_dic = json.loads(line)
        ids = line_num
        label = line_dic["answerKey"]
        hypothesis_list = line_dic["question"]["choices"]
        premise = [line_dic["question"]["stem"]]
        for i, h_dic in enumerate(hypothesis_list):
            if h_dic['label'] == label:
                value_list.append(((f"{ids}-{i}", premise, h_dic["text"], "right"), lemma))
            else:
                value_list.append(((f"{ids}-{i}", premise, h_dic["text"], "wrong"), lemma))
    elif task_name == 'qnli':

        line_dic = line
        ids = line_num
        premise = [line_dic[1]]
        hypothesis = line_dic[2]
        try:
            label = line_dic[3]
        except:
            label = ''
        value_list.append(((ids, premise, [hypothesis], label),lemma))
    elif task_name == 'roc':
        line_dict = line
        ids = line_num
        premise = [line_dict["InputSentence1"], line_dict['InputSentence2'],
                line_dict['InputSentence3'], line_dict['InputSentence4']]
        hypothesis1 = line_dict["RandomFifthSentenceQuiz1"]
        #print('hypothesis1:', hypothesis1)
        hypothesis2 = line_dict["RandomFifthSentenceQuiz2"]
        #print('hypothesis2:', hypothesis2)
        ending = line_dict['AnswerRightEnding']
        if ending == '2':
            end1 = 'wrong'
            end2 = 'right'
        elif ending == '1':
            end1 = 'right'
            end2 = 'wrong'
        #value_list.append(((ids+"-1", premise, hypothesis1, end1), lemma))
        #value_list.append(((ids+"-2", premise, hypothesis2, end2), lemma))
        value_list.append(((ids, premise, [hypothesis1, hypothesis2], [end1, end2]), lemma))
    elif task_name in ["reclor", "race"]:
        hypothesis_list = []
        line_dict = line
        #ids = line_dict["id"]
        ids = line_num
        premise = [line_dict["article"], line_dict["question"]]
        hypothesis_list.append(line_dict["option0"])
        hypothesis_list.append(line_dict["option1"])
        hypothesis_list.append(line_dict["option2"])
        hypothesis_list.append(line_dict["option3"])
        label = line_dict['answer']
        label_list = ['wrong', 'wrong', 'wrong', 'wrong']
        label_list[int(label)] = "right"
        #for i, h in enumerate(hypothesis_list):
        #    value_list.append(((f"{ids}-{i}", premise, h, label_list[i]), lemma))
        value_list.append(((ids, premise, hypothesis_list, label_list), lemma))

    elif task_name in ["ubuntu"]:
        line_dict = line
        ids = line_dict["id"]
        premise = [line_dict["premise"]]
        hypothesis_list = eval(line_dict["hypothesis_list"])
        label = line_dict['label']
        label_list = ['wrong'] * 100
        label_list[int(label)] = "right"
        value_list.append((ids, premise, hypothesis_list, label_list))
        
    elif task_name == "copa":
        line_dict = line
        #ids = line_dict["guid"]
        ids = line_num
        premise = [line_dict["premise"]]
        hypothesis1 = line_dict["option1"]
        print('hypothesis1:', hypothesis1)
        hypothesis2 = line_dict["option2"]
        print('hypothesis2:', hypothesis2)
        ending = line_dict['label']
        if ending == '1' and file_type == "train":
            end1 = 'wrong'
            end2 = 'right'
        elif ending == '0' and file_type == "train":
            end1 = 'right'
            end2 = 'wrong'
        elif ending == '1':
            end1 = 'right'
            end2 = 'wrong'
        elif ending == "2":
            end1 = 'wrong'
            end2 = 'right'

        print("ending:", ending)
        #value_list.append(((ids+"-1", premise, hypothesis1, end1), lemma))
        #value_list.append(((ids+"-2", premise, hypothesis2, end2), lemma))
        value_list.append(((ids, premise, [hypothesis1, hypothesis2], [end1, end2]), lemma))
    elif task_name == 'swag':
        hypothesis_list = []
        line_dict = line
        ids = line_dict["video-id"]
        premise = [line_dict["sent1"], line_dict["sent2"]]
        hypothesis_list.append(line_dict["ending0"])
        hypothesis_list.append(line_dict["ending1"])
        hypothesis_list.append(line_dict["ending2"])
        hypothesis_list.append(line_dict["ending3"])
        ending = line_dict['label']
        label_list = ['wrong', 'wrong', 'wrong', 'wrong']
        label_list[int(ending)] = "right"
        for i, h in enumerate(hypothesis_list):
            value_list.append((ids, premise, h, label_list[i]))
    elif task_name in ["arct", "arct2"]:
        hypothesis_list = []
        line_dict = line
        ids = line_dict[0]
        premise = [line_dict[4], line_dict[5]]
        label_list = ["wrong", "wrong"]
        label = line_dict[3]
        label_list[int(label)] = "right"
        hypothesis_list.append(line_dict[1])
        hypothesis_list.append(line_dict[2])
        #value_list.append((ids, premise, hypothesis_list, label_list))
        value_list.append(((ids, premise, hypothesis_list, label_list), lemma))
    return value_list

def transformat(task_name, file_type, file_name, file_w_name, lemma, end):
    path = file_w_name[0:file_w_name.rfind("/")]
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(file_name, newline='\n') as f, open(file_w_name, 'w') as f_w:
        header = ["guid", "premise", "hypothesis", "label"]
        f_writer = csv.DictWriter(f_w, header)
        f_writer.writeheader()
        if task_name in ['snli', 'mnli_matched', 'mnli_mismatched', 'common_qa', "anli"]:
            f_reader = f
        elif task_name in ['qnli', 'arct', 'arct2']:
            #f_reader = csv.DictReader(f)
            f_reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            f_reader.__next__()
        elif task_name in ['roc','copa','swag', "race", "reclor", "ubuntu"]:
            f_reader = csv.DictReader(f)
        
        inputs = []
        ids = []
        for i,line in enumerate(f_reader):
            #line_dic = json.loads(line)
            #line_dic = line
            write_dic = {}
            print(line)
            #line = line.strip().split("\t")
            # Ignore sentences that have no gold label.
            #if line_dic["gold_label"] == "-":
            #    print(line_dic)
            #    continue
            value_list = get_values(line, str(i), task_name, file_type, lemma)
            inputs.extend(value_list)
        print("type:", type(inputs))
        inputs = [[x] for x in inputs]
        
        cores = 5
        pool = multiprocessing.Pool(processes=cores)
        start_time = time.time()
        lemma_list = pool.starmap(lemma_unit, inputs)
        pool.close()
        pool.join()
        for val_list in lemma_list:
            for t in val_list:
                if t != None:
                    print("t",t)
                    write_dic["guid"] = t[0]
                    if end:
                        write_dic["premise"] = ""
                    else:
                        write_dic["premise"] = '\t\t'.join(t[1])
                    write_dic["hypothesis"] = t[2]
                    write_dic["label"] = t[3]
                    f_writer.writerow(write_dic) 

def lemma_unit(value_list):
    parentheses_table = str.maketrans({"(": None, ")": None})
    ids, premise_list, hypothesis_list, label_list = value_list[0]
    print("value_list", value_list)
    #exit()
    lemma = value_list[1]
    print("lemma:", lemma)
    if '-' in label_list:
        return []
    print('hypothesis:', hypothesis_list)
    ## Remove '(' and ')' from the premises and hypotheses.
    #premise = premise.translate(parentheses_table)
    #hypothesis = hypothesis.translate(parentheses_table)    
    premise_tokens = []
    hypothesis_tokens = []
    value_list = []
    premise_list = [premise.translate(parentheses_table) for premise in premise_list]
    hypothesis_list = [hypothesis.translate(parentheses_table) for hypothesis in hypothesis_list]
    if lemma == True:
        for p in premise_list:
            print("p:",p)
            premise_tokens.append(' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(p)]))
        for h in hypothesis_list:
            print("h:",h)
            hypothesis_tokens.append(' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(h)]))
    else:
        for p in premise_list:
            print("p:",p)
            #premise_tokens.append(' '.join([w for w in nltk.word_tokenize(p)]))
            premise_tokens.append(p)
        for h in hypothesis_list:
            print("h:",h)
            #hypothesis_tokens.append(' '.join([w for w in nltk.word_tokenize(h)]))
            hypothesis_tokens.append(h)
    for i, h_tokens in enumerate(hypothesis_tokens):
        value_list.append([ids+"-"+str(i+1), premise_tokens, h_tokens, label_list[i]])
    return value_list

input_files_dict = {
        "snli":{"train": '/home/shanshan/generate_noise/ESIM/data/data_source/snli_1.0/snli_1.0_train.jsonl', \
                "dev": '/home/shanshan/generate_noise/ESIM/data/data_source/snli_1.0/snli_1.0_dev.jsonl', \
                "test": '/home/shanshan/generate_noise/ESIM/data/data_source/snli_1.0/snli_1.0_test.jsonl'}, \
        "anli":{"train": '../data_source/anli_v0.1/R1/test.jsonl', \
                "dev": '../data_source/anli_v0.1/R2/test.jsonl', \
                "test": '../data_source/anli_v0.1/R3/test.jsonl'}, \
        "mnli_matched":{"train": "../data_source/multinli_1.0/multinli_1.0_train.jsonl", \
                        "dev": "../data_source/multinli_1.0/multinli_1.0_dev_matched.jsonl",\
                        "test": "../data_source/multinli_1.0/multinli_1.0_dev_matched.jsonl"}, \
        "mnli_mismatched":{"train": "./multinli_1.0/multinli_1.0_train.jsonl", \
                "dev": "./multinli_1.0/multinli_1.0_dev_mismatched.jsonl", \
                "test": "./multinli_1.0/multinli_1.0_dev_mismatched.jsonl"}, \
        "qnli":{"train": '../data_source/QNLI/train.tsv', \
                "dev": '../data_source/QNLI/dev.tsv', \
                "test": '../data_source/QNLI/dev.tsv'}, \
        "roc":{"train": "/home/shanshan/generate_noise/ESIM/data/data_source/ROC/cloze_valid.csv",\
                "dev": '/home/shanshan/generate_noise/ESIM/data/data_source/ROC/cloze_valid.csv',\
                "test": "/home/shanshan/generate_noise/ESIM/data/data_source/ROC/cloze_test.csv"},\
        "copa":{"train": "/home/shanshan/generate_noise/ESIM/data/data_source/COPA/train.csv",\
                "dev": '/home/shanshan/generate_noise/ESIM/data/data_source/COPA/test.csv',\
                "test": "/home/shanshan/generate_noise/ESIM/data/data_source/COPA/test.csv"},\
        "swag":{"train": "../data_source/res/swag/data/train.csv",\
                "dev": '../data_source/res/swag/data/val.csv',\
                "test": "../data_source/res/swag/data/val.csv"},\
        "arct":{"train": "/home/shanshan/generate_noise/ESIM/data/data_source/ARCT/train-full.txt",\
                "dev": '/home/shanshan/generate_noise/ESIM/data/data_source/ARCT/dev-full.txt',\
                "test": "/home/shanshan/generate_noise/ESIM/data/data_source/ARCT/test-full.txt"},\
        "arct2":{"train": "../data_source/ARCT2/train-adv-negated.csv",\
                "dev": '../data_source/ARCT2/dev-adv-negated.csv',\
                "test": "../data_source/ARCT2/test_adv_changeorder.csv"},\
        "common_qa":{"train": "../data_source/commonsenseQA/train_rand_split.jsonl",\
                "dev": '../data_source/commonsenseQA/dev_rand_split.jsonl',\
                "test": "../data_source/commonsenseQA/dev_rand_split.jsonl"},\
        "reclor":{"train": "/home/shanshan/generate_noise/ESIM/data/data_source/reclor/train.csv",\
                "dev": '/home/shanshan/generate_noise/ESIM/data/data_source/reclor/val.csv',\
                "test": "/home/shanshan/generate_noise/ESIM/data/data_source/reclor/val.csv"},\
        "race":{"train": "../data_source/RACE/train.csv",\
                "dev": '../data_source/RACE/dev.csv',\
                "test": "../data_source/RACE/test.csv"},\
        "ubuntu":{"train": "../data_source/ubuntu/dstc7_nli/train.csv",\
                "dev": '../data_source/ubuntu/dstc7_nli/dev.csv',\
                "test": "../data_source/ubuntu/dstc7_nli/dev.csv"},\
    }
'''
#task_name = 'anli'
#for lemma in ['', 'lemma']:
#for lemma in ['']:
#output_dev = f'./{task_name.upper()}/test/{lemma}/test_R2.csv'
#   output_train = f'./{task_name.upper()}/test/{lemma}/test_R1.csv'
#    output_test = f'./{task_name.upper()}/test/{lemma}/test_R3.csv'
#    output_dev = f'./{task_name.upper()}/dev/{lemma}/original_dev.csv'
#    output_train = f'./{task_name.upper()}/train/{lemma}/original_train.csv'
#    output_test = f'./{task_name.upper()}/test/{lemma}/original_test.csv' 
#    
#    transformat(input_files_dict[task_name]["dev"], output_dev, lemma)
#    transformat(input_files_dict[task_name]["train"], output_train, lemma)
#    transformat(input_files_dict[task_name]["test"], output_test, lemma)
'''

def check_path(file_path):    
    file_dir = os.path.dirname(os.path.realpath(file_path))
    if os.path.exists(Path(file_path)) == True:
        return True
    else:
        if os.path.exists(file_dir) == True:
            return True
        else:
            return False
        
def process_data(task_name, file_type, outputf, inputf=None, lemmatize=False, end=False):

    if task_name in input_files_dict:
        e_input = input_files_dict[task_name]
        inf = e_input[file_type]
    else:
        inf = inputf
    outf = outputf
    print(Path(inf))
    print(outf)
    assert os.path.exists(Path(inf))
    assert check_path(outf)
    print("-------------transformat-----------------") 
    transformat(task_name, file_type, inf, outf, lemmatize, end)


if __name__ == "__main__":
    process_data()
