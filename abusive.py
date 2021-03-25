#-*- encoding: utf-8 -*-
import re
import hgtk
        
def matching_blacklist(abusive_set, input_sentence):
    result_list = list()
    for i,abusive_word in enumerate(abusive_set):
        if abusive_word in input_sentence:
            input_split = input_sentence.split(" ")
            for j,input_word in enumerate(input_split):
                 if re.match('[a-zA-Z]+', input_word) and input_word == abusive_word:
                    result_list.append(abusive_word)
                 else:
                    result_list.append(abusive_word)
    return result_list

def edit_distancing(abusive_set, input_sentence):
    result_list = list()
    for i,abusive_word in enumerate(abusive_set):
        for j,input_word in enumerate(input_sentence.split(" ")):
            if hgtk.text.decompose(abusive_word).replace("ᴥ","") == hgtk.text.decompose(input_word).replace("ᴥ","") and not hgtk.checker.is_latin1(input_word):
                result_list.append(input_word)
    return result_list

def n_gram_token(abusive_set, input_sentence, n=2):
    result_list = list()
    if len(input_sentence.split(" ")) < n :
        return result_list
    n_gram_list = list()
    sentence_token_list = input_sentence.split(" ")
    sentence_token_list = list(filter(lambda a: a != '', sentence_token_list))
    for i in range(len(sentence_token_list) - n + 1):
        n_gram_element = ""
        for j in range(n):
                n_gram_element+=sentence_token_list[i + j]
        n_gram_list.append(n_gram_element)
    for i,abusive_word in enumerate(abusive_set):
        for j,n_gram_word in enumerate(n_gram_list):
            if(abusive_word in n_gram_word):
                result_list.append(n_gram_word)
    return result_list

def remove_duplicate(abusive_set, sentence):
    result_list = list()

def remove_whitelist(white_list, abusive_word_list):
    result_list = list()
    for i,abusive_word in enumerate(abusive_word_list):
        if not abusive_word in white_list:
            result_list.append(abusive_word)
    return result_list

def set_dict(filename):
    dict_file = open(filename, 'r', encoding='UTF8')
    dict_set = set()
    for line in dict_file.readlines():
        dict_set.add(line.split('\t')[0])
        
    return dict_set


if __name__ == '__main__':
 
    abusive_dic_file = open("/data/abusive_dictionary.txt",'r', encoding='UTF8')
    abusive_dic_set = set()
    for line in abusive_dic_file.readlines():
        abusive_dic_set.add(line.split("\t")[0])
    whitelist_dic_file = open("/data/abusive_dictionary2.txt",'r', encoding='UTF8')
    whitelist_dic_set = set()
    for line in whitelist_dic_file.readlines():
        whitelist_dic_set.add(line.split("\t")[0])
        
    input_sentence = input("input: ")
    input_sentence = input_sentence.lower()
    input_no_punc = re.sub("[!@#$%^&*().?\"~/<>:;'{}]","",input_sentence)

    abusive_word_list = list()

    abusive_word_list += matching_blacklist(abusive_dict_set, input_sentence)
    abusive_word_list += matching_blacklist(abusive_dict_set, input_no_punc)

    if abusive_word_list == 0:
        abusive_word_list += edit_distancing(abusive_dict_set, input_sentence)
        abusive_word_list += n_gram_token(abusive_dict_set, input_sentence)

    remove_whitelist(whitelist_dict_set, abusive_word_list)
    
    print("input_sentence", input_sentence)
    if len(abusive_word_list) == 0:
        result['input'] = input_sentence
        result['tag'] = 0
    else:
        result['input'] = input_sentence
        result['tag'] = 1
        result['abusive_words'] = abusive_word_list

    json.dumps(result)
