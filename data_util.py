# -*- coding: utf-8 -*-
import jieba
import collections
import codecs
import os
import numpy as np
import random
import json
from generate_knowledges import get_knowledge

UNK=u'_UNK'
O=u'_O'
PAD=u'_PAD'
splitter='|&|'
splitter_slot_names='||'

def generate_training_data(data_file,knowledge_path,test_mode,vocabulary_size=None,sequence_length=15):
    """generate training data from a file, return training,validation and test set.
      traininig data contain x,y_intent,y_slots: x for sentence,y_intent for intent,y_slots for slots"""

    cache_file = knowledge_path+'/cache_data.npy'
    print(cache_file,"exists or not:",os.path.exists(cache_file))
    if os.path.exists(cache_file):
         np.load(cache_file)
    #0.generate knowledge and load knowledge to jieba
    get_knowledge(data_file,knowledge_path,test_mode=test_mode)
    slot_values_file = knowledge_path+'/slot_values.txt'
    jieba.load_userdict(slot_values_file)

    #1. retrieve raw data from file system
    data_dict=generate_raw_data(data_file,knowledge_path=knowledge_path)

    #2.create vocabulary
    #2.1 create vocabulary for input and save to file system;
    sentence_user_speech_list=list(data_dict.keys())
    word2id_x=create_or_load_vocabulary(sentence_user_speech_list,knowledge_path,vocabulary_size=vocabulary_size)

    #2.2 create vocabulary for intent and save to file system;
    word2id_intent=create_or_load_vocabulary_intent(data_dict,knowledge_path)

    #2.3 create vocabulary for slots and save to file system;
    word2id_slotname=create_or_load_vocabulary_slotname_save(data_dict,knowledge_path)

    #2.4 get knowledge dict from file system.
    #3. get training,valid,test data
    traing_data, valid_data, test_data=get_training_valid_test_data(data_dict, word2id_x, sequence_length, word2id_intent, word2id_slotname)

    vocab_size=len(word2id_x)
    intent_num_classes=len(word2id_intent)
    slot_num_classes=len(word2id_slotname)

    result=traing_data,valid_data,test_data,vocab_size,intent_num_classes,slot_num_classes,word2id_x
    # result=traing_data,valid_data,test_data
    if not os.path.exists(cache_file):
        np.save(cache_file,result)
    #5. return data.

    return result

def process_qa(file_name,word2id_x,sequence_length):
    #1.read qa file
    source_file_object=codecs.open(file_name,mode='r',encoding='utf-8')
    lines=source_file_object.readlines()
    #2.put to dict:q2a,a2q;q_list
    q2a_dict={}
    a2q_dict={}
    q_list=[]
    q_list_index=[]
    for i,line in enumerate(lines):
        question,answer=line.strip().split(splitter)
        q2a_dict[question]=answer
        a2q_dict[answer]=question
        q_list.append(question)
        # 3.process qa as list of index, so later it can be feed
        question_index= index_sentence_with_vocabulary(question, word2id_x, sequence_length=sequence_length)
        q_list_index.append(question_index)
    print("process_qa.total length:",len(lines),";length of q_list_index:",len(q_list_index))
    return q2a_dict,a2q_dict,q_list,q_list_index

def generate_raw_data(source_file_name,knowledge_path=None,target_file=None):
    target_file = None
    with open(source_file_name, encoding="utf8") as f:
        result_dict = {}
        train_set = {}
        train_set["rasa_nlu_data"] = {}
        train_set["rasa_nlu_data"]["common_examples"] = []
        dict_set = []
        entitiys_name = []
        for line in f:
            if line.strip() == "":
                continue
            elif "@" in line:
                continue
            elif "text" in line:
                entitiys_name = []
                cols = line.strip().split(",")
                for name in cols[2:]:
                    entitiys_name.append(name)
                continue
            else:
                tokens = line.strip().split("|")
                common_example = {}
                common_example["user"] = tokens[0]
                common_example["intent"] = tokens[1]
                common_example["slots"] = []
                if len(tokens) < 3:
                    train_set["rasa_nlu_data"]["common_examples"].append(common_example)
                    continue
                entitiys = tokens[2].split("，")
                for i, e in enumerate(entitiys):
                    try:
                        entity = {}
                        entity[entitiys_name[i]] = e
                        common_example["slots"].append(entity)
                        dict_set.append(e)
                    except:
                        pass
                train_set["rasa_nlu_data"]["common_examples"].append(common_example)
                for i, e in enumerate(common_example['slots']):
                    if i==0:
                        common_example['slots']=dict(e.items())
                    else:
                        t=common_example['slots']
                        d=t.copy()
                        d.update(e)
                        common_example['slots']=d
                result_dict[common_example['user']] = common_example

    # print to have a look
    print("generate_raw_data.length of result list:",len(result_dict))

    #save it to file system.
    if target_file is None:
        target_file=knowledge_path+'/raw_data.txt'
    target_object=codecs.open(target_file,'w','utf-8')
    for k,v in result_dict.items():
        target_object.write(k+splitter+v['intent']+"\n") #+splitter+str(v)+
    target_object.close()

    return result_dict

def get_training_valid_test_data(data_dict,word2id_x,sequence_length,word2id_intent,word2id_slotname):
    """
    generate training,validation and test data.
    :param data_dict:
    :param word2id_x:
    :param sequence_length:
    :param word2id_intent:
    :param word2id_slotname:
    :return: traing_data,valid_data,test_data. e.g.traing_data is:(x_list, y_intent_list, y_slots_list)
    """
    x_list = []
    y_intent_list = []
    y_slots_list = []
    x_knowledge_list=[]

    x_list_valid = []
    y_intent_list_valid = []
    y_slots_list_valid = []
    x_knowledge_list_valid = []

    x_list_test = []
    y_intent_list_test = []
    y_slots_list_test = []
    x_knowledge_list_test = []

    ii = 0
    for user_speech, dictt in data_dict.items():
        if len(str(user_speech)) <= 2: continue
        x = index_sentence_with_vocabulary(user_speech, word2id_x, sequence_length=sequence_length)
        if 'intent' in dictt:
            y_intent = word2id_intent[dictt['intent']]
        else:
            continue
        y_slots = get_y_slots(dictt, word2id_slotname, sequence_length=sequence_length)
        if ii % 6 != 0:
            x_list.append(x)
            y_intent_list.append(y_intent)
            y_slots_list.append(y_slots)
            #x_knowledge_list.append(x_knowledge)
        else:
            random_variable = np.abs(np.random.randn())
            if random_variable >= 0.4:
                x_list_valid.append(x)
                y_intent_list_valid.append(y_intent)
                y_slots_list_valid.append(y_slots)
            else:
                x_list_test.append(x)
                y_intent_list_test.append(y_intent)
                y_slots_list_test.append(y_slots)
        ii = ii + 1
    traing_data = x_list, y_intent_list, y_slots_list
    valid_data = x_list_valid, y_intent_list_valid, y_slots_list_valid
    test_data = x_list_test, y_intent_list_test, y_slots_list_test

    return traing_data,valid_data,test_data

def create_or_load_vocabulary(sentence_list,knowledge_path,vocabulary_size=None):
    "create vocabulary for x"
    vocabulary_x=knowledge_path+'/vocabulary_x'

    if os.path.exists(vocabulary_x): #if exist,load from file system;else, create vocabulary.
        vocabulary_x_object=codecs.open(vocabulary_x,'r','utf-8')
        vocabulary_x_lines=vocabulary_x_object.readlines()
        word2id={}
        for i,line in enumerate(vocabulary_x_lines):
            if len(line.strip())>2:
                try:
                    word,id=line.strip().split(splitter) #splitter
                    word2id[word]=int(id)
                except:#x='::40103'x
                    print('line.strip().split.error.line--->{}'.format(line))
                    continue

            else:
                print(line)
        #print("###word2id.x",word2id)
        return word2id

    #1. counter frequency of words
    counter=collections.Counter()
    for sentence in sentence_list:
        sentence=sentence.strip()
        if len(sentence)<=0:continue
        seg_list=tokenize_sentence(sentence)
        counter.update(seg_list)

    #2.sort and transform
    if vocabulary_size is not None:
        counter=counter.most_common(vocabulary_size)
    counter.update([UNK])
    counter.update([PAD])
    counter.update([O])
    word2id={element:i for i,element in enumerate(counter)}

    save_vocabulary_file_system(counter,vocabulary_x)
    return word2id #,id2word

def create_or_load_vocabulary_intent(data_dict,knowledge_path):
    vocabulary_intent=knowledge_path+'/vocabulary_intent'
    if os.path.exists(vocabulary_intent):
        vocabulary_intent_object=codecs.open(vocabulary_intent,'r','utf-8')
        vocabulary_intent_lines=vocabulary_intent_object.readlines()
        word2id_intent={}
        for i,line in enumerate(vocabulary_intent_lines):
            intent,id=line.strip().split(splitter)
            word2id_intent[intent]=int(id)
        #print("###word2id_intent:",word2id_intent)
        return word2id_intent

    dict_values=data_dict.values()
    counter_intent = collections.Counter()
    for value in dict_values:
        if 'intent' in value:
            intent = value['intent'].strip()
            counter_intent.update([intent])
        else:
            #print("intent not exists===>");print(value)
            pass

    print("intent_counter:",counter_intent)
    print("number of intent:",len(counter_intent))

    word2id_intent={element:i for i,element in enumerate(counter_intent)} #intent:i for i,intent in enumerate(intent_list)
    save_vocabulary_file_system(counter_intent, vocabulary_intent)
    return word2id_intent

def create_or_load_vocabulary_slotname_save(data_dict,knowledge_path):

    vocabulary_slotnames=knowledge_path+'/vocabulary_slotnames'
    if os.path.exists(vocabulary_slotnames):
        vocabulary_slotname_object=codecs.open(vocabulary_slotnames,'r','utf-8')
        vocabulary_slotname_lines=vocabulary_slotname_object.readlines()
        word2id_slotname={}
        for i,line in enumerate(vocabulary_slotname_lines):
            slotname,id=line.strip().split(splitter)
            word2id_slotname[slotname]=int(id)
        #print("###word2id_slotname:",word2id_slotname)
        return word2id_slotname

    dict_values=data_dict.values()
    #slot_names:
    counter_slot_name = collections.Counter()
    for data in dict_values:
        if 'slots' in data:
            if data['slots'] != []:
                counter_slot_name.update(data['slots'].keys())
    counter_slot_name.update([O])
    list_slot_name=list(counter_slot_name)
    #list_slot_name=[O]+list_slot_name
    word2id_slotname={element:i for i,element in enumerate(list_slot_name)}
    save_vocabulary_file_system(counter_slot_name, vocabulary_slotnames)
    return word2id_slotname

def tokenize_sentence(sentence,knowledge_path=None):
    """tokenize sentence"""
    #sentence=sentence.strip()
    result_list=None
    try:
        result_object=jieba.cut(sentence,cut_all=False)
        seg_sentence = " ".join(result_object)
        result_list = seg_sentence.split()

        #seg_sentence_list=[]
        #for i,element in enumerate(sentence):
        #    if u'\u4e00' <= element <= u'\u9fff':#chinese
        #        seg_sentence_list.append(element)
        #result_list.extend(seg_sentence_list)
    except:
        print("tokenize_sentence.error:",sentence)
    return result_list

def index_sentence_with_vocabulary(sentence,word2id,sequence_length=None,knowledge_path=None):
    """index sentence with vocabulary, return list of index"""
    #print("index_sentence_with_vocabulary:",knowledge_path)
    result_list=tokenize_sentence(sentence,knowledge_path=knowledge_path)
    result_list=result_list[0:sequence_length] #truncate
    unk_id=word2id[UNK]
    index_list=[word2id[PAD]]*sequence_length  #pad
    for i,element in enumerate(result_list):
        index_list[i]=word2id.get(element,unk_id)
    #print("####index_sentence_with_vocabulary.sentence:",sentence);print(index_list)
    return index_list

def save_vocabulary_file_system(counter,file_name):
    if os.path.exists(file_name):
        return
    file_object = codecs.open(file_name, 'a', 'utf-8')

    for id,element in enumerate(counter):
        file_object.write(element+splitter+str(id)+"\n")
    file_object.close()

def load_knowledge(knowledge_path):
    knowledge_pair_file=knowledge_path+'/slot_pairs.txt'
    knowledge_pair_object = codecs.open(knowledge_pair_file, 'r', 'utf-8')
    lines=knowledge_pair_object.readlines()
    knowledge_dict={}
    for i,line in enumerate(lines):
        line=line.strip()
        slot_value,slot_name=line.split(splitter)
        if splitter_slot_names in slot_name:
            slot_name=slot_name.split(splitter_slot_names)[0]
        knowledge_dict[slot_value]=slot_name
    return knowledge_dict


def get_y_slots(dictt,word2id_slotname,sequence_length=None):
    user_speech=dictt['user']
    slots=dictt['slots']
    if slots !=[]:
        slots_reverse={v:k for k,v in slots.items()}
        user_speech_tokenized=tokenize_sentence(user_speech)
        result=[word2id_slotname[O]]*sequence_length
        for i,word in enumerate(user_speech_tokenized):
            if i<sequence_length-1:
                slot_name=slots_reverse.get(word,None)
                if slot_name is not None:
                    result[i]=word2id_slotname[slot_name]
        return result

def write_data_for_fasttext(training_array,target_file,test_file):
    target_object=codecs.open(target_file,'a','utf-8')
    test_object = codecs.open(test_file, 'a', 'utf-8')
    i=0
    if not os.path.exists(target_object) and not os.path.exists(test_object):
        for element in training_array:#element:(x,y_intent,y_slots)
            #print("element:",element)
            x_list,y_intent,y_slots=element
            x_list=['w'+str(element) for element in x_list]
            x_string=" ".join(x_list)
            y_string=str(y_intent)
            if i%5!=0:
                target_object.write(x_string+" __label__"+y_string+"\n")
            else:
                test_object.write(x_string + " __label__" + y_string + "\n")
            i=i+1
        target_object.close()
        test_object.close()

