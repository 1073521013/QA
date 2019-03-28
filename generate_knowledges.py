# -*- coding: utf-8 -*-
#1.read a line
#2.get 1)user_speech, 2)intent, 3)slots, 4)get knowledges for the slots
import json
import codecs
import os
import random
slot_values_file='slot_values.txt'
slot_value_name_pair_file='slot_pairs.txt'
slot_names_file='slot_names.txt'
splitter='|&|'
splitter_slot_names='||'

def get_knowledge(data_source_file,knowledge_path,test_mode=False):
    #if target file not exist, create; otherwise return
    slot_value_name_pair_filee=knowledge_path+"/"+slot_value_name_pair_file
    slot_values_filee=knowledge_path + "/" + slot_values_file
    slot_names_filee=knowledge_path + "/" + slot_names_file
    if  os.path.exists(slot_value_name_pair_filee) and os.path.exists(slot_values_filee) and os.path.exists(slot_names_filee):
        print("knowledge exists. will not generate it.")
        return
    else:
        print("knowledge not exists. will start to generate it.")


    knowledge_dict = {}
    slot_name_set=set()
    with open(data_source_file, encoding="utf8") as f:
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
                        for j,k in e.items():
                            knowledge_dict[k]=j
                            slot_name_set.add(j)
                    else:
                        d={}
                        for j,k in e.items():
                            d[k]=j
                            knowledge_dict.update(d)
                            slot_name_set.add(j)


    #print
    #1.write slot_value to file system
    #2.write slot_value-slot_name pair to file system
    #3.write total slot name to file systm
    #ii = 0
    slot_values_file_object = codecs.open(slot_values_filee, 'w', 'utf-8')
    slot_value_name_pair_file_object=codecs.open(slot_value_name_pair_filee,'w','utf-8')
    slot_names_file_object=codecs.open(slot_names_filee,'w','utf-8')

    #if not os.path.exists(slot_value_name_pair_file) and not os.path.exists(slot_values_file) :
    for k, v in knowledge_dict.items():
        if len(k)<6: #only save short context
            slot_value_name_pair_file_object.write(k+splitter+v+"\n")
            seg_value=str(100000) if len(k)==1 else str(2000)
            slot_values_file_object.write(k+" "+seg_value+"\n")
        #ii = ii + 1
    slot_values_file_object.close()
    slot_value_name_pair_file_object.close()
    print("slot_name_set:",slot_name_set)
    #if not os.path.exists(slot_names_file):
    for element in slot_name_set:
        slot_names_file_object.write(element+"\n")
    slot_names_file_object.close()

    return #knowledge_dict

    #print("knowledge_dict:",knowledge_dict)

#

#get_knowledge(data_source,knowledge_path)
