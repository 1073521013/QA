# -*- coding: utf-8 -*-
# prediction using model.process--->1.load data. 2.create session. 3.feed data. 4.predict
import sys
import tensorflow as tf
import numpy as np
import ostest.txt
from model import joint_knowledge_model
from data_util import *
import math
import pickle

# configuration
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", 0.005, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")  # 6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.87, "Rate of decay for learning rate.")  # 0.87一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir", "checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length", 25, "max sentence length")  ############################# importantly
tf.app.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", False, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 1, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_step", 1000, "how many step to validate.")  # 1000做一次检验
tf.app.flags.DEFINE_integer("hidden_size", 100, "hidden size") #128
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")

tf.app.flags.DEFINE_boolean("enable_knowledge", True, "whether to use knwoledge or not.")
tf.app.flags.DEFINE_string("knowledge_path", "knowledge", "file for data source")
tf.app.flags.DEFINE_string("data_source", "knowledge/my_data.txt", "file for data source")
tf.app.flags.DEFINE_boolean("test_mode", False, "whether use test mode. if true, only use a small amount of data")

tf.app.flags.DEFINE_string("validation_file", "my_test.txt", "validation file")

# create session and load the model from checkpoint
# load vocabulary for intent and slot name
word2id = create_or_load_vocabulary(None, FLAGS.knowledge_path)
id2word = {value: key for key, value in word2id.items()}
word2id_intent = create_or_load_vocabulary_intent(None, FLAGS.knowledge_path)
id2word_intent = {value: key for key, value in word2id_intent.items()}
word2id_slotname = create_or_load_vocabulary_slotname_save(None, FLAGS.knowledge_path)
id2word_slotname = {value: key for key, value in word2id_slotname.items()}
knowledge_dict = load_knowledge(FLAGS.knowledge_path)

basic_pair = FLAGS.knowledge_path + '/qa_data.txt'
q2a_dict, a2q_dict, q_list, q_list_index = process_qa(basic_pair, word2id, FLAGS.sequence_length)

intent_num_classes = len(word2id_intent)
vocab_size = len(word2id)
slots_num_classes = len(id2word_slotname)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
FLAGS.batch_size = 1
sequence_length_batch = [FLAGS.sequence_length] * FLAGS.batch_size
model = joint_knowledge_model(intent_num_classes, FLAGS.learning_rate, FLAGS.decay_steps, FLAGS.decay_rate,
                              FLAGS.sequence_length, vocab_size, FLAGS.embed_size, FLAGS.hidden_size,
                              sequence_length_batch, slots_num_classes, FLAGS.is_training, S_Q_len=len(q_list_index))
# initialize Saver
saver = tf.train.Saver()
print('restoring Variables from Checkpoint!')
saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

slot_values_file = FLAGS.knowledge_path + '/slot_values.txt'
jieba.load_userdict(slot_values_file)

def accuracy(pred, actual):
    pred = np.array(pred)
    actual = np.array(actual)
    """Returns percentage of correctly classified labels"""
    return sum(pred==actual) / len(actual)

def main(_):
    # sentence = u'我想在香草园买点吃的，请问有什么地方可以推荐吗？'
    # # indices=[240, 277, 104, 274, 344, 259, 19, 372, 235, 338, 338, 338, 338, 338, 338] #[283, 180, 362, 277, 99, 338, 338, 338, 338, 338, 338, 338, 338, 338, 338] #u'帮我把客厅的灯打开'
    # intent, intent_logits, slots, slot_list, similiarity_list_result = predict(sentence)
    # print(sentence)
    # print('intent:{},intent_logits:{}'.format(intent, intent_logits))
    y = []
    y_ = []
    for line in codecs.open('test.txt', 'r', encoding='utf8'):
        x = line.strip()
        x = x.split('|')
        y.append(int(x[1]))
        intent, intent_logits, slots, slot_list, similiarity_list_result = predict(str(x[0]))
        y0 = intent[6:]
        y_.append(int(y0))
    print('pred: ', y_)
    print('ture: ', y)
    print(accuracy(y, y_))
    # for slot_name,slot_value in slots.items():
    #    print('slot_name:{},slot_value:{}'.format(slot_name, slot_value))
    for i, element in enumerate(slot_list):
        slot_name, slot_value = element
        print('slot_name:{},slot_value:{}'.format(slot_name, slot_value))

    # accuracy_similiarity, accuracy_classification = accuarcy_for_similiarity_validation_set()
    # print("accuracy_similiarity:", accuracy_similiarity, ";accuracy_classification:", accuracy_classification)

    predict_interactive()


def accuarcy_for_similiarity_validation_set():  # read validation data from outside file, and compute accuarcy for classification model and similiarity model
    # 1.get validation set
    source_file_name = FLAGS.knowledge_path + "/" + FLAGS.validation_file
    dict_pair = generate_raw_data(source_file_name, knowledge_path=FLAGS.knowledge_path,
                                  target_file=source_file_name + '_raw_data')
    # 2.loop each data
    count_similiarity_right = 0
    count_classification_right = 0
    len_validation = len(dict_pair)

    i = 0
    for sentence, value in dict_pair.items():
        # 3.call predict
        intent, intent_logits, slots, slot_list, similiarity_list_result = predict(sentence)
        y_intent_target = value['intent']
        similiar_intent = similiarity_list_result[0]
        if similiar_intent == y_intent_target:
            count_similiarity_right += 1
        if intent == y_intent_target:
            count_classification_right += 1
        # if i % 10 == 0:
        #     print(i, "count_similiarity_right%:", str(float(count_similiarity_right) / float(i + 1)),
        #           ";count_classification_right%:", str(float(count_classification_right) / float(i + 1)))
        #     print('sentence:{},y_intent_target:{},intent_classification:{},intent_similiar:{}'.format(sentence,
        #                                                                                               y_intent_target,
        #                                                                                               intent,
        #                                                                                               similiar_intent))
        i = i + 1
    # 4.get accuracy
    accuracy_similiarity = float(count_similiarity_right) / float(len_validation)
    accuracy_classification = float(count_classification_right) / float(len_validation)

    return accuracy_similiarity, accuracy_classification


def accuarcy_for_similiarity_validation_setX():  # read cached validation data
    # 1.get validation set
    traing_data, valid_data, test_data, vocab_size, intent_num_classes, slots_num_classes = generate_training_data(
        FLAGS.data_source, FLAGS.knowledge_path, FLAGS.test_mode, sequence_length=FLAGS.sequence_length)
    x_valid, y_intent_valid, y_slots_valid = valid_data
    # 2.loop each data
    count_similiarity_right = 0
    count_classification_right = 0
    len_validation = len(x_valid)
    for i, x_indices in enumerate(x_valid):
        y_intent = y_intent_valid[i]
        sentence = get_sentence_from_index(x_indices)
        # 3.call predict
        intent, intent_logits, slots, slot_list, similiarity_list_result = predict(sentence)
        y_intent_target = id2word_intent[y_intent]
        similiar_intent = similiarity_list_result[0]
        if similiar_intent == y_intent_target:
            count_similiarity_right += 1
        if intent == y_intent_target:
            count_classification_right += 1
        if i % 10 == 0:
            print(i, "count_similiarity_right%:", str(float(count_similiarity_right) / float(i + 1)),
                  ";count_classification_right%:", str(float(count_classification_right) / float(i + 1)))
            print('sentence:{},y_intent_target:{},intent_classification:{},intent_similiar:{}'.format(sentence,
                                                                                                      y_intent_target,
                                                                                                      intent,
                                                                                                      similiar_intent))
    # 4.get accuracy
    accuracy_similiarity = float(count_similiarity_right) / float(len_validation)
    accuracy_classification = float(count_classification_right) / float(len_validation)

    return accuracy_similiarity, accuracy_classification


def get_sentence_from_index(x_indices):
    sentence = [id2word.get(index, UNK) for index in x_indices]
    sentence = "".join(sentence)
    return sentence


def predict(sentence, enable_knowledge=1):
    """
    :param sentence: a sentence.
    :return: intent and slots
    """
    # print("FLAGS.knowledge_path====>:",FLAGS.knowledge_path)
    sentence_indices = index_sentence_with_vocabulary(sentence, word2id, FLAGS.sequence_length,
                                                      knowledge_path=FLAGS.knowledge_path)
    y_slots = get_y_slots_by_knowledge(sentence, FLAGS.sequence_length, enable_knowledge=enable_knowledge,
                                       knowledge_path=FLAGS.knowledge_path)
    # print("predict.y_slots:",y_slots)
    qa_list_length = len(q_list_index)
    feed_dict = {model.x: np.reshape(sentence_indices, (1, FLAGS.sequence_length)),
                 model.y_slots: np.reshape(y_slots, (1, FLAGS.sequence_length)),
                 model.S_Q: np.reshape(q_list_index, (qa_list_length, FLAGS.sequence_length)),
                 # should be:[self.S_Q_len, self.sentence_len]
                 model.dropout_keep_prob: 1.0}
    logits_intent, logits_slots, similiarity_list = sess.run(
        [model.logits_intent, model.logits_slots, model.similiarity_list], feed_dict)  # similiarity_list:[1,None]
    intent, intent_logits, slots, slot_list, similiarity_list_result = get_result(logits_intent, logits_slots,
                                                                                  sentence_indices, similiarity_list)
    return intent, intent_logits, slots, slot_list, similiarity_list_result


def get_y_slots_by_knowledge(sentence, sequence_length, enable_knowledge=1, knowledge_path=None):
    """get y_slots using dictt.e.g. dictt={'slots': {'全部范围': '全', '房间': '储藏室', '设备名': '四开开关'}, 'user': '替我把储藏室四开开关全关闭一下', 'intent': '关设备<房间><全部范围><设备名>'}"""
    # knowledge_dict=#{'储藏室': '房间', '全': '全部范围', '四开开关': '设备名'}
    user_speech_tokenized = tokenize_sentence(sentence,
                                              knowledge_path=knowledge_path)  # ['替', '我', '把', '储藏室', '四开', '开关', '全', '关闭', '一下']
    result = [word2id_slotname[O]] * sequence_length
    if enable_knowledge == '1' or enable_knowledge == 1:
        for i, word in enumerate(user_speech_tokenized):
            slot_name = knowledge_dict.get(word, None)
            ##TODO print('i:{},word_index:{},word:{},slot_name:{}'.format(i,word,id2word.get(word,UNK),slot_name))
            if slot_name is not None:
                try:
                    result[i] = word2id_slotname[slot_name]
                except:
                    pass
    return result


def predict_interactive():
    sys.stdout.write("Please Input Story.>")
    sys.stdout.flush()
    question = sys.stdin.readline()
    enable_knowledge = 1
    while question:
        if question.find("disable_knowledge") >= 0:
            enable_knowledge = 0
            print("knowledge disabled")
            print("Please Input Story>")
            sys.stdout.flush()
            question = sys.stdin.readline()
        elif question.find("enable_knowledge") >= 0:
            enable_knowledge = 1
            # 3.read new input
            print("knowledge enabled")
            print("Please Input Story>")
            sys.stdout.flush()
            question = sys.stdin.readline()

        # 1.predict using quesiton
        intent, intent_logits, slots, slot_list, similiarity_list = predict(question, enable_knowledge=enable_knowledge)
        # 2.print
        print('intent:{},intent_logits:{}'.format(intent, intent_logits))
        for i,similiarity in enumerate(similiarity_list):
           print('i:{},similiarity:{}'.format(i, similiarity))
        for slot_name, slot_value in slots.items():
           print('slot_name:{}-->slot_value:{}'.format(slot_name, slot_value))
        # for i, element in enumerate(slot_list):
        #     slot_name, slot_value = element
        #     print('slot_name:{},slot_value:{}'.format(slot_name, slot_value))
        # 3.read new input
        print("Please Input Story>")
        sys.stdout.flush()
        question = sys.stdin.readline()


def get_result(logits_intent, logits_slots, sentence_indices, similiarity_list, top_number=3):
    index_intent = np.argmax(logits_intent[0])  # index of intent
    intent_logits = logits_intent[0][index_intent]
    # print("intent_logits:",index_intent)
    intent = id2word_intent[index_intent]

    slots = []
    indices_slots = np.argmax(logits_slots[0], axis=1)  # [sequence_length]
    for i, index in enumerate(indices_slots):
        slots.append(id2word_slotname[index])
    slots_dict = {}
    slot_list = []
    for i, slot in enumerate(slots):
        word = id2word[sentence_indices[i]]
        # print(i,"slot:",slot,";word:",word)
        if slot != O and word != PAD and word != UNK:
            slots_dict[slot] = word
            slot_list.append((slot, word))

    # get top answer for the similiarity list.
    similiarity_list_top = np.argsort(similiarity_list)[-top_number:]
    # similiarity_list_top = np.argsort(similiarity_list)[:top_number]
    similiarity_list_top = similiarity_list_top[::-1]
    # print(similiarity_list_top)
    similiarity_list_result = []
    for k, index in enumerate(similiarity_list_top):
        question = q_list[index]
        answer = q2a_dict[question]
        similiarity_list_result.append(answer)  # TODO print('similiarity.index:{} question:{}, answer:{}'.format(k,question, answer))
    return intent, intent_logits, slots_dict, slot_list, similiarity_list_result


if __name__ == "__main__":
    tf.app.run()
