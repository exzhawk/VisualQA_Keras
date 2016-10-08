# -*- encoding: utf-8 -*-
# Author: Epix
import cPickle
import codecs
import json
import os
from collections import defaultdict, Counter

import spacy
from nltk import word_tokenize

from settings import *

spacy_t = None


def get_spacy_tokenizer():
    global spacy_t
    if spacy_t is None:
        spacy_t = spacy.en.English()
        return spacy_t
    else:
        return spacy_t


def data_path(*filename):
    return os.path.join(DATA_FOLDER, *filename)


def tokenize(sentence,
             # method='simple',
             # method='nltk',
             method='spacy',
             ):
    if method == 'simple':
        sentence = re_filter.sub(' ', sentence)
        sentence = re_filter2.sub(' ', sentence)
        sentence = sentence.lower().strip()
        return sentence.split()
    elif method == 'nltk':
        return word_tokenize(sentence)
    elif method == 'spacy':
        sentence = sentence.replace(u'/', ' or ').lower()
        return [word.norm_ for word in get_spacy_tokenizer()(sentence)]


class Count:
    def __init__(self, default_func=None):
        if default_func:
            self.d = defaultdict(default_func)
        else:
            self.d = defaultdict(lambda: 0)

    def update(self, s):
        self.d[s] += 1

    def get_counter(self):
        return Counter(self.d)


def count_questions_len():
    """
    count max len of one question in train set
    :return:
    """
    questions_j = json.load(open(data_path('f_OpenEnded_mscoco_train2014_questions.json'), 'r'))
    max_len = 0
    for question in questions_j['questions']:
        question_str = question['question']
        question_len = len(tokenize(question_str))
        if question_len > max_len:
            max_len = question_len
    print('max length of question: {}'.format(max_len))


def gen_answers_id_mapping():
    """
    generate top common answers to id mapping
    :return: None
    """
    answer_j = json.load(open(data_path('f_mscoco_train2014_annotations.json')))
    answer_str_count = Count()
    for annotation in answer_j['annotations']:
        for answer in annotation['answers']:
            answer_str_count.update(answer['answer'])

    c = answer_str_count.get_counter()
    json.dump({answer[0]: index for index, answer in enumerate(c.most_common(MAX_ANSWER))},
              open(data_path('f_answers_id_map.json'), 'wb'), indent=2)


def count_question_word():
    """
    count all words and frequency in question
    :return:
    """
    questions_j = json.load(open(data_path('f_OpenEnded_mscoco_train2014_questions.json'), 'r'))
    question_word_count = Count()
    for question in questions_j['questions']:
        question_str = question['question']
        question_words = tokenize(question_str)
        for question_word in question_words:
            question_word_count.update(question_word)
    c = question_word_count.get_counter()
    pass


def gen_question_word_set():
    """
    generate all words used in question
    :return:
    """
    questions_j = json.load(open(data_path('f_OpenEnded_mscoco_train2014_questions.json'), 'r'))
    question_word_set = set()
    for question in questions_j['questions']:
        question_str = question['question']
        question_words = tokenize(question_str)
        question_word_set |= set(question_words)
    json.dump({'words': list(question_word_set)}, open(data_path('f_question_words.json'), 'wb'), indent=2)
    print('words are used in questions: {}'.format(len(question_word_set)))


def gen_image_id_filename():
    """
    generate picture id to filename mapping
    :return:
    """
    mapping = dict()
    annotations_train = json.load(open(data_path('f_mscoco_train2014_annotations.json')))
    for annotation in annotations_train['annotations']:
        image_id = annotation['image_id']
        image_path = data_path('train2014', 'COCO_train2014_{:0>12}.jpg'.format(image_id))
        mapping[image_id] = image_path
    annotations_val = json.load(open(data_path('f_mscoco_val2014_annotations.json')))
    for annotation in annotations_val['annotations']:
        image_id = annotation['image_id']
        image_path = data_path('val2014', 'COCO_val2014_{:0>12}.jpg'.format(image_id))
        mapping[image_id] = image_path
    json.dump(mapping, open(data_path('f_image_id_path_map.json'), 'wb'), indent=2)


def gen_data(m='train'):
    """
    generate data for training and testing
    :param m: gen type
    :return:
    """
    answers_mapping = json.load(open(data_path('f_answers_id_map.json'), 'r'))
    question_word_mapping = json.load(open(data_path('f_word_id_map.json'), 'r'))
    images_mapping = cPickle.load(open(data_path('image_id_feature_map.pkl'), 'rb'))
    annotations = json.load(open(data_path('f_mscoco_{}2014_annotations.json'.format(m)), 'r'))['annotations']
    questions = json.load(open(data_path('f_OpenEnded_mscoco_{}2014_questions.json'.format(m)), 'r'))['questions']
    questions_index = {q['question_id']: q['question'] for q in questions}
    images_list = []
    questions_list = []
    answers_list = []
    for annotation in annotations:
        image_id = annotation['image_id']
        image_feature = images_mapping[str(image_id)]

        question_id = annotation['question_id']
        question_str = questions_index[question_id]
        question_word_list = tokenize(question_str)
        question_word_id_list = [question_word_mapping.get(word, 0) for word in question_word_list]

        possible_answers = get_possible_answers(annotation['answers'], answers_mapping)
        if m == 'train':
            if not possible_answers:
                continue
            # answer_str = random.choice(possible_answers)
            # answer_str = possible_answers[0]
            answer_str = Counter(possible_answers).most_common()[0][0]
            answer_id = answers_mapping[answer_str]
        else:
            answer_id = [answers_mapping[answer_str] for answer_str in possible_answers]

        questions_list.append(question_word_id_list)
        images_list.append(image_feature)
        answers_list.append(answer_id)
    cPickle.dump((images_list, questions_list, answers_list),
                 open(data_path('{}_matrix.pkl'.format(m)), 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)


def get_possible_answers(answers, known_answers):
    """
    return possible answers in know answers
    :param answers:
    :param known_answers:
    :return:
    """
    known_possible_answers = []
    for answer in answers:
        if answer['answer_confidence'] == 'yes' and answer['answer'] in known_answers:
            known_possible_answers.append(answer['answer'])
    return known_possible_answers
    # return random.choice(known_possible_answers)


def utf8decode_test(filename='../data/wiki.en.model.vec'):
    with codecs.open(filename, 'r', encoding='utf8', errors='strict') as t:
        index = 0
        try:
            for index, line in enumerate(t):
                pass
        except:
            print(index)
    pass


if __name__ == '__main__':
    # utf8decode_test()
    pass
