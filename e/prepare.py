# -*- encoding: utf-8 -*-
# Author: Epix
import cPickle
import codecs
import json

import gensim
import numpy as np
import numpy
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.preprocessing import sequence
from keras.utils import np_utils
from tqdm import tqdm

from get_vgg16 import VGG16_dense
from utils import data_path, count_questions_len, gen_answers_id_mapping, gen_question_word_set, gen_data, \
    QUESTION_LENGTH, MAX_ANSWER


def gen_word_vector_mapping(
        # model_name='googlenews',
        # model_name='glove',
        model_name='fasttext',
):
    """
    generate word vec mapping
    :return:
    """
    files = {'googlenews': ('GoogleNews-vectors-negative300.bin', True),
             'glove': ('glove.840B.300d.txt', False),
             'fasttext': ('wiki.en.model.vec', False)}
    f = files[model_name]
    model = gensim.models.Word2Vec.load_word2vec_format(data_path(f[0]), binary=f[1], unicode_errors='strict')
    word_list = json.load(open(data_path('f_question_words.json'), 'r'))['words']
    word_dict = dict()
    no_match_words = []
    for word in word_list:
        try:
            v = model[word]
            word_dict[word] = v
        except:
            no_match_words.append(word)
    json.dump({'missing': no_match_words}, open(data_path('missing_words.json'), 'w'), indent=2)
    cPickle.dump(word_dict, open(data_path('word_dict.pkl'), 'wb'))
    pass


def gen_word_vector_mapping_glove():
    """
    generate word vec mapping
    :return:
    """
    word_list = json.load(open(data_path('f_question_words.json'), 'r'))['words']
    word_set = set(word_list)
    glove_file = codecs.open(data_path('glove.840B.300d.txt'), 'r', encoding='utf8')
    word_dict = dict()
    for line in glove_file:
        seg = line.split(' ')
        word = seg[0]
        if word in word_set:
            word_dict[word] = numpy.asarray(seg[1:])
            word_set.remove(word)
    json.dump({'missing': list(word_set)}, open(data_path('missing_words.json'), 'w'), indent=2)
    cPickle.dump(word_dict, open(data_path('word_dict.pkl'), 'wb'))


def gen_image_id_feature():
    """
    generate image id to feature mapping
    :return:
    """
    image_id_path_mapping = json.load(open(data_path('f_image_id_path_map.json'), 'r'))
    get_vgg16_dense = VGG16_dense(include_top=True, weights='imagenet')
    image_id_feature_mapping = dict()
    bar = tqdm(total=len(image_id_path_mapping))
    for image_id, image_path in image_id_path_mapping.items():
        bar.update()
        x = preprocess_image(image_path)
        y = get_vgg16_dense([x])
        image_id_feature_mapping[image_id] = y[0][0]
    bar.close()
    cPickle.dump(image_id_feature_mapping, open(data_path('image_id_feature_map.pkl'), 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL)


def preprocess_image(image_path):
    """
    resize and preprocess input image
    :param image_path:
    :return:
    """
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def gen_question_word_id_vec():
    """
    generate question word to vector
    :return:
    """
    word_dict = cPickle.load(open(data_path('word_dict.pkl'), 'rb'))
    word_vec = [numpy.zeros(300), numpy.zeros(300)]
    word_id_mapping = dict()
    for index, (word, vec) in enumerate(word_dict.items(), start=2):
        word_vec.append(vec)
        word_id_mapping[word] = index

    cPickle.dump(word_vec, open(data_path('word_vec.pkl'), 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    json.dump(word_id_mapping, open(data_path('f_word_id_map.json'), 'w'), indent=2)


def get_matrix(m='train'):
    images_list, questions_list, answers_list = cPickle.load(open(data_path('{}_matrix.pkl'.format(m)), 'rb'))
    images_list = numpy.asarray(images_list)
    questions_list = numpy.asarray(questions_list)
    questions_list = sequence.pad_sequences(questions_list, maxlen=QUESTION_LENGTH)
    if m == 'train':
        answers_list = numpy.asarray(answers_list)
        answers_list = np_utils.to_categorical(answers_list, MAX_ANSWER)
    else:
        pass
    return images_list, questions_list, answers_list


def prepare_all():
    count_questions_len()
    gen_answers_id_mapping()
    gen_question_word_set()
    gen_word_vector_mapping()
    # gen_word_vector_mapping_glove()
    gen_question_word_id_vec()
    gen_data('train')
    gen_data('val')


def val_result(p_answers=None, val_answers=None):
    """
    evaluate predict result and accuracy
    :return:
    """

    if p_answers is None:
        p_answers = cPickle.load(open(data_path('predict.pkl'), 'rb'))
    if val_answers is None:
        val_images, val_questions, val_answers = get_matrix('val')
    assert len(p_answers) == len(val_answers)
    total = len(p_answers)
    count = 0
    for predict, val in zip(p_answers, val_answers):
        if predict in val:
            count += 1
    print(count, total, float(count) / float(total))
    pass


if __name__ == '__main__':
    gen_word_vector_mapping()
    pass
