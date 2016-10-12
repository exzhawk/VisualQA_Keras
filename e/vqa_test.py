# -*- encoding: utf-8 -*-
# Author: Epix
import cPickle

import numpy as np
from keras.layers import Dense, Input, merge, Dropout
from keras.layers import Embedding, LSTM
from keras.models import Model
from keras.optimizers import RMSprop

from prepare import get_matrix, val_result, prepare_all
from utils import data_path, QUESTION_LENGTH, MAX_ANSWER


def VQA():
    word_vec_list = cPickle.load(open(data_path('word_vec.pkl'), 'rb'))
    word_vec_len = len(word_vec_list)
    word_vec_list = np.asarray(word_vec_list)
    img_input = Input(shape=(4096,), name='input_img')
    x_img = Dense(1024, activation='tanh', name='fc1')(img_input)
    question_input = Input(shape=(QUESTION_LENGTH,), name='input_question')
    x_str = Embedding(word_vec_len, 300, input_length=QUESTION_LENGTH, mask_zero=True, weights=[word_vec_list])(
        question_input)
    x_str = LSTM(2048, dropout_W=0.5, consume_less='gpu')(x_str)
    x_str = Dense(1024, activation='tanh', name='fc4')(x_str)
    x_f = merge([x_img, x_str], mode='mul', name='merge1')
    x_f = Dense(MAX_ANSWER, activation='tanh', name='fc5')(x_f)
    x_f = Dropout(0.5)(x_f)
    x_f = Dense(MAX_ANSWER, activation='tanh', name='fc6')(x_f)
    x_f = Dropout(0.5)(x_f)
    x_f = Dense(MAX_ANSWER, activation='softmax', name='predictions')(x_f)
    model = Model(input=[img_input, question_input], output=x_f)

    return model


if __name__ == '__main__':
    # prepare_all()
    batch_size = 500
    epoch = 1
    train_images, train_questions, train_answers = get_matrix('train')
    val_images, val_questions, val_answers = get_matrix('val')
    m = VQA()
    rmsprop = RMSprop(lr=3e-4)
    m.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    for i in range(100):
        print(i)
        m.fit([train_images, train_questions], train_answers, batch_size=batch_size, nb_epoch=epoch)
        # m.save('vqa.h5')
        # m = load_model('vqa.h5')

        p = m.predict([val_images, val_questions], batch_size=batch_size, verbose=1)
        p_answers = p.argmax(axis=-1)
        cPickle.dump(p_answers, open(data_path('predict.pkl'), 'wb'), cPickle.HIGHEST_PROTOCOL)
        val_result(p_answers, val_answers)
