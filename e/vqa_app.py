# -*- encoding: utf-8 -*-
# Author: Epix
import json
import os
import sqlite3
import uuid

import numpy
from flask import Flask
from flask import g
from flask import request
from flask import send_from_directory
from keras.models import load_model
from keras.preprocessing import sequence

from get_vgg16 import VGG16_dense
from prepare import preprocess_image
from utils import data_path, tokenize, QUESTION_LENGTH

app = Flask(__name__)
UPLOAD_PATH = 'upload'
STATIC_PATH = 'static'
DATABASE = 'upload.db'


def get_db():
    db = g.get('_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


def get_vgg():
    vgg = g.get('_vgg', None)
    if vgg is None:
        vgg = g._vgg = VGG16_dense(include_top=True, weights='imagenet')
    return vgg


def get_qa():
    qa = g.get('_qa', None)
    if qa is None:
        qa = g._qa = load_model('vqa.h5')
    return qa


def get_word_id_map():
    words = g.get('_words', None)
    if words is None:
        words = g._words = json.load(open(data_path('f_word_id_map.json'), 'r'))
    return words


def get_answers_map():
    answers = g.get('_answers', None)
    if answers is None:
        answers_mapping = g._answers = json.load(open(data_path('f_answers_id_map.json'), 'r'))
        answers = {answer_id: answer_str for answer_str, answer_id in answers_mapping.items()}
    return answers


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


@app.route('/answer', methods=['POST'])
def get_answer():
    j = request.json
    image_path = j['image_path']
    question_str = j['question']
    x_image = preprocess_image(image_path)
    image_feature = get_vgg()([x_image])[0]
    words = get_word_id_map()
    question_word_list = tokenize(question_str)
    question_word_id_list = [words.get(word, 0) for word in question_word_list]
    questions_list = numpy.asarray([question_word_id_list])
    questions_list = sequence.pad_sequences(questions_list, maxlen=QUESTION_LENGTH)
    qa_model = get_qa()
    p = qa_model.predict([image_feature, questions_list])
    p_answers = p.argmax(axis=-1)
    answer_id = p_answers[0]
    answers_map = get_answers_map()
    answer_str = answers_map[answer_id]
    print({'path': image_path, 'question': question_str, 'answer': answer_str})
    return answer_str


@app.route('/upload', methods=['POST'])
def upload():
    upload_files = request.files.values()
    output_filename = os.path.join(UPLOAD_PATH, uuid.uuid4().hex + '.jpg')
    list(upload_files)[0].save(output_filename)
    return output_filename.replace('\\', '/')


@app.route('/upload/<path:path>', methods=['GET'])
def get_upload(path):
    return send_from_directory(UPLOAD_PATH, path)


@app.route('/', methods=['GET'])
def homepage():
    return send_from_directory(STATIC_PATH, 'index.html')


@app.route('/<path:path>')
def send_static(path):
    return send_from_directory(STATIC_PATH, path)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        debug=True
    )
