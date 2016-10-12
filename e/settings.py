# -*- encoding: utf-8 -*-
# Author: Epix
import re

DATA_FOLDER = '../data'
re_filter = re.compile('[^a-zA-Z0-9]')
re_filter2 = re.compile('\s+')
QUESTION_LENGTH = 25
MAX_ANSWER = 1000
