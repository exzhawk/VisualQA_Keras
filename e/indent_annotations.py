# -*- encoding: utf-8 -*-
# Author: Epix
import json
import os

from settings import *

for f in os.listdir(os.path.join(DATA_FOLDER, 'annotations')):
    print(f)
    j = json.load(open(os.path.join(DATA_FOLDER, 'annotations', f), 'rb'))
    json.dump(j, open(os.path.join(DATA_FOLDER, 'f_{}'.format(f)), 'w'), indent=2)
