import ast
import random
import re

import numpy as np

lm_pattern = re.compile(r'''\[(\[\d*,\s*\d*\](,\s*)?)+\]''')

SIM_RADIUS = 3.0  # simulator coordinates range from (-3, -3) to (3, 3)
REAL_RADIUS = 170.0  # limited by the y-axis of the table
REAL_OFFSET = 85.0


PROMPT = {
    'circle': '[[322, 334], [362, 284], [356, 226], [322, 181], [250, 164], [190, 181], [147, 227], [153, 283], [192, 336], [256, 360]]',
    'rect': '[[233, 317], [277, 317], [328, 317], [329, 253], [330, 193], [234, 194], [187, 195], [183, 254], [185, 318], [283, 192]]',
    'triangle': '[[255, 348], [218, 296], [173, 244], [139, 185], [299, 295], [340, 237], [375, 186], [316, 186], [259, 185], [200, 185]]',
    'pyramid': '[[253, 357], [215, 305], [170, 253], [135, 195], [296, 304], [255, 253], [339, 253], [381, 195], [302, 194], [216, 194]]',
    'letterK': '[[176, 344], [176, 278], [176, 216], [176, 145], [248, 271], [311, 337], [269, 217], [336, 157]]',
    'letterZ': '[[186, 331], [260, 331], [332, 331], [276, 279], [215, 229], [173, 181], [247, 181], [326, 181]]',
    'cross': '[[257, 346], [257, 294], [257, 219], [257, 168], [204, 254], [155, 254], [309, 254], [361, 254]]',
    'rhombus': '[[257, 368], [198, 313], [136, 253], [186, 201], [256, 150], [320, 200], [374, 259], [318, 316]]',
    'arrow': '[[145, 256], [221, 256], [294, 256], [366, 256], [328, 302], [285, 347], [330, 209], [280, 161]]',
    'grid': '[[205, 312], [256, 312], [313, 312], [205, 256], [256, 256], [313, 256], [205, 198], [256, 198]]',
    'letterR': '[[200, 344], [200, 282], [200, 220], [200, 158], [263, 348], [312, 301], [261, 260], [295, 211], [333, 160]]',
    'letterO': '[[262, 379], [350, 347], [395, 270], [378, 189], [315, 131], [213, 124], [134, 172], [120, 267], [166, 344]]',
    'letterB': '[[200, 344], [200, 282], [200, 220], [200, 158], [261, 344], [302, 299], [257, 255], [325, 209], [274, 163]]',
    'letterT': '[[258, 358], [196, 358], [132, 358], [332, 358], [397, 358], [258, 296], [258, 234], [258, 176], [258, 119]]',
    'line0': '[[256, 463], [256, 403], [256, 341], [256, 282], [256, 221], [256, 161], [256, 104], [256, 46]]',
    'line1': '[[403, 402], [361, 360], [317, 316], [275, 274], [232, 231], [190, 188], [150, 148], [109, 107]]',
    'line2': '[[465, 255], [404, 255], [342, 255], [284, 255], [222, 254], [162, 254], [105, 254], [47, 254]]',
    'line3': '[[403, 107], [361, 149], [317, 193], [275, 235], [232, 278], [190, 321], [150, 361], [109, 402]]'
}


def get_landmarks(num, shape='circle'):
    def remap(v, offset):
        new_range = SIM_RADIUS - boundary
        pos_at_origin = (v - offset) / scale * new_range * 2

        return pos_at_origin

    # for testing, it is a triangle consists of 10 objects
    prompt = PROMPT[shape]

    # prompt = ''
    # while not lm_pattern.match(prompt):
    #     prompt = input('please input the landmark settings: ')

    landmarks = ast.literal_eval(prompt)
    min_x, max_x = min(x for x, _ in landmarks), max(x for x, _ in landmarks)
    min_y, max_y = min(y for _, y in landmarks), max(y for _, y in landmarks)
    mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2

    scale = max(max_y - min_y, max_x - min_x)
    boundary = 0.4
    landmarks = [(remap(x, mid_x), remap(y, mid_y)) for x, y in landmarks]

    landmarks = np.array(random.sample(landmarks, num))
    return landmarks


def real2sim_coord(x, y):
    sim_x = x / REAL_RADIUS * SIM_RADIUS
    sim_y = -(y - REAL_OFFSET) / REAL_RADIUS * SIM_RADIUS
    return sim_x, sim_y


def sim2real_coord(x, y):
    real_x = x / SIM_RADIUS * REAL_RADIUS
    real_y = -(y / SIM_RADIUS * REAL_RADIUS) + REAL_OFFSET
    return real_x, real_y
