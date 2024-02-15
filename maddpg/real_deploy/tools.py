import ast
import random
import re

import numpy as np

lm_pattern = re.compile(r'''\[(\[\d*,\s*\d*\](,\s*)?)+\]''')

SIM_RADIUS = 3.0  # simulator coordinates range from (-3, -3) to (3, 3)
REAL_RADIUS = 255  # limited by the y-axis of the table
REAL_OFFSET = 0


PROMPT = {
    # 'circle': '[[322, 334], [362, 284], [356, 226], [322, 181], [250, 164], [190, 181], [147, 227], [153, 283], [192, 336], [256, 360]]',
    # 'circle': '[[358, 257], [334, 324], [272, 359], [203, 347], [157, 292], [157, 222], [203, 168], [272, 155], [334, 191]]',
    'circle': '[[358, 257], [328, 330], [255, 360], [181, 330], [151, 257], [181, 184], [254, 154], [328, 184]]',
    # 'rect': '[[233, 317], [277, 317], [328, 317], [329, 253], [330, 193], [234, 194], [187, 195], [183, 254], [185, 318], [283, 192]]',
    'rect': '[[185, 318], [255, 317], [328, 317], [329, 253], [330, 193], [259, 194], [187, 195], [183, 254]]',
    # 'triangle': '[[255, 348], [218, 296], [173, 244], [139, 185], [299, 295], [340, 237], [375, 186], [316, 186], [259, 185], [200, 185]]',
    'triangle': '[[255, 348], [218, 296], [173, 244], [299, 295], [340, 237], [139, 185], [375, 186], [259, 185]]',
    'pyramid': '[[253, 357], [215, 305], [170, 253], [135, 195], [296, 304], [255, 253], [339, 253], [381, 195], [302, 194], [216, 194]]'
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
    sim_y = (y - REAL_OFFSET) / REAL_RADIUS * SIM_RADIUS
    return sim_x, sim_y


def sim2real_coord(x, y):
    real_x = x / SIM_RADIUS * REAL_RADIUS
    real_y = (y / SIM_RADIUS * REAL_RADIUS) + REAL_OFFSET
    return real_x, real_y
