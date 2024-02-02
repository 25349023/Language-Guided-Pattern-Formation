import ast
import random
import re

import numpy as np

lm_pattern = re.compile(r'''\[(\(('.*?'|".*?"),\s*\[\d*,\s*\d*,\s*\d*,\s*\d*\]\)(,\s*)?)+\]''')

SIM_RADIUS = 3.0  # simulator coordinates range from (-3, -3) to (3, 3)
REAL_RADIUS = 170.0  # limited by the y-axis of the table
REAL_OFFSET = 85.0


def get_landmarks(num):
    def remap(v, offset, reverse=False):
        new_range = SIM_RADIUS - boundary
        pos_at_origin = (v - offset) / scale * new_range * 2

        if reverse:
            return -pos_at_origin
        else:
            return pos_at_origin

    # for testing, it is a circle
    prompt = """[('a cherry', [246, 206, 20, 20]), ('a cherry', [266, 206, 20, 20]), ('a cherry', [286, 206, 20, 20]), ('a cherry', [306, 206, 20, 20]), ('a cherry', [326, 206, 20, 20]), ('a cherry', [226, 226, 20, 20]), ('a cherry', [346, 226, 20, 20]), ('a cherry', [206, 246, 20, 20]), ('a cherry', [366, 246, 20, 20]), ('a cherry', [206, 266, 20, 20]), ('a cherry', [366, 266, 20, 20]), ('a cherry', [206, 286, 20, 20]), ('a cherry', [366, 286, 20, 20]), ('a cherry', [226, 306, 20, 20]), ('a cherry', [346, 306, 20, 20]), ('a cherry', [246, 326, 20, 20]), ('a cherry', [266, 326, 20, 20]), ('a cherry', [286, 326, 20, 20]), ('a cherry', [306, 326, 20, 20]), ('a cherry', [326, 326, 20, 20])]"""

    # prompt = ''
    # while not lm_pattern.match(prompt):
    #     prompt = input('please input the landmark settings: ')

    landmarks = ast.literal_eval(prompt)
    landmarks = [(x + w / 2, y + h / 2) for obj, (x, y, w, h) in landmarks]
    min_x, max_x = min(x for x, _ in landmarks), max(x for x, _ in landmarks)
    min_y, max_y = min(y for _, y in landmarks), max(y for _, y in landmarks)
    mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2

    scale = max(max_y - min_y, max_x - min_x)
    boundary = 0.4
    landmarks = [(remap(x, mid_x), remap(y, mid_y, True)) for x, y in landmarks]

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
