import ast
import itertools
import pathlib
import random
import re

import clip
import numpy as np
import torch
from PIL import Image

template = 'Some black boxes arranged in the shape of {}'

NEG_SAMPLES = [
    'random', 'circle', 'square', 'triangle', 'hexagon', 'star', 'Trapezium', 'diamond', 'kite',
    *[f'capital letter {chr(i)}' for i in range(ord('A'), ord('Z') + 1)],
    'right triangle', 'pyramid', 'cross', 'drop', 'grid', 'rhombus'
]


def convert2img(coordinates):
    raw_img = np.ones((512, 512), dtype=np.uint8) * 255
    for x, y in coordinates:
        y = 512 - y
        lx, ly = max(x - 10, 0), max(y - 10, 0)
        rx, ry = min(x + 10, 512), min(y + 10, 512)
        raw_img[ly:ry, lx:rx] = 0

    image = Image.fromarray(raw_img)
    return image


def clip_score(clip_model, prompt, raw_image):
    model, preprocess = clip_model
    image = preprocess(raw_image).unsqueeze(0).to(device)

    word_bank_without_prompt = [ns for ns in NEG_SAMPLES if ns.lower() not in prompt.lower()]
    neg_samples = random.sample(word_bank_without_prompt, 4)
    raw_text = [template.format(sh) for sh in [prompt] + neg_samples]

    text = clip.tokenize(raw_text).to(device)

    with torch.no_grad():
        # image_features = model.encode_image(image)
        # text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    for i in probs.argsort()[::-1]:
        print(f'{probs[i]:.4f} <== {raw_text[i]}')


if __name__ == '__main__':
    lm_pattern = re.compile(r'''\[(\[\d*,\s*\d*\](,\s*)?)+\]''')

    pathlib.Path('gpt_eval').mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_args = clip.load("ViT-L/14@336px", device=device)

    num = 0
    while True:
        coord_str = input("Input coord: ").strip()

        if not lm_pattern.match(coord_str):
            continue

        coord = ast.literal_eval(coord_str)

        img = convert2img(coord)

        prompt = input("Input original prompt: ").strip()
        clip_score(clip_args, prompt, img)

        # with open(f'output/coord_{num:03}.txt', 'w') as f:
        #     f.write(coord_str)

        img.save(f'gpt_eval/{num:03}-{prompt}.png')
        num += 1
        print()
