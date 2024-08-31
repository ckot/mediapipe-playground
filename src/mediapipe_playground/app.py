# import os

import numpy as np
import gradio as gr

from mediapipe_playground.utils import get_segmenter, MULTICLASS_SEGMENTER, Segmenter

BACKGROUND = 0
HAIR = 1
BODY_SKIN  = 2
FACE_SKIN = 3
CLOTHES = 4
OTHER = 5 # (accessories)

BG_COLOR = (192, 192, 192) # gray
HAIR_COLOR = (255, 255, 0) # yellow
BODY_COLOR = (255, 153, 51) # orange
FACE_COLOR = (255, 204, 153) # pastel orange
CLOTHES_COLOR = (0, 128, 255) # blue
OTHER_COLOR = (153,  51, 255) # purple
MASK_COLOR = (255, 255, 255) # white
COLORS = [
    BG_COLOR,
    HAIR_COLOR,
    BODY_COLOR,
    FACE_COLOR,
    CLOTHES_COLOR,
    OTHER_COLOR
]

COLORS_BY_NAME = {
    "background": BG_COLOR,
    "hair": HAIR_COLOR,
    "body_skin": BODY_COLOR,
    "face_skin": FACE_COLOR,
    "clothes": CLOTHES_COLOR,
    "other": OTHER_COLOR
}

def segment_image(input_img):
    seg = get_segmenter()
    resized_img = seg.resize_image(input_img)
    masks = seg.segment_image(resized_img)
    zeros = np.zeros(resized_img.shape, dtype=np.uint8)
    output_img = resized_img.copy()
    for mask_name, mask in masks.items():
        color_mask = zeros.copy()
        color_mask[:] = COLORS_BY_NAME[mask_name]
        cond = np.stack((mask,) * 3, axis=-1) == 1
        output_img = np.where(cond, color_mask, output_img)
    return output_img


def app():
    Segmenter(MULTICLASS_SEGMENTER)
    demo = gr.Interface(segment_image, gr.Image(height=256, width=256), "image")
    demo.launch()