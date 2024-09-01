from pathlib import Path

import cv2
import httpx
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

HAIR_SEGMENTER = "hair_segmentation"
MULTICLASS_SEGMENTER = "multiclass_segmentation"
FACE_LANDMARK_DETECTOR = "face_landmarks"

MODELS = {
  "hair_segmentation": {
      "URL": "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite",
      "path": Path("./models/hair_segmenter.tflite"),
      "img_size": 512,
      "masks": ["hair", "other"]
  },
  "multiclass_segmentation": {
      "URL": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
      "path": Path("./models/selfie_multiclass_256x256.tflite"),
      "img_size": 256,
      "masks": ["background", "hair", "body_skin", "face_skin", "clothes", "other"]
  },
  "face_landmarks": {
      "URL":  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      "path": Path("./models/face_landmarker_v2_with_blendshapes.task")
  }
}


SEGMENTER = None

def download_and_save_model(url, save_path):
    print(f"downloading: {save_path}")
    r = httpx.get(url, timeout=20)
    with open(save_path, "wb") as fh:
        fh.write(r.content)


def get_segmenter():
    if SEGMENTER is None:
        raise Exception("Segmenter not initialized")
    return SEGMENTER

# not using, but I want to keep it somewhere
def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img



class Segmenter:
    def __init__(self, segmenter_model_name, landmarker_model_name):
        global SEGMENTER
        self.__ensure_model_downloaded(segmenter_model_name)
        self.__ensure_model_downloaded(landmarker_model_name)
        seg_model_data = MODELS[segmenter_model_name]
        self._img_size = (seg_model_data["img_size"], seg_model_data["img_size"])
        self._masks = seg_model_data["masks"]
        segmenterOptions = vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(model_asset_path=seg_model_data["path"]),
            output_category_mask=True,
            output_confidence_masks=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(segmenterOptions)

        lm_model_data = MODELS[landmarker_model_name]
        faceLandmarkerOptions = vision.FaceLandmarkerOptions(
            base_options = python.BaseOptions(model_asset_path=lm_model_data["path"]),
            running_mode = vision.RunningMode.IMAGE
        )
        self._detector = vision.FaceLandmarker.create_from_options(faceLandmarkerOptions)

        SEGMENTER = self


    def resize_image(self, image):
        return cv2.resize(image, self._img_size)


    def segment_image(self, resized_image):
        ret_val = {}
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_image)
        segmentation_result = self._segmenter.segment(mp_img)
        categories = segmentation_result.category_mask.numpy_view() # img_size x img_size of 0 - num classes
        landmarks_mask = np.zeros(self._img_size, dtype=np.uint8)
        detection_result = self._detector.detect(mp_img)
        # landmarks are normalized (0-1) 3d ignore z coord and
        # multiply x * width and y * height
        width = self._img_size[0]
        height = self._img_size[1]
        # grab landmarks for 1st face only
        face_landmarks = detection_result.face_landmarks[0]
        for i, lm in enumerate(face_landmarks, start=1):
            x = np.floor(lm.x * width).astype(int)
            y = np.floor(lm.y * height).astype(int)
            print(f"{i}: ({lm.x}, {lm.y}) => ({x},{y})")
            landmarks_mask[x, y] = 1
        print(landmarks_mask)
        # convert categories img to N images of same shape of 0 or 1
        # mapped to their category name
        for cat_id, mask_name in enumerate(self._masks):
            ret_val[mask_name] = (categories == cat_id).astype(int)
        ret_val["landmarks"] = landmarks_mask
        return ret_val

    def __ensure_model_downloaded(self, model_name):
        if model_name not in MODELS:
            raise Exception(f"unknown segmenter: {model_name}")
        model_info = MODELS[model_name]
        if not model_info["path"].exists():
            download_and_save_model(model_info["URL"], model_info["path"])
