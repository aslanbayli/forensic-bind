import json
import logging as log
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torch.utils.data import Dataset
from more_itertools import batched


#######################################
# VIDEO LOADING (OpenCV only)
#######################################

class VideoDataset(Dataset):
    def __init__(self, video_paths):
        self.video_paths = video_paths

    def __getitem__(self, idx: int):
        video_path = deepcopy(self.video_paths[idx])
        log.log(log.INFO, f"Processing {video_path}")
        frames, fps = self.load_video(video_path)
        return video_path, frames, fps

    def __len__(self):
        return len(self.video_paths)

    @classmethod
    def load_video(cls, video_path: Path):
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                log.error(f"Video {video_path} has no frames")
                return None, None

            return np.stack(frames), int(fps)

        except Exception as e:
            log.error(f"Failed to load {video_path}: {e}")
            return None, None


#######################################
# FACEFORENSICS DATASET
#######################################

class FaceForensics(VideoDataset):
    def __init__(self, root, target_root, methods, include_original, compression):
        self.root = Path(root)
        self.target_root = Path(target_root)
        self.target_root.mkdir(exist_ok=True, parents=True)
        self.methods = list(methods)
        self.include_original = include_original
        self.compression = compression

        if include_original and "original_sequences" not in self.methods:
            self.methods.append("original_sequences")

        self.video_paths = []
        for method in self.methods:
            if method == "original_sequences":
                method_root = self.root / "original_sequences"
            else:
                method_root = self.root / "manipulated_sequences" / method

            method_root = method_root / self.compression
            self.video_paths.extend(list(method_root.rglob("*.mp4")))

        # remove videos already processed
        self.video_paths = [
            p for p in self.video_paths
            if not (Path(str(p).replace(str(self.root), str(self.target_root))).exists())
        ]

    def __getitem__(self, idx):
        video_path = deepcopy(self.video_paths[idx])
        target_video_path = Path(str(video_path).replace(str(self.root), str(self.target_root)))
        boxes_path = target_video_path / "bboxes.json"
        lmks_path = target_video_path / "landmarks.json"
        frames, fps = self.load_video(video_path)
        return video_path, frames, fps, target_video_path, boxes_path, lmks_path


#######################################
# FACE DETECTOR (MTCNN)
#######################################

class FaceDetector:
    def __init__(self, device="cpu", batch_size=32, threshold=0.9, increment=0.1):
        self.device = device
        self.batch_size = batch_size
        self.detector = MTCNN(keep_all=True, device=device)

    @torch.no_grad()
    def __call__(self, frames):
        boxes, lmks, scores = [], [], []

        for frame in frames:
            b, p = self.detector.detect(frame)

            if b is None:
                boxes.append([])
                lmks.append([])
                scores.append([])
            else:
                boxes.append(b)
                lmks.append([])
                scores.append(p)

        return boxes, lmks, scores


#######################################
# SAVING RESULTS
#######################################

def save_results(
    faces, boxes, lmks, video_path, target_video_path,
    boxes_path, lmks_path, save_bbs, save_lmks,
    fps, ext=".png", quality=75, delete_orig=False, save_as="frames"
):
    target_video_path.mkdir(exist_ok=True, parents=True)

    if save_as == "frames":
        for i, frame_faces in enumerate(faces):
            for j, face in enumerate(frame_faces):
                face_name = f"{str(i).zfill(5)}_{j}{ext}"
                cv2.imwrite(
                    str(target_video_path / face_name),
                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                )
    else:
        raise ValueError("Invalid save_as option")

    if save_bbs:
        save_bbs_to_file(boxes_path, boxes)

    if save_lmks:
        save_lmks_to_file(lmks_path, lmks)

    if delete_orig:
        video_path.unlink()


def save_lmks_to_file(filename, lmks):
    out = {i: (lm.tolist() if not isinstance(lm, list) else lm) for i, lm in enumerate(lmks)}
    with open(filename, "w") as f:
        json.dump(out, f)


def save_bbs_to_file(filename, boxes):
    out = {i: (bb.tolist() if not isinstance(bb, list) else bb) for i, bb in enumerate(boxes)}
    with open(filename, "w") as f:
        json.dump(out, f, indent=4)


#######################################
# BBOX HELPERS
#######################################

def scale_bbox(bbox, height, width, scale_factor):
    left, top, right, bottom = bbox
    size_bb = int(max(right - left, bottom - top) * scale_factor)
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    left = max(int(center_x - size_bb // 2), 0)
    top = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - left, size_bb)
    size_bb = min(height - top, size_bb)
    return left, top, left + size_bb, top + size_bb


def apply_bbox(image, bbox, scale_factor=None):
    if not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox)
    bbox = bbox.astype(int)
    if scale_factor:
        bbox = scale_bbox(bbox, image.shape[0], image.shape[1], scale_factor)
    left, top, right, bottom = bbox
    return image[top:bottom, left:right, :]


def apply_bboxes(frames, bboxes, scale=None):
    out = []
    for i, bb_list in enumerate(bboxes):
        faces = []
        for bb in bb_list:
            faces.append(apply_bbox(frames[i], bb, scale_factor=scale))
        out.append(faces)
    return out


#######################################
# LOGGER
#######################################

def setup_logging(logdir: Path, args):
    os.makedirs(logdir, exist_ok=True)
    filename = os.path.join(logdir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    log.basicConfig(
        filename=filename,
        level=log.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
