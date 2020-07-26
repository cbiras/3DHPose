import os
import json
import torch
import pprint
import argparse
import importlib
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")


from backend.ExtremeNet.nnet.py_factory import NetworkFactory

from backend.ExtremeNet.configs import system_configs
from backend.ExtremeNet.utils import crop_image, normalize_
from backend.ExtremeNet.external.nms import soft_nms_with_points as soft_nms
from backend.ExtremeNet.lib.detection_opr.box_utils.box import DetBox

torch.backends.cudnn.benchmark = False

class_name = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
classes_originID = {
        'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
        'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9,
        'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
        'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17,
        'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22,
        'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
        'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33,
        'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37,
        'kite': 38, 'baseball bat': 39, 'baseball glove': 40,
        'skateboard': 41, 'surfboard': 42, 'tennis racket': 43,
        'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48,
        'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53,
        'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57,
        'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61,
        'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65,
        'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73,
        'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77,
        'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81,
        'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86,
        'scissors': 87, 'teddy bear': 88, 'hair drier': 89,
        'toothbrush': 90}

image_ext = ['jpg', 'jpeg', 'png', 'webp']


def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs /= ratios[:, 1][:, None, None]
    ys /= ratios[:, 0][:, None, None]
    xs -= borders[:, 2][:, None, None]
    ys -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)


def _rescale_ex_pts(detections, ratios, borders, sizes):
    xs, ys = detections[..., 5:13:2], detections[..., 6:13:2]
    xs /= ratios[:, 1][:, None, None]
    ys /= ratios[:, 0][:, None, None]
    xs -= borders[:, 2][:, None, None]
    ys -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)


def _box_inside(box2, box1):
    inside = (box2[0] >= box1[0] and box2[1] >= box1[1] and \
              box2[2] <= box1[2] and box2[3] <= box1[3])
    return inside


def kp_decode(nnet, images, K, kernel=3, aggr_weight=0.1,
              scores_thresh=0.1, center_thresh=0.1, debug=False):
    detections = nnet.test(
        [images], kernel=kernel, aggr_weight=aggr_weight,
        scores_thresh=scores_thresh, center_thresh=center_thresh, debug=debug)
    detections = detections.data.cpu().numpy()
    return detections

class PersonDetector ( object ):
    def __init__(self):
        #preparare config file
        self.cfg_file = os.path.join(
        system_configs.config_dir, "ExtremeNet.json")
        with open(self.cfg_file, "r") as f:
            self.configs = json.load(f)

        self.configs["system"]["snapshot_name"] = 'ExtremeNet'
        system_configs.update_config(self.configs["system"])

        #preparare model
        self.nnet = NetworkFactory(None)

        self.nnet.load_pretrained_params('backend/ExtremeNet/cache/ExtremeNet_250000.pkl')
        self.nnet.cuda()
        self.nnet.eval_mode()

        #parametrii
        self.test_save_type = self.configs["db"]["test_save_type"]
        self.K = self.configs["db"]["top_k"]
        self.aggr_weight = self.configs["db"]["aggr_weight"]
        self.scores_thresh = self.configs["db"]["scores_thresh"]
        self.center_thresh = self.configs["db"]["center_thresh"]
        self.suppres_ghost = True
        self.nms_kernel = 3

        self.scales = self.configs["db"]["test_scales"]
        self.weight_exp = 8
        self.categories = self.configs["db"]["categories"]
        self.nms_threshold = self.configs["db"]["nms_threshold"]
        self.max_per_image = self.configs["db"]["max_per_image"]
        self.nms_algorithm = {
            "nms": 0,
            "linear_soft_nms": 1,
            "exp_soft_nms": 2
        }["exp_soft_nms"]

        self.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)



    def detect(self,image, image_id):

        all_results = []
        data_dict = dict(data=image, image_id=image_id)
        result_dict = self.inference(data_dict)
        all_results.append(result_dict)

        return self._make_result_dict(all_results)

    def inference(self,data_dict):
        top_bboxes = {}
        image = data_dict['data']
        image_id = data_dict['image_id']

        height, width = image.shape[0:2]
        detections = []

        for scale in self.scales:
            new_height = int(height * scale)
            new_width = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])

            inp_height = new_height | 127
            inp_width = new_width | 127

            images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes = np.zeros((1, 2), dtype=np.float32)

            out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            height_ratio = out_height / inp_height
            width_ratio = out_width / inp_width

            resized_image = cv2.resize(image, (new_width, new_height))
            resized_image, border, offset = crop_image(
                resized_image, new_center, [inp_height, inp_width])

            resized_image = resized_image / 255.
            normalize_(resized_image, self.mean, self.std)

            images[0] = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0] = [int(height * scale), int(width * scale)]
            ratios[0] = [height_ratio, width_ratio]

            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            images = torch.from_numpy(images)
            dets = kp_decode(
                self.nnet, images, self.K, aggr_weight=self.aggr_weight,
                scores_thresh=self.scores_thresh, center_thresh=self.center_thresh,
                kernel=self.nms_kernel, debug=False)
            dets = dets.reshape(2, -1, 14)
            dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
            dets[1, :, [5, 7, 9, 11]] = out_width - dets[1, :, [5, 7, 9, 11]]
            dets[1, :, [7, 8, 11, 12]] = dets[1, :, [11, 12, 7, 8]].copy()
            dets = dets.reshape(1, -1, 14)

            _rescale_dets(dets, ratios, borders, sizes)
            _rescale_ex_pts(dets, ratios, borders, sizes)
            dets[:, :, 0:4] /= scale
            dets[:, :, 5:13] /= scale
            detections.append(dets)

        detections = np.concatenate(detections, axis=1)

        classes = detections[..., -1]
        classes = classes[0]
        detections = detections[0]

        # reject detections with negative scores
        keep_inds = (detections[:, 4] > 0)
        detections = detections[keep_inds]
        classes = classes[keep_inds]

        top_bboxes[image_id] = {}
        for j in range(self.categories):
            keep_inds = (classes == j)
            top_bboxes[image_id][j + 1] = \
                detections[keep_inds].astype(np.float32)
            soft_nms(top_bboxes[image_id][j + 1],
                     Nt=self.nms_threshold, method=self.nms_algorithm)

        scores = np.hstack([
            top_bboxes[image_id][j][:, 4]
            for j in range(1, self.categories + 1)
        ])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, 4] >= thresh)
                top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]

        if self.suppres_ghost:
            for j in range(1, self.categories + 1):
                n = len(top_bboxes[image_id][j])
                for k in range(n):
                    inside_score = 0
                    if top_bboxes[image_id][j][k, 4] > 0.2:
                        for t in range(n):
                            if _box_inside(top_bboxes[image_id][j][t],
                                           top_bboxes[image_id][j][k]):
                                inside_score += top_bboxes[image_id][j][t, 4]
                        if inside_score > top_bboxes[image_id][j][k, 4] * 3:
                            top_bboxes[image_id][j][k, 4] /= 2
        result_boxes = []
        for j in range(1, self.categories + 1):
            keep_inds = (top_bboxes[image_id][j][:, 4] > 0.5)
            cat_name = class_name[j]
            for bbox in top_bboxes[image_id][j][keep_inds]:
                bbox = bbox[0:4].astype(np.int32)
                dbox = DetBox (
                    bbox[0], bbox[1],
                    bbox[2] - bbox[0], bbox[3] - bbox[1],
                    tag=cat_name, score=bbox[-1] )
                result_boxes.append(dbox)

        result_dict = data_dict.copy()
        result_dict['result_boxes'] = result_boxes
        return result_dict

    def _make_result_dict(self, all_results):
        coco_records = []

        for result in all_results:
            result_boxes = result['result_boxes']
            if self.test_save_type == "coco":
                image_id = int(result['image_id'])
                for rb in result_boxes:
                    if rb.tag == 'person':
                        record = {'image_id': image_id, 'category_id': classes_originID[rb.tag],
                                  'score': rb.score, 'bbox': [rb.x, rb.y, rb.w, rb.h], 'data': result['data']}
                        coco_records.append(record)
            else:
                raise Exception(
                    "Unimplemented save type: " + str(config.test_save_type))
        return coco_records

if __name__ == '__main__':
    detector = PersonDetector ( )
    img = cv2.imread ( './images/16004479832_a748d55f21_k.jpg' )
    bbox = detector.detect ( img, 0 )
    print ( str(bbox[1:-1]) )
