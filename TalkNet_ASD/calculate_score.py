import os
import cv2
import json
import numpy as np
import torch

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.core import encode_mask_results
from mmdet.core.visualization import imshow_det_bboxes
from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
from mmcv.runner import load_checkpoint
from tqdm import tqdm

config_file = '/home/zqr/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/home/zqr/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

def test(model, imgs):

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results

def get_bbox_data(path, annotation):
    model = init_detector(
        config_file, checkpoint_file, device='cuda:0')
    checkpoint = load_checkpoint(model, checkpoint_file)
    if 'CLASSES' in checkpoint.get('meta', {}):
        classes = checkpoint['meta']['CLASSES']
        map_class = {i: v for i, v in enumerate(classes)}

    new_annotation = dict()
    for k, v in annotation.items():
        k_ = k.split('_')
        img_path = os.path.join(path, '_'.join(k_[0:3]), 'pyframes', str('%06d' % int(k_[3])) + '.jpg')
        print(img_path)
    
        result = test(model, img_path)
        dets = []
        for idx, elem in enumerate(result):
            label = map_class[idx]
            if label == 'person':
                for bbox in elem:
                    score = bbox[-1]
                    if score < 0.8:
                        continue
                    dets.append({'fname':img_path, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
        new_annotation[k] = dets[int(v['id'])]
    return new_annotation
    #bbox: [lh, lw, rh, rw]        

def bb_intersection_over_union(boxA, boxB, evalCol = False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

annotation_path = '/home/sharing/disk3/zhanghanlei/Datasets/MIntRec/private/speaker_annotation/human/speaker_annotations.json'
data_path = '/home/sharing/disk3/zhanghanlei/Datasets/MIntRec/private/speaker_annotation/Talknet'
annotation = json.load(open(annotation_path, 'r'))

missed = 0
total = 0 
wrong = 0
for k, v in tqdm(annotation.items(), desc = 'Progress'):
    bbox = v['bbox']
    k_ = k.split('_')
    frame = int(k_[3])
    data_id = '_'.join(k_[0:3])
    best_persons_path = os.path.join(data_path, data_id, 'pywork', 'best_persons.npy')
    best_persons = np.load(best_persons_path)
    for kk, vv in annotation.items():
        kk_ = kk.split('_')
        if '_'.join(kk_[0:3]) == data_id:
            bbox = vv['bbox']
            frame = int(kk_[3])
            break
    total += 1
    if best_persons[frame][0] == 0 and best_persons[frame][1] == 0 and best_persons[frame][2] == 0 and best_persons[frame][3] == 0:
        missed += 1
        continue
    iou = bb_intersection_over_union(best_persons[frame], bbox[:-1])
    if iou < 0.9:
        wrong += 1

missing_rate = missed / total
hit_ratio = (total - wrong) / total
print('Missing rate is {}'.format(missing_rate))
print('Hit ratio is {}'.format(hit_ratio))



    
