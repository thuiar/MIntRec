from genericpath import exists
import sys
import time
import os
from numpy.lib.type_check import imag
import tqdm
import torch
import argparse
import glob
import subprocess
import warnings
import cv2
import pickle
import pdb
import math
import python_speech_features
import json

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet

import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.core import encode_mask_results
from mmdet.core.visualization import imshow_det_bboxes
from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
from mmcv.runner import load_checkpoint


def process_video(read_file_path, output_file_path):

    cap = cv2.VideoCapture(read_file_path)

    if not cap.isOpened():
        print('The directory is wrong.')
    
    cnt = 0
    while True:

        ret, frame = cap.read()
        if frame is None:
            break
        
        write_path = os.path.join(output_file_path, str('%06d' % cnt) + '.jpg')
        
        cv2.imwrite(write_path, frame)
        cnt += 1

        if not ret:
            break
    
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

def get_bbox_data(args):
    model = init_detector(
        args.config_file, args.checkpoint_file, device='cuda:0')
    checkpoint = load_checkpoint(model, args.checkpoint_file)
    if 'CLASSES' in checkpoint.get('meta', {}):
        classes = checkpoint['meta']['CLASSES']
        map_class = {i: v for i, v in enumerate(classes)}

    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        result = test(model, fname)
        dets.append([])
        for idx, elem in enumerate(result):
            label = map_class[idx]
            if label == 'person':
                for bbox in elem:
                    score = bbox[-1]
                    if score < 0.8:
                        continue
                    dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
        #bbox: [lh, lw, rh, rw]        
    return dets

def scene_detect(args):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source = videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(args.pyworkPath, 'scene.pckl')
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
    
    return sceneList

def inference_video(args, persons):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device='cuda')
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        dets.append([])
        for person in persons[fidx]:
            personBbox = person['bbox']
            if personBbox[3] - personBbox[1] < 150 or personBbox[2] - personBbox[0] < 150:
                continue
            personImage = imageNumpy[int(personBbox[1]) : int(personBbox[3]), int(personBbox[0]) : int(personBbox[2])]
            
            bboxes = DET.detect_faces(personImage, conf_th=0.8, scales=[args.facedetScale])
            idx = -1
            s = -1
            for i, bbox in enumerate(bboxes):
                if bbox[4] > s:
                    idx = i
                    s = bbox[4]
            if idx == -1:
                continue
            bbox = bboxes[idx]
            dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1] + [personBbox[0], personBbox[1], personBbox[0], personBbox[1]]).tolist(), 
                                'person_bbox': personBbox, 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
        sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
    
    return dets

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

def track_shot(args, sceneFaces):
    # CPU: Face tracking
    iouThres  = 0.5     # Minimum IOU between consecutive face detections
    tracks    = []
    while True:
        track     = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum    = np.array([ f['frame'] for f in track ])
            bboxes      = np.array([np.array(f['bbox']) for f in track])
            personBboxes = np.array([np.array(f['person_bbox']) for f in track])
            frameI      = np.arange(frameNum[0],frameNum[-1]+1)
            bboxesI    = []
            personBboxesI = []
            for ij in range(0,4):
                interpfn  = interp1d(frameNum, bboxes[:,ij])
                bboxesI.append(interpfn(frameI))
                personInterpfn = interp1d(frameNum, personBboxes[:,ij])
                personBboxesI.append(personInterpfn(frameI))
            bboxesI  = np.stack(bboxesI, axis=1)
            personBboxesI = np.stack(personBboxesI, axis=1)
            print('track')
            print(track)
            if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
                tracks.append({'frame':frameI,'bbox':bboxesI, 'person_bbox': personBboxesI})
    
    return tracks

def crop_video(args, track, cropFile):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
    dets = {'x':[], 'y':[], 's':[]}
    personDets = {'x':[], 'y':[], 's':[]}
    for det in track['bbox']: # Read the tracks
        dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
        dets['y'].append((det[1]+det[3])/2) # crop center x 
        dets['x'].append((det[0]+det[2])/2) # crop center y
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for det in track['person_bbox']: # Read the tracks
        personDets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
        personDets['y'].append((det[1]+det[3])/2) # crop center x 
        personDets['x'].append((det[0]+det[2])/2) # crop center y
    personDets['s'] = signal.medfilt(personDets['s'], kernel_size=13)  # Smooth detections 
    personDets['x'] = signal.medfilt(personDets['x'], kernel_size=13)
    personDets['y'] = signal.medfilt(personDets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs  = args.cropScale
        bs  = dets['s'][fidx]   # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
        image = cv2.imread(flist[frame])
        frame = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my  = dets['y'][fidx] + bsi  # BBox center Y
        mx  = dets['x'][fidx] + bsi  # BBox center X
        face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        
        
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp    = cropFile + '.wav'
    audioStart  = (track['frame'][0]) / 25
    audioEnd    = (track['frame'][-1]+1) / 25
    vOut.release()
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
              (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
    output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
              (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track':track, 'proc_track':dets, 'person_proc_track': personDets}

def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
    np.save(featuresPath, mfcc)

def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained TalkNet
    s = talkNet()
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
    s.eval()
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
    for file in tqdm.tqdm(files, total = len(files)):
        fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224,224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = np.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
        audioFeature = audioFeature[:int(round(length * 100)),:]
        videoFeature = videoFeature[:int(round(length * 25)),:,:]
        allScore = [] # Evaluation use TalkNet
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)	
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels = None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = np.round((np.mean(np.array(allScore), axis = 0)), 1).astype(float)
        allScores.append(allScore)	
    return allScores

def visualization(tracks, scores, args):
    # CPU: visulize the result for video format
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
            s = np.mean(s)
            faces[frame].append({'track':tidx, 'score':float(s),'s':track['person_proc_track']['s'][fidx], 'x':track['person_proc_track']['x'][fidx], 'y':track['person_proc_track']['y'][fidx],
                        's_':track['proc_track']['s'][fidx], 'x_':track['proc_track']['x'][fidx], 'y_':track['proc_track']['y'][fidx]})
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
    colorDict = {0: 0, 1: 255}
    best_persons = []
    for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
        image = cv2.imread(fname)
        best_face_idx = -1
        draw_face_idx = -1
        best_face_score = -100
        for i, face in enumerate(faces[fidx]):
            if face['score'] > best_face_score:
                draw_face_idx = best_face_idx
                best_face_idx = i
                best_face_score = face['score']
            else:
                draw_face_idx = i

            if draw_face_idx == -1:
                continue
            draw_face = faces[fidx][draw_face_idx]
            clr = colorDict[0]
            #txt = round(draw_face['score'], 1)
            cv2.rectangle(image, (int(draw_face['x']-draw_face['s']), int(draw_face['y']-draw_face['s'])), (int(draw_face['x']+draw_face['s']), int(draw_face['y']+draw_face['s'])),(0,255,255),10)
            cv2.rectangle(image, (int(draw_face['x_']-draw_face['s_']), int(draw_face['y_']-draw_face['s_'])), (int(draw_face['x_']+draw_face['s_']), int(draw_face['y_']+draw_face['s_'])),(0,125,255),10)
            #cv2.putText(image,'%s'%(txt), (int(draw_face['x']-draw_face['s']), int(draw_face['y']-draw_face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
        
        if best_face_idx != -1:
            face = faces[fidx][best_face_idx]
            clr = colorDict[1]
            #txt = round(face['score'], 1)
            cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,255,255), 10)
            cv2.rectangle(image, (int(face['x_']-face['s_']), int(face['y_']-face['s_'])), (int(face['x_']+face['s_']), int(face['y_']+face['s_'])),(0,125,255),10)
            #cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
        
            best_track_idx = faces[fidx][best_face_idx]['track']
            best_frame_idx = -1
            for idx, frame in enumerate(tracks[best_track_idx]['track']['frame'].tolist()):
                if frame == fidx:
                    best_frame_idx = idx
            best_persons.append(tracks[best_track_idx]['track']['person_bbox'][best_frame_idx])
        else:
            best_persons.append([0, 0, 0, 0])
        
        write_path = os.path.join('detect', str('%06d' % fidx) + '.jpg')
        cv2.imwrite(write_path, image)

        vOut.write(image)
    vOut.release()
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
        (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
        args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
    output = subprocess.call(command, shell=True, stdout=None)

    print(best_persons)
    return best_persons


# Main function
def main(args):
    # This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
    # ```
    # .
    # ├── pyavi
    # │   ├── audio.wav (Audio from input video)
    # │   ├── video.avi (Copy of the input video)
    # │   ├── video_only.avi (Output video without audio)
    # │   └── video_out.avi  (Output video with audio)
    # ├── pycrop (The detected face videos and audios)
    # │   ├── 000000.avi
    # │   ├── 000000.wav
    # │   ├── 000001.avi
    # │   ├── 000001.wav
    # │   └── ...
    # ├── pyframes (All the video frames in this video)
    # │   ├── 000001.jpg
    # │   ├── 000002.jpg
    # │   └── ...	
    # └── pywork
    #     ├── faces.pckl (face detection result)
    #     ├── scene.pckl (scene detection result)
    #     ├── scores.pckl (ASD result)
    #     └── tracks.pckl (face tracking result)
    # ```

    # Initialization 

    if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
        
        Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
        cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
        subprocess.call(cmd, shell=True, stdout=None)


    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')
    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
    os.makedirs(args.pyframesPath, exist_ok = True) # Save all the video frames
    os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
    os.makedirs(args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process

    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
        (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))

    # Extract the video frames
    process_video(args.videoFilePath, args.pyframesPath)

    # Scene detection for the video frames
    scene = scene_detect(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))	

    # TODO: person detection with mmdetection 
    persons = get_bbox_data(args)
    
    # Face detection for the video frames
    faces = inference_video(args, persons)
    
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))

    # Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack: # Discard the shot frames less than minTrack frames
            allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))
    print(allTracks)
    
    # Face clips cropping
    for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d'%ii)))
    
    
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %args.pycropPath)
    

    # Active Speaker Detection by TalkNet
    files = glob.glob("%s/*.avi"%args.pycropPath)
    files.sort()
    scores = evaluate_network(files, args)
    print(scores)
    
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)

    # Visualization, save the result as the new video	
    best_persons = visualization(vidTracks, scores, args)
    np.save(os.path.join(args.pyworkPath, 'best_persons.npy'), np.array(best_persons))
    return best_persons

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description = "TalkNet Demo or Columnbia ASD Evaluation")

    parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')
    parser.add_argument('--saveFolder',            type=str, default="/home/zhanghanlei/detect_speaker/",  help='Path for inputs, tmps and outputs')
    parser.add_argument('--dataPath',            type=str, default="/home/sharing/disk1/zhanghanlei/Datasets/private/raw_data",  help='Path for inputs, tmps and outputs')


    parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
    parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
    parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
    parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
    parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
    parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

    parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columnbia dataset')
    parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
    parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

    parser.add_argument('--colSavePath',           type=str, default="/data08/col",  help='Path for inputs, tmps and outputs')
    parser.add_argument('--config_file', type=str, default='/home/sharing/disk1/zhanghanlei/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', help="The detection configuration file.")
    parser.add_argument('--checkpoint_file', type=str, default='/home/sharing/disk1/zhanghanlei/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', help="The detection checkpoint file.")


    args = parser.parse_args()
  
    process_list = ['S04_E01_29']
    
    args.dataPath = '/home/sharing/Datasets/MIntRec/raw_data'
    args.saveFolder = '/home/sharing/Datasets/MIntRec-finegrained_feats/Talknet'
    # for file in process_list:
    #     file_list = file.split('_')
    #     args.videoFilePath = os.path.join(args.dataPath, file_list[0], file_list[1], file_list[2] + '.mp4')
    #     os.makedirs(file, exist_ok=True)
    #     args.savePath = file
    #     print(main(args))
    # load_path = os.path.join(args.saveFolder, 'S04_E01_42', 'pywork', 'best_persons.npy')
    # save_numpy = np.load(load_path)
    # print('11111111', save_numpy)
    # if not os.path.exists(args.saveFolder):
    #     os.makedirs(args.saveFolder)
    
    for s_path in os.listdir(args.dataPath):
        
        s_path_dir = os.path.join(args.dataPath, s_path)

        for e_path in os.listdir(s_path_dir):
            e_path_dir = os.path.join(s_path_dir, e_path)
            
            for file in os.listdir(e_path_dir):
                video_clip = str(s_path) + '_' + str(e_path) + '_' + str(file)[:-4]
                file_list = video_clip.split('_')
                if video_clip not in process_list:
                    continue

                args.videoFilePath = os.path.join(args.dataPath, file_list[0], file_list[1], file_list[2] + '.mp4')
                args.savePath = os.path.join(args.saveFolder, video_clip)
                print(args.savePath)
                if os.path.exists(args.savePath):
                    continue
                
                os.makedirs(args.savePath, exist_ok=True)

                try: 
                    print(main(args))
                except Exception as e:
                    print(e)
