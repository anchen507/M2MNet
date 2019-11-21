
import PIL.Image as Image
import random
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import argparse
import glob
from PIL import Image
import PIL
import random
import gc
import os
import sys
import numpy as np
import tensorflow as tf
parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', dest='save_dir', default='./data', help='dir of patches')
parser.add_argument('--is_color', dest='is_color', type=int, default=1, help='color:1, grayscale:0')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=50, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=50, help='stride')
parser.add_argument('--step', dest='step', type=int, default=50, help='step')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=16, help='batch size')
parser.add_argument('--clip_length ', dest='clip_length', type=int, default= 4, help='CLIP_LENGTH')
parser.add_argument('--from_file', dest='from_file', default="data/color_img_clean_pats.npy", help='get pic from file')
args = parser.parse_args()
TRAIN_LIST_PATH = 'data/train.list'
DATA_PATH='data/'
def get_video_indices(filename):
    lines = open(filename, 'r')
    #Shuffle data
    lines = list(lines)
    video_indices = list(range(len(lines)))
    random.seed(time.time())
    random.shuffle(video_indices)
    return video_indices

def frame_process(clip, x,y, channel_num):
    frames_num = len(clip)
    croped_frames = np.zeros([frames_num, args.pat_size, args.pat_size, channel_num]).astype(np.float32)
    for i in range(frames_num):
        img = clip[i]
        img_crop = img[x:x + args.pat_size, y:y + args.pat_size, :]
        if img_crop.shape[0]==args.pat_size and img_crop.shape[1] == args.pat_size:
            croped_frames[i, :, :, :] = img_crop
    return croped_frames


def get_first_image(filename):
    for parent, dirnames, filenames in os.walk(filename):
        image_name = str(filename) + '/' + str(filenames[0])
    return image_name

def convert_images_to_clip(filename, clip_length,x,y,is_color):
    clip = []
    if is_color:
       channel_num = 3
    else:
       channel_num = 1
    
    for parent, dirnames, filenames in os.walk(filename):
        filenames = sorted(filenames)
        if len(filenames) < clip_length:
            for i in range(0, len(filenames)):
                image_name = str(filename) + '/' + str(filenames[i])
                if is_color:
                   img = Image.open(image_name).convert("RGB")
                else:
                   img = Image.open(image_name).convert("L")
                img_data =  np.reshape(np.array(img, dtype="uint8"), (img.size[1], img.size[0], channel_num))
                clip.append(img_data)
            for i in range(clip_length - len(filenames)):
                image_name = str(filename) + '/' + str(filenames[len(filenames) - 1])
                if is_color:
                   img = Image.open(image_name).convert("RGB")
                else:
                   img = Image.open(image_name).convert("L")

                img_data =  np.reshape(np.array(img, dtype="uint8"), (img.size[1], img.size[0], channel_num))
                clip.append(img_data)
        else:
            s_index = random.randint(0, len(filenames) - clip_length)
            for i in range(s_index, s_index + clip_length):
                image_name = str(filename) + '/' + str(filenames[i])
                if is_color:
                   img = Image.open(image_name).convert("RGB")
                else:
                   img = Image.open(image_name).convert("L")
                img_data = np.reshape(np.array(img, dtype="uint8"),(img.size[1], img.size[0], channel_num))
                clip.append(img_data)
    if len(clip) == 0:
       print(filename)
    clip = frame_process(clip, x,y,channel_num)
    return clip


train_data = get_video_indices(TRAIN_LIST_PATH)
count = 0
print("number of training data %d" % len(train_data))
lines = open(TRAIN_LIST_PATH, 'r')
lines = list(lines)

clips = []
for i in range(len(train_data)):
    line = lines[i].strip('\n').split()
    dirname = DATA_PATH+ line[0]
    image_name = get_first_image(dirname)
    if args.is_color:
       img = Image.open(image_name).convert("RGB")
    else:
       img = Image.open(image_name).convert("L")
    im_w, im_h = img.size
    print("processing image: %d" %i)
    for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
        for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
            i_clip = convert_images_to_clip(dirname,args.clip_length,x,y,args.is_color)
            clips.append(i_clip)
            count += 1
inputs = np.array(clips).astype(np.float32)/255.0
print(inputs.shape)
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
np.save(args.from_file, inputs)

