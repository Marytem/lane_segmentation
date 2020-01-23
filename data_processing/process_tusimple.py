import numpy as np
import cv2 as cv2
import json
from shutil import copyfile
import random


random.seed(230)

structure_path = 'data/tusimple/segm_structure/'
data_path = 'data/tusimple/train_set/'
labels = ['0601', '0531', '0313']
clips = []
ratio = 0.1

#
for label in labels:
    labe_filename = data_path + 'label_data_%s.json' % label
    clips += [json.loads(line) for line in open(labe_filename).readlines()]
random.shuffle(clips)

for clip, i in zip(clips, range(len(clips))):
    if i % 100 == 0:
        print(i)
    lanes = clip['lanes']
    filepath = clip['raw_file']
    ysamples = clip['h_samples']
    mode = 'train'
    if i / len(clips) > 1 - ratio:
        mode = 'val'
    lanes = [[(x, y) for (x, y) in zip(lane, ysamples) if x >= 0] for lane in lanes]

    copyfile(data_path + filepath,
             '{}{}_fr/{}/{}_frame_{}.jpg'.format(structure_path, mode, mode, mode, i))

    label_image = np.zeros(raw_image.shape[:2], dtype=np.uint8)
    for lane in lanes:
        cv2.polylines(label_image, np.int32([lane]), isClosed=False, color=(255, 255, 255),
                      thickness=5)
    cv2.imwrite('{}{}_masks/{}/{}_mask_{}.png'.format(structure_path, mode, mode, mode, i),
                label_image)