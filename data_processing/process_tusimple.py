import numpy as np
import cv2 as cv2
import json
from shutil import copyfile
import random

#  fix dividing
random.seed(230)

structure_path = '../data/'
train_data_path = '../unprocessed_data/TuSimple/train_set/'
test_data_path = '../unprocessed_data/TuSimple/test_set/'
train_labels = ['0601', '0531', '0313']
train_clips = []
ratio = 0.1

for label in train_labels:
    labe_filename = train_data_path + 'label_data_%s.json' % label
    train_clips += [json.loads(line) for line in open(labe_filename).readlines()]
random.shuffle(train_clips)

test_labe_filename = '../unprocessed_data/TuSimple/test_label.json'
test_clips = [json.loads(line) for line in open(test_labe_filename).readlines()]

for data_path, clips in zip((test_data_path, train_data_path), (test_clips, train_clips)):
    for clip, i in zip(clips, range(3, len(test_clips))):

        # progress
        if i % 100 == 0:
            print(i)

        lanes = clip['lanes']
        filepath = clip['raw_file']
        ysamples = clip['h_samples']
        mode = 'train'
        if i / len(clips) > 1 - ratio:
            mode = 'val'

        copyfile(data_path + filepath,
                 '{}{}_fr/{}/{}_frame_{}.jpg'.format(structure_path, mode, mode, mode, i))

        lanes = [[(x, y) for (x, y) in zip(lane, ysamples) if x >= 0] for lane in lanes]
        raw_image = cv2.imread(data_path + filepath)
        label_image = np.zeros(raw_image.shape[:2], dtype=np.uint8)
        for lane in lanes:
            cv2.polylines(label_image, np.int32([lane]), isClosed=False, color=(255, 255, 255),
                          thickness=5)
        cv2.imwrite('{}{}_masks/{}/{}_mask_{}.png'.format(structure_path, mode, mode, mode, i),
                    label_image)

    break
