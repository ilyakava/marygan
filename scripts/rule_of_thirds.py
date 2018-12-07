
from shutil import copyfile
from matplotlib.image import imread, imsave

from tqdm import tqdm

import pdb

# attr_file = '/Users/artsyinc/Documents/MATH630/research/data/celeba/list_attr_celeba.txt'
landmark_file = '/scratch0/ilya/locDoc/data/celeba/list_landmarks_align_celeba.txt'
src_folder = '/scratch0/ilya/locDoc/data/celeba/img_align_celeba'
dst_folder = '/scratch0/ilya/locDoc/data/celeba_thirds/img'

male_idx = 20

with open(landmark_file,'r') as f:
    for i, x in enumerate(tqdm(f)):
        x = x.rstrip()
        if i < 2:
            continue
        fname = x.split(' ')[0]
        landmarks = x.split(' ')[1:]
        landmarks = [int(a) for a in landmarks if len(a) > 0]

        if len(landmarks) == 10:
            nose_to_left_eye = landmarks[0] - landmarks[4]
            eye_distance = landmarks[0] - landmarks[2]
            rotation = nose_to_left_eye / float(eye_distance)
            # print('%s %.04f' % (fname, rotation))
            
            if (rotation > 0.35) and (rotation < 0.65):
                eye_height = (landmarks[1] + landmarks[3]) // 2
                mouth_height = (landmarks[7] + landmarks[9]) // 2
                nose_height = landmarks[5]
                # origin is top left
                thirds_height = mouth_height - eye_height
                top = eye_height - thirds_height
                bottom = nose_height + thirds_height

                left = landmarks[4] - thirds_height
                right = landmarks[4] + thirds_height

                # nose_to_right_eye = landmarks[2] - landmarks[4]

                img = imread('%s/%s' % (src_folder, fname))
                crop = img[top:bottom, left:right]
                imsave('%s/%s' % (dst_folder, fname), crop)
                # img.close()

