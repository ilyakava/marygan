import numpy as np
from shutil import copyfile
from matplotlib.image import imread, imsave

from tqdm import tqdm

import pdb

# attr_file = '/Users/artsyinc/Documents/MATH630/research/data/celeba/list_attr_celeba.txt'
landmark_file = '/scratch0/ilya/locDoc/data/celeba/list_landmarks_align_celeba.txt'
src_folder = '/scratch0/ilya/locDoc/data/celeba/img_align_celeba'
dst_folder = '/scratch0/ilya/locDoc/data/celeba_sides/right/img_crop'
dst_folder2 = '/scratch0/ilya/locDoc/data/celeba_sides/left/img_crop'

rotations = []

with open(landmark_file,'r') as f:
    for i, x in enumerate(tqdm(f)):
        x = x.rstrip()
        if i < 2:
            continue
        fname = x.split(' ')[0]
        landmarks = x.split(' ')[1:]
        landmarks = [int(a) for a in landmarks if len(a) > 0]
        lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y = landmarks

        if len(landmarks) == 10:
            nose_to_left_eye = landmarks[0] - landmarks[4]
            eye_distance = landmarks[0] - landmarks[2]
            rotation = nose_to_left_eye / float(eye_distance)
            # print('%s %.04f' % (fname, rotation))
            
            rotations.append(rotation)
            left_profile = (rotation < 0.25) and (rotation > -0.5)
            right_profile = (rotation > 0.75) and (rotation < 1.5)

            if right_profile or left_profile:
                eye_height = (landmarks[1] + landmarks[3]) // 2
                mouth_height = (landmarks[7] + landmarks[9]) // 2
                nose_height = landmarks[5]
                # origin is top left
                thirds_height = mouth_height - eye_height
                top = eye_height - thirds_height
                bottom = nose_height + thirds_height

                # make up for rotation
                x = rotation
                right_of_nose_prop = (np.cos(np.pi*(x + 1/2)/2) + 1) / 2.0
                left_of_nose_prop = 1 - right_of_nose_prop

                right = int((2*thirds_height) * right_of_nose_prop + nose_x)
                left = int(nose_x - ((2*thirds_height) * left_of_nose_prop))


                # nose_to_right_eye = landmarks[2] - landmarks[4]
                # prefix = '%0.2f' % rotation

                img = imread('%s/%s' % (src_folder, fname))
                [h,w,c] = img.shape
                right = min(right, h)
                left = max(left, 0)
                crop = img[top:bottom, left:right]
                if right_profile:
                    imsave('%s/%s' % (dst_folder, fname), crop)
                else:
                    imsave('%s/%s' % (dst_folder2, fname), crop)
                # img.close()
# pdb.set_trace()
