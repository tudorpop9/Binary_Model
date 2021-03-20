import datetime
import os
import random
import threading
import time
from threading import Thread
import tifffile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cv2 import cv2
from skimage.io import imread, imshow
from tqdm import tqdm
import io
from PIL import Image

# Input images size
# original dimensions/16

DST_PARENT_DIR = './Vaihingen/'
PARENT_DIR = './Vaihingen/'
ORIGINAL_PATH = 'Originals/'
SEGMENTED_PATH = 'SegmentedOriginals/'

DST_SEGMENTED_PATH = "Variation_1_Segmented/"
DST_ORIGINAL_PATH = "Variation_1_Originals/"

ORIGINAL_RESIZED_PATH = "Resized_Originals_Variation_1/"
SEGMENTED_RESIZED_PATH = "Resized_Segmented_Variation_1/"
SEGMENTED_ONE_HOT_PATH = "Resized_Segmented_One_Hot/"

RESULTS_PATH = "./Results/"
LABEL_TYPES_PATH = "results_on_"

IMG_WIDTH = 250
IMG_HEIGHT = 250
IMG_CHANNELS = 3
SAMPLE_SIZE = 20000
BATCH_SIZE = 16
# current labels
labels = {
    0: (255, 255, 255),  # white, paved area/road
    1: (0, 0, 255),  # blue, buildings
    2: (0, 255, 255),  # light blue, low vegetation
    3: (0, 255, 0),  # green, high vegetation
    4: (255, 0, 0),  # red, bare earth
    5: (255, 255, 0)  # yellow, vehicle/car
}

labels_limited = {
    0: (255, 255, 255),  # white, paved area/road
    1: (0, 0, 255),  # blue, buildings
    2: (0, 0, 0),  # others
}

BUILDING_LABEL_IDX = 1
ROAD_LABEL_IDX = 0
OTHER_LABEL_IDX = 2

one_hot_labels = {
    0: [1, 0, 0],  # white, paved area/road
    1: [0, 1, 0],  # blue, buildings
    2: [0, 0, 1],  # background
}

N_OF_LABELS = len(labels)
seed = np.random.seed(42)
SEED_42 = 42
global_random = random.Random(SEED_42)


def decode_png_img(img):
    # img = tf.image.decode_png(img, channels=IMG_CHANNELS)
    # img = tf.image.convert_image_dtype(img, tf.uint8)
    print(img)
    img = imread(img)[:, :, :IMG_CHANNELS]
    # img = one_hot_enc(img.numpy())
    img = tf.convert_to_tensor(img, tf.uint8)

    return img


def decode_tif_img(img):
    # img = tf.image.decode_image(img, channels=N_OF_LABELS, dtype=tf.dtypes.uint8)
    img = tf.image.decode_png(img, channels=IMG_CHANNELS)
    img = imread(img)[:, :, :IMG_CHANNELS]
    # img = one_hot_enc(img.numpy())
    img = tf.convert_to_tensor(img, tf.uint8)
    # img = tf.image.convert_image_dtype(img, tf.uint8)

    return img

# actually loads an image, its maks and returns the pair
def combine_img_masks(original_path: tf.Tensor, segmented_path: tf.Tensor):

    original_image = tf.io.read_file(original_path)
    original_image = decode_png_img(original_image)

    mask_image = tf.io.read_file(segmented_path)
    mask_image = decode_tif_img(mask_image)
    return original_image, mask_image


class DataSetTool:
    def __init__(self):
        pass



    def to_limited_label_mask(self, mask):
        encoded_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        # print(input_img)
        # print(image)
        for row_idx in range(0, mask.shape[0]):
            for col_idx in range(0, mask.shape[1]):
                # if current pixel value has the current label value flag it in result
                # it uses the fact that all return_array values are initially 0
                if tuple(mask[row_idx, col_idx]) == labels_limited[BUILDING_LABEL_IDX]:
                    encoded_img[row_idx][col_idx] = labels_limited[BUILDING_LABEL_IDX]
                elif tuple(mask[row_idx, col_idx]) == labels_limited[ROAD_LABEL_IDX]:
                    encoded_img[row_idx][col_idx] = labels_limited[ROAD_LABEL_IDX]
                else:
                    encoded_img[row_idx][col_idx] = labels_limited[OTHER_LABEL_IDX]

        return encoded_img


    def get_max_channel_idx(self, image_channels):
        max_idx = 0
        for channel in image_channels:
            if image_channels[max_idx] < channel:
                idxs = np.where(image_channels == channel)
                max_idx = idxs[0][0]

        return max_idx

    def parse_prediction(self, predicted_image, labels):
        return_array = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

        for row_idx in range(0, predicted_image.shape[0]):
            for col_idx in range(0, predicted_image.shape[1]):
                try:
                    max_val_idx = self.get_max_channel_idx(predicted_image[row_idx][col_idx])
                    label = labels[max_val_idx]
                    return_array[row_idx][col_idx] = label
                except:
                    print("aici")

        return return_array

    def decode_one_hot_limited_labels(self, mask, one_hot_labels):
        return_array = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

        for row_idx in range(0, mask.shape[0]):
            for col_idx in range(0, mask.shape[1]):
                # try:
                if list(mask[row_idx][col_idx]) == one_hot_labels[BUILDING_LABEL_IDX]:
                    return_array[row_idx][col_idx] = labels_limited[BUILDING_LABEL_IDX]
                elif list(mask[row_idx][col_idx]) == one_hot_labels[ROAD_LABEL_IDX]:
                    return_array[row_idx][col_idx] = labels_limited[ROAD_LABEL_IDX]
                else:
                    return_array[row_idx][col_idx] = labels_limited[OTHER_LABEL_IDX]
            # except:
            #     print("aici")

        return return_array

    def augment_data_set(self):
        data_ids = os.listdir(PARENT_DIR + SEGMENTED_RESIZED_PATH)
        no_threads = 8
        aug_threads = []
        root_orig_path = DST_PARENT_DIR + ORIGINAL_RESIZED_PATH
        root_segm_path = DST_PARENT_DIR + SEGMENTED_RESIZED_PATH

        factor = int(len(data_ids) / no_threads)
        for thIdx in range(0, len(data_ids), factor):
            if thIdx == no_threads - 1:
                # give rest of the data to last thread
                th = Thread(target=self.thread_aug_data_function, args=(data_ids[thIdx:],
                                                                        root_orig_path, root_segm_path,))
                aug_threads.append(th)
                th.start()
            else:
                # give thread a chunk of data to process
                th = Thread(target=self.thread_aug_data_function, args=(data_ids[thIdx: thIdx + factor - 1],
                                                                        root_orig_path, root_segm_path,))
                aug_threads.append(th)
                th.start()

        for th in aug_threads:
            th.join()

    # resize extracted segmented
    def resize_segmented(self):
        ids = os.listdir(PARENT_DIR + DST_SEGMENTED_PATH)
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            path = PARENT_DIR + DST_SEGMENTED_PATH + id_.split('.')[0] + '.png'
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(DST_PARENT_DIR + SEGMENTED_RESIZED_PATH + id_, resized_img)


    # resize extracted original
    def resize_original(self):
        ids = os.listdir(PARENT_DIR + DST_ORIGINAL_PATH)
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            path = PARENT_DIR + DST_ORIGINAL_PATH + id_
            img_ = cv2.imread(path, cv2.IMREAD_COLOR)

            resized_img = cv2.resize(img_, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            cv2.imwrite(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + id_, resized_img)

    # original image data set extraction
    def split_original(self):
        ids = os.listdir(PARENT_DIR + ORIGINAL_PATH)
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            count = 0

            path = PARENT_DIR + ORIGINAL_PATH + id_
            img_ = cv2.imread(path, cv2.IMREAD_COLOR)

            fragmentShape = 1000, 1000, 3
            fragment = np.zeros(fragmentShape, dtype=np.uint8)
            for offset_i in range(0, img_.shape[0] // 1000):
                for offset_j in range(0, img_.shape[0] // 1000):

                    for i in range(0, img_.shape[0] // 6):
                        for j in range(0, img_.shape[1] // 6):
                            fragment[i, j] = img_[i + offset_i * 1000, j + offset_j * 1000]

                    cv2.imwrite(
                        DST_PARENT_DIR + DST_ORIGINAL_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                        fragment)
                    count = count + 1

            for offset_i in range(0, img_.shape[0] // 1000 - 1):
                for offset_j in range(0, img_.shape[0] // 1000 - 1):

                    for i in range(0, img_.shape[0] // 6):
                        for j in range(0, img_.shape[1] // 6):
                            fragment[i, j] = img_[
                                500 + i + offset_i * 1000, 500 + j + offset_j * 1000]  # 500 pt a porni de la 500, nu 0 si a termina la 5500

                    cv2.imwrite(
                        DST_PARENT_DIR + DST_ORIGINAL_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                        fragment)
                    count = count + 1

    # segmented image data extractions
    def split_segmented(self):
        ids = os.listdir(PARENT_DIR + SEGMENTED_PATH)
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            count = 0

            path = PARENT_DIR + SEGMENTED_PATH + id_
            img_ = cv2.imread(path, cv2.IMREAD_COLOR)

            fragmentShape = 1000, 1000, 3
            fragment = np.zeros(fragmentShape, dtype=np.uint8)
            for offset_i in range(0, img_.shape[0] // 1000):
                for offset_j in range(0, img_.shape[0] // 1000):

                    for i in range(0, img_.shape[0] // 6):
                        for j in range(0, img_.shape[1] // 6):
                            fragment[i, j] = img_[i + offset_i * 1000, j + offset_j * 1000]

                    cv2.imwrite(
                        DST_PARENT_DIR + DST_SEGMENTED_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                        fragment)
                    count = count + 1

            for offset_i in range(0, img_.shape[0] // 1000 - 1):
                for offset_j in range(0, img_.shape[0] // 1000 - 1):

                    for i in range(0, img_.shape[0] // 6):
                        for j in range(0, img_.shape[1] // 6):
                            fragment[i, j] = img_[
                                500 + i + offset_i * 1000, 500 + j + offset_j * 1000]  # 500 pt a porni de la 500, nu 0 si a termina la 5500

                    cv2.imwrite(
                        DST_PARENT_DIR + DST_SEGMENTED_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                        fragment)
                    count = count + 1

    def thread_aug_data_function(self, data_fragment, root_orig_path, root_segm_path):
        for n, id_ in tqdm(enumerate(data_fragment), total=len(data_fragment)):
            # print(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + id_)
            original_img = imread(root_orig_path + id_)[:, :, :IMG_CHANNELS]
            segmented_img = imread(root_segm_path + id_)[:, :, :IMG_CHANNELS]
            aug_originals = []
            aug_segmented = []

            # roatated images
            rot_o_90 = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
            aug_originals.append(rot_o_90)
            rot_o_180 = cv2.rotate(original_img, cv2.ROTATE_180)
            aug_originals.append(rot_o_180)
            rot_o_270 = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            aug_originals.append(rot_o_270)

            # horizontally flipped images
            flip_o_h_org = cv2.flip(original_img, 1)
            aug_originals.append(flip_o_h_org)
            flip_o_h_90 = cv2.flip(rot_o_90, 1)
            aug_originals.append(flip_o_h_90)
            flip_o_h_180 = cv2.flip(rot_o_180, 1)
            aug_originals.append(flip_o_h_180)
            flip_o_h_270 = cv2.flip(rot_o_270, 1)
            aug_originals.append(flip_o_h_270)

            # brighter images
            brighter_o_ = cv2.add(original_img, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_)
            brighter_o_rot_o_90 = cv2.add(rot_o_90, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_rot_o_90)
            brighter_o_rot_o_180 = cv2.add(rot_o_180, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_rot_o_180)
            brighter_o_rot_o_270 = cv2.add(rot_o_270, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_rot_o_270)
            brighter_o_flip_o_h_org = cv2.add(flip_o_h_org, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_flip_o_h_org)
            brighter_o_flip_o_h_90 = cv2.add(flip_o_h_90, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_flip_o_h_90)
            brighter_o_flip_o_h_180 = cv2.add(flip_o_h_180, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_flip_o_h_180)
            brighter_o_flip_o_h_270 = cv2.add(flip_o_h_270, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_flip_o_h_270)

            # dimmer images
            dimmer_o_ = cv2.subtract(original_img, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_)
            dimmer_o_rot_o_90 = cv2.subtract(rot_o_90, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_rot_o_90)
            dimmer_o_rot_o_180 = cv2.subtract(rot_o_180, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_rot_o_180)
            dimmer_o_rot_o_270 = cv2.subtract(rot_o_270, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_rot_o_270)
            dimmer_o_flip_o_h_org = cv2.subtract(flip_o_h_org, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_flip_o_h_org)
            dimmer_o_flip_o_h_90 = cv2.subtract(flip_o_h_90, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_flip_o_h_90)
            dimmer_o_flip_o_h_180 = cv2.subtract(flip_o_h_180, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_flip_o_h_180)
            dimmer_o_flip_o_h_270 = cv2.subtract(flip_o_h_270, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_flip_o_h_270)

            # same operations for segmented data
            rot_s_90 = cv2.rotate(segmented_img, cv2.ROTATE_90_CLOCKWISE)
            aug_segmented.append(rot_s_90)
            rot_s_180 = cv2.rotate(segmented_img, cv2.ROTATE_180)
            aug_segmented.append(rot_s_180)
            rot_s_270 = cv2.rotate(segmented_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            aug_segmented.append(rot_s_270)

            flip_s_h_org = cv2.flip(segmented_img, 1)
            aug_segmented.append(flip_s_h_org)
            flip_s_h_90 = cv2.flip(rot_s_90, 1)
            aug_segmented.append(flip_s_h_90)
            flip_s_h_180 = cv2.flip(rot_s_180, 1)
            aug_segmented.append(flip_s_h_180)
            flip_s_h_270 = cv2.flip(rot_s_270, 1)
            aug_segmented.append(flip_s_h_270)

            # brighter segmented images
            aug_segmented.append(segmented_img.copy())
            aug_segmented.append(rot_s_90.copy())
            aug_segmented.append(rot_s_180.copy())
            aug_segmented.append(rot_s_270.copy())
            aug_segmented.append(flip_s_h_org.copy())
            aug_segmented.append(flip_s_h_90.copy())
            aug_segmented.append(flip_s_h_180.copy())
            aug_segmented.append(flip_s_h_270.copy())

            # dimmer segmented images
            aug_segmented.append(segmented_img.copy())
            aug_segmented.append(rot_s_90.copy())
            aug_segmented.append(rot_s_180.copy())
            aug_segmented.append(rot_s_270.copy())
            aug_segmented.append(flip_s_h_org.copy())
            aug_segmented.append(flip_s_h_90.copy())
            aug_segmented.append(flip_s_h_180.copy())
            aug_segmented.append(flip_s_h_270.copy())

            for i in range(0, len(aug_segmented)):
                # saves all the augmented originals
                rgb_orig = cv2.cvtColor(aug_originals[i], cv2.COLOR_BGR2RGB)
                cv2.imwrite(root_orig_path + id_.split('.')[0] + '_' + str((i + 1)) + '.png', rgb_orig)
                # saves all the augmented segmented
                rgb_segm = cv2.cvtColor(aug_segmented[i], cv2.COLOR_BGR2RGB)
                cv2.imwrite(root_segm_path + id_.split('.')[0] + '_' + str((i + 1)) + '.png', rgb_segm)

        pass

    # creates an input pipeline
    def get_input_pipeline(self):
        # read ids of the input images
        # the images are inside a subfolder with the same name because of the 'get_generator' function
        originals_root_dir = PARENT_DIR + ORIGINAL_RESIZED_PATH
        masks_root_dir = PARENT_DIR + SEGMENTED_RESIZED_PATH
        # get an array with relative path for each image
        originals_ids = os.listdir(originals_root_dir)
        originals_ids.sort(reverse=False)
        global_random.shuffle(originals_ids)
        originals_full_paths = [originals_root_dir + id_ for id_ in originals_ids]

        mask_ids = os.listdir(masks_root_dir)
        mask_ids.sort(reverse=False)
        global_random.shuffle(mask_ids)
        masks_full_paths = [masks_root_dir + id_ for id_ in mask_ids]

        # create dataset using relative path names
        originals_ds = tf.data.Dataset.from_tensor_slices(originals_full_paths)
        masks_ds = tf.data.Dataset.from_tensor_slices(masks_full_paths)

        train_ds = tf.data.Dataset.zip((originals_ds, masks_ds))
        train_ds = train_ds.map(lambda x, y: tf.py_function(func=combine_img_masks,
                                                            inp=[x, y], Tout=(tf.uint8, tf.uint8)),
                                num_parallel_calls=4,
                                deterministic=False)

        # train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE).cache()
        train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        # print(train_ds.element_spec)
        # train_ds.prefetch(10)
        # for image, label in train_ds.take(1):
        #     print(image.shape)
        #     print(label.shape)
        #     print()

        # exit(1)
        return train_ds_batched