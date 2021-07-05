import datetime
import json
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
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
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
    img = tf.image.decode_png(img, channels=IMG_CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    # print(img)
    # img = imread(img)[:, :, :IMG_CHANNELS]
    # img = one_hot_enc(img.numpy())
    # img = tf.convert_to_tensor(img, tf.uint8)

    return img


def decode_tif_img(img):
    # img = tf.image.decode_image(img, channels=N_OF_LABELS, dtype=tf.dtypes.uint8)
    img = tf.image.decode_png(img, channels=1)
    # img = imread(img)[:, :, :IMG_CHANNELS]
    # img = one_hot_enc(img.numpy())
    # img = tf.convert_to_tensor(img, tf.uint8)
    img = tf.image.convert_image_dtype(img, tf.uint8)

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

    # resize extracted segmented and
    def resize_segmented(self):
        ids = os.listdir(PARENT_DIR + DST_SEGMENTED_PATH)
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            path = PARENT_DIR + DST_SEGMENTED_PATH + id_.split('.')[0] + '.png'
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

            for i in range(0, resized_img.shape[0]):
                for j in range(0, resized_img.shape[1]):
                    # pick a class "paved area"
                    # if label does not represent the road, make it 0
                    if resized_img[i][j] != 255:
                        resized_img[i][j] = 0.0
                    else:
                        resized_img[i][j] = 1.0

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
            segmented_img = imread(root_segm_path + id_)
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

            # # brighter images
            # brighter_o_ = cv2.add(original_img, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(brighter_o_)
            # brighter_o_rot_o_90 = cv2.add(rot_o_90, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(brighter_o_rot_o_90)
            # brighter_o_rot_o_180 = cv2.add(rot_o_180, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(brighter_o_rot_o_180)
            # brighter_o_rot_o_270 = cv2.add(rot_o_270, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(brighter_o_rot_o_270)
            # brighter_o_flip_o_h_org = cv2.add(flip_o_h_org, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(brighter_o_flip_o_h_org)
            # brighter_o_flip_o_h_90 = cv2.add(flip_o_h_90, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(brighter_o_flip_o_h_90)
            # brighter_o_flip_o_h_180 = cv2.add(flip_o_h_180, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(brighter_o_flip_o_h_180)
            # brighter_o_flip_o_h_270 = cv2.add(flip_o_h_270, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(brighter_o_flip_o_h_270)
            #
            # # dimmer images
            # dimmer_o_ = cv2.subtract(original_img, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(dimmer_o_)
            # dimmer_o_rot_o_90 = cv2.subtract(rot_o_90, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(dimmer_o_rot_o_90)
            # dimmer_o_rot_o_180 = cv2.subtract(rot_o_180, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(dimmer_o_rot_o_180)
            # dimmer_o_rot_o_270 = cv2.subtract(rot_o_270, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(dimmer_o_rot_o_270)
            # dimmer_o_flip_o_h_org = cv2.subtract(flip_o_h_org, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(dimmer_o_flip_o_h_org)
            # dimmer_o_flip_o_h_90 = cv2.subtract(flip_o_h_90, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(dimmer_o_flip_o_h_90)
            # dimmer_o_flip_o_h_180 = cv2.subtract(flip_o_h_180, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(dimmer_o_flip_o_h_180)
            # dimmer_o_flip_o_h_270 = cv2.subtract(flip_o_h_270, np.array([random.randint(50, 80) / 1.0]))
            # aug_originals.append(dimmer_o_flip_o_h_270)

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

            # # brighter segmented images
            # aug_segmented.append(segmented_img.copy())
            # aug_segmented.append(rot_s_90.copy())
            # aug_segmented.append(rot_s_180.copy())
            # aug_segmented.append(rot_s_270.copy())
            # aug_segmented.append(flip_s_h_org.copy())
            # aug_segmented.append(flip_s_h_90.copy())
            # aug_segmented.append(flip_s_h_180.copy())
            # aug_segmented.append(flip_s_h_270.copy())
            #
            # # dimmer segmented images
            # aug_segmented.append(segmented_img.copy())
            # aug_segmented.append(rot_s_90.copy())
            # aug_segmented.append(rot_s_180.copy())
            # aug_segmented.append(rot_s_270.copy())
            # aug_segmented.append(flip_s_h_org.copy())
            # aug_segmented.append(flip_s_h_90.copy())
            # aug_segmented.append(flip_s_h_180.copy())
            # aug_segmented.append(flip_s_h_270.copy())

            for i in range(0, len(aug_segmented)):
                # saves all the augmented originals
                rgb_orig = cv2.cvtColor(aug_originals[i], cv2.COLOR_BGR2RGB)
                cv2.imwrite(root_orig_path + id_.split('.')[0] + '_' + str((i + 1)) + '.png', rgb_orig)
                # saves all the augmented segmented
                # gs_segm = cv2.cvtColor(aug_segmented[i], cv2.COLOR_BGR2GRAY)
                cv2.imwrite(root_segm_path + id_.split('.')[0] + '_' + str((i + 1)) + '.png', aug_segmented[i])

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

        ds_batched = train_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

        train_ds_batched = ds_batched.take(round(len(originals_ids) * 0.8))
        validation_ds_batched = ds_batched.skip(round(len(originals_ids) * 0.8))
        #
        # train_ds_batched = tf.compat.v1.data.make_initializable_iterator(train_ds_batched)
        # validation_ds_batched = tf.compat.v1.data.make_initializable_iterator(validation_ds_batched)
        #
        # train_ds_batched = tf.compat.v1.Session.run(train_ds_batched.initializer)
        # validation_ds_batched = tf.compat.v1.Session.run(validation_ds_batched.initializer)

        return train_ds_batched, validation_ds_batched

    def get_data_set(self):
        train_ids = os.listdir(PARENT_DIR + ORIGINAL_RESIZED_PATH)
        sample_ds = random.sample(train_ids, 5000)

        X_train = np.zeros((len(sample_ds), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)
        Y_train = np.zeros((len(sample_ds), IMG_WIDTH, IMG_HEIGHT), dtype=np.bool)

        for n, id_ in tqdm(enumerate(sample_ds), total=len(sample_ds)):
            # Actual train image
            # print(DST_PARENT_DIR + ORIGINAL_PATH + id_)
            img = imread(PARENT_DIR + ORIGINAL_RESIZED_PATH + id_.split(".")[0] + '.png')[:, :, :IMG_CHANNELS]
            X_train[n] = img
            mask = imread(PARENT_DIR + SEGMENTED_RESIZED_PATH + id_.split(".")[0] + '.png')
            Y_train[n] = mask

        return X_train, Y_train

    def decode_binary_mask(self, mask):
        return_array = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
        for row_idx in range(0, mask.shape[0]):
            for col_idx in range(0, mask.shape[1]):
                if mask[row_idx][col_idx] == 1:
                    return_array[row_idx][col_idx] = 255

        return return_array

    def manual_model_testing(self, model):
        current_day = datetime.datetime.now()
        train_ids = os.listdir(PARENT_DIR + ORIGINAL_RESIZED_PATH)
        random_images_idx = random.sample(train_ids, 100)
        X_train = np.zeros((100, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        ground_truth = np.zeros((100, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

        for n, id_ in tqdm(enumerate(random_images_idx), total=len(random_images_idx)):
            img = imread(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + train_ids[n])[:, :,
                  :IMG_CHANNELS]
            X_train[n] = img

            mask = imread(DST_PARENT_DIR + SEGMENTED_RESIZED_PATH + train_ids[n].split('.')[0] + '.png')

            # ground_truth[n] = mask
            ground_truth[n] = self.decode_binary_mask(mask)

        preds_train = model.predict(X_train, verbose=1)

        # Binarizationing the results
        preds_train_t = (preds_train > 0.5).astype(np.uint8)

        print("Enter 0 to exit, any other number to predict another image: ")
        continue_flag = input()

        while int(continue_flag) > 0:
            i = random.randint(0, len(preds_train_t))

            trainPath = "%s%sstrain%03d.png" % (RESULTS_PATH, LABEL_TYPES_PATH, i)
            controlPath = "%s%scontrolMask%03d.png" % (
                RESULTS_PATH, LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(current_day.day).zfill(2) + '/',
                i)
            predictionPath = "%s%sprediction%03d.png" % (
                RESULTS_PATH, LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(current_day.day).zfill(2) + '/',
                i)

            today_result_dir = RESULTS_PATH + LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(
                current_day.day).zfill(2)
            if not os.path.exists(today_result_dir):
                os.mkdir(today_result_dir)

            imshow(X_train[i])
            plt.savefig(trainPath)
            plt.show()

            imshow(np.squeeze(ground_truth[i]))
            plt.savefig(controlPath)
            plt.show()

            # interpreted_prediction = data_set.parse_prediction(preds_train[i], labels_limited)
            imshow(np.squeeze(preds_train_t[i]))
            plt.savefig(predictionPath)
            plt.show()

            print("Enter 0 to exit, any positive number to predict another image: ")
            continue_flag = input()

    def _get_statistics_dict(self):
        stats = {
            '0': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            '1': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            '2': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            '3': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            '4': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            '5': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            'accuracy': 0.0,
            'macro avg': {'precision': 0.0,
                          'recall': 0.0,
                          'f1-score': 0.0,
                          'support': 0},
            'weighted avg': {'precision': 0.0,
                             'recall': 0.0,
                             'f1-score': 0.0,
                             'support': 0},
        }
        return stats

    def print_per_class_statistics(self, validation_split, model: tf.keras.Model):

        global normalized_conf_matrix, initial_conf_matrix
        dict_stats_file = './binary_model_stats.json'

        train_ids = os.listdir(PARENT_DIR + ORIGINAL_RESIZED_PATH)
        random.shuffle(train_ids)
        split_idx = int(len(train_ids) * validation_split)
        validation_fragment = train_ids[:split_idx]
        validation_size = len(validation_fragment)

        batch_size = 4
        batch_idx = 0
        no_batches = 0

        images = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        ground_truth = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

        stats = self._get_statistics_dict()
        normalized_conf_matrix = np.zeros((1, 1), dtype=np.float)

        for n, id_ in tqdm(enumerate(validation_fragment), total=len(validation_fragment)):
            img = imread(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + train_ids[n])[:, :, :IMG_CHANNELS]
            images[batch_idx] = img

            mask = imread(DST_PARENT_DIR + SEGMENTED_RESIZED_PATH + train_ids[n].split('.')[0] + '.png')
            ground_truth[batch_idx] = mask
            batch_idx += 1

            if batch_idx == batch_size - 1 or n == validation_size - 1:
                batch_idx = 0
                no_batches += 1

                predictions = model.predict(images)
                predictions_max_score = np.where(predictions > 0.5, 1, 0).flatten()
                ground_truth_max_score = ground_truth.flatten()

                # initial_conf_matrix = tf.math.confusion_matrix(num_classes=1,
                #                                                labels=ground_truth_max_score,
                #                                                predictions=predictions_max_score).numpy()
                # normalized_conf_matrix += np.around(normalize(initial_conf_matrix, axis=1, norm='l1'), decimals=2)

                report = classification_report(ground_truth_max_score, predictions_max_score, output_dict=True)

                for label_i in report.keys():
                    if label_i != 'accuracy':
                        for metric in report[label_i].keys():
                            stats[label_i][metric] += report[label_i][metric]
                    else:
                        stats[label_i] += report[label_i]

        # final avg statistics
        for label_i in stats.keys():
            if label_i != 'accuracy':
                for metric in stats[label_i].keys():
                    stats[label_i][metric] /= no_batches
            else:
                stats[label_i] /= no_batches
        print(stats)

        # store it in a file
        with open(dict_stats_file, 'w') as file:
            json.dump(stats, file, indent=4)

        # final confusion matrix
        # normalized_conf_matrix = np.around(normalize(normalized_conf_matrix, axis=1, norm='l1'), decimals=2)
        # print(normalized_conf_matrix)

    def manual_model_testing_on_validation_set(self, model: tf.keras.Model, validation_split=0.2):
        current_day = datetime.datetime.now()

        train_ids = os.listdir(PARENT_DIR + ORIGINAL_RESIZED_PATH)
        random.shuffle(train_ids)
        split_idx = int(len(train_ids) * validation_split)
        validation_fragment = train_ids[:split_idx]
        validation_set_size = len(validation_fragment)

        original_image_as_np_array = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

        print("Enter 0 to exit, any other number to predict an image: ")
        continue_flag = input()

        while int(continue_flag) > 0:

            i = random.randint(0, validation_set_size - 1)
            img_name = validation_fragment[i]
            print("Chose a random image from validation set: " + str(i))

            # generates paths for each image in the sample
            original_img_path = "%s%s_original_%s" % (RESULTS_PATH,
                                                      LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(
                                                          current_day.day).zfill(2) + '/',
                                                      img_name)
            ground_truth_path = "%s%s_grTruth_%s" % (RESULTS_PATH,
                                                     LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(
                                                         current_day.day).zfill(2) + '/',
                                                     img_name)
            prediction_path = "%s%s_prediction_%s" % (RESULTS_PATH,
                                                      LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(
                                                          current_day.day).zfill(2) + '/',
                                                      img_name)

            # if the results directory for today does not exist, create it
            today_result_dir = RESULTS_PATH + LABEL_TYPES_PATH + \
                               str(current_day.month).zfill(2) + str(current_day.day).zfill(2)
            if not os.path.exists(today_result_dir):
                os.mkdir(today_result_dir)

            # read the randomly chosen image and its ground-truth
            img = imread(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + train_ids[i])[:, :, :IMG_CHANNELS]
            original_image_as_np_array[0] = img

            mask = imread(DST_PARENT_DIR + SEGMENTED_RESIZED_PATH + train_ids[i])

            # parse the one-hot representation to rgb
            print('Please wait, decoding ground-truth image.. ')
            ground_truth = mask*255

            # predict the random image
            prediction_arr = model.predict(original_image_as_np_array)

            # save and display sample
            imshow(img)
            # rgb_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(original_img_path, rgb_original)
            plt.savefig(original_img_path)
            plt.show()

            imshow(np.squeeze(ground_truth))
            # rgb_ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(ground_truth_path, rgb_ground_truth)
            plt.savefig(ground_truth_path)
            plt.show()

            print('Please wait, decoding the predicted image.. ')
            # interpreted_prediction = np.where(prediction_arr > 0.5, 1, 0)
            interpreted_prediction = prediction_arr[0]
            imshow(np.squeeze(interpreted_prediction))
            # rgb_interpreted_prediction = cv2.cvtColor(interpreted_prediction, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(prediction_path, rgb_interpreted_prediction)
            plt.savefig(prediction_path)
            plt.show()

            print("Enter 0 to exit, any positive number to predict another image: ")
            continue_flag = input()