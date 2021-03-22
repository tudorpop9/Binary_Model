import datetime
import os

import tensorflow as tf
import DataSetTool
import Unet

ds_tool = DataSetTool.DataSetTool()
unet_tool = Unet.Unet()


# ds_tool.resize_original()
# ds_tool.resize_segmented()
# ds_tool.augment_data_set()

model = unet_tool.create_binary_model()

print('Do you want to train the model first? [y/n]')
to_train = input()

# applies one hot encoding on label images
print('One hot encoding labeled images..')
# labels, one_hot_y_train = to_one_hot(N_OF_LABELS, Y_train)


current_day = datetime.datetime.now()
# if flag is an even number we perform a fit operation, training the model and save its best results
if to_train.lower() == 'y':

    if os.path.exists('./binary_semantic_segmentation.h5'):
        model.load_weights('binary_semantic_segmentation.h5')

    metric = 'accuracy'

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir='logs' + '/logs_on_' + str(current_day.month).zfill(2) + str(current_day.day).zfill(2)),
        tf.keras.callbacks.ModelCheckpoint(filepath='./binary_semantic_segmentation.h5', monitor=metric,
                                           verbose=2, save_best_only=True, mode='max')
    ]

    train_ds_batched, validation_ds_batched = ds_tool.get_input_pipeline()

    model.fit(train_ds_batched,
              # validation_data=validation_ds_batched,
              batch_size=16,
              callbacks=callbacks,
              epochs=3,
              verbose=1)

# otherwise we load the weights from another run
else:
    model.load_weights('binary_semantic_segmentation.h5')

# observing results
ds_tool.manual_model_testing(model)