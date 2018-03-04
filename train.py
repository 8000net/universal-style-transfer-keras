from keras.preprocessing.image import ImageDataGenerator

from model import EncoderDecoder
from util import count_num_samples

TRAIN_PATH = 'data'
TARGET_SIZE = (256, 256, 3)
BATCH_SIZE = 4
epochs = 1

datagen = ImageDataGenerator()
gen = datagen.flow_from_directory(TRAIN_PATH, target_size=TARGET_SIZE,
                                  batch_size=BATCH_SIZE, class_mode=None)

num_samples = count_num_samples(TRAIN_PATH)
steps_per_epoch = num_samples // batch_size

encoder_decoder = EncoderDecoder()

encoder_decoder.model.fit_generator(gen, steps_per_epoch=steps_per_epoch,
        epochs=epochs)
encoder_decoder.export_decoder()
