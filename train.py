from keras.preprocessing.image import ImageDataGenerator

from model import EncoderDecoder
from util import count_num_samples

TRAIN_PATH = 'data'
TARGET_SIZE = (256, 256)
BATCH_SIZE = 4
epochs = 1

datagen = ImageDataGenerator()
gen = datagen.flow_from_directory(TRAIN_PATH, target_size=TARGET_SIZE,
                                  batch_size=BATCH_SIZE, class_mode=None)


def create_gen(img_dir, target_size, batch_size):
    datagen = ImageDataGenerator()
    gen = datagen.flow_from_directory(img_dir, target_size=target_size,
                                      batch_size=batch_size, class_mode=None)

    def tuple_gen():
        for img in gen:
            if img.shape[0] != batch_size:
                continue

            # (X, y)
            yield (img, img)

    return tuple_gen()


gen = create_gen(TRAIN_PATH, TARGET_SIZE, BATCH_SIZE)

num_samples = count_num_samples(TRAIN_PATH)
steps_per_epoch = num_samples // BATCH_SIZE

encoder_decoder = EncoderDecoder()

encoder_decoder.model.fit_generator(gen, steps_per_epoch=steps_per_epoch,
        epochs=epochs)
encoder_decoder.export_decoder()
