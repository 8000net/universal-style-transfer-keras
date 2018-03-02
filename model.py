from keras.models import Model
from keras.layers import Conv2D
import keras.backend as K
from vgg import VGG19, preprocess_input
from decoder import decoder_layers

LAMBDA=0.01

def l2_loss(x):
    return K.sum(K.square(x)) / 2

class EncoderDecoder:
    def __init__(self, input_shape=(256, 256, 3), target_layer=5):
        self.input_shape = input_shape
        self.target_layer = target_layer
        self.encoder = VGG19(input_shape=input_shape, target_layer=target_layer)
        self.loss = self.create_loss_fn(self.encoder)
        self.decoder_layers = self.create_decoder(target_layer)
        self.model = Model(self.encoder.inputs, self.decoder_layers)
        self.model.compile('adam', self.loss)

    def create_loss_fn(self, encoder):
        def get_encodings(inputs):
            encoder = VGG19(inputs, self.input_shape, self.target_layer)
            return encoder.output

        def loss(img_in, img_out):
            encoding_in = get_encodings(img_in)
            encoding_out = get_encodings(img_out)
            return l2_loss(img_out - img_in) + \
                   LAMBDA*l2_loss(encoding_out - encoding_in)
        return loss

    def create_decoder(self, target_layer):
        layers = decoder_layers(self.encoder.output, target_layer)
        return Conv2D(3, (3, 3), activation='relu', padding='same',
                name='decoder_out')(layers)

    def export_decoder(self):
        pass


#ed = EncoderDecoder()
#ed.model.summary()
