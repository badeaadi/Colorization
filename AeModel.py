import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import load_model
import os
# import SGD and Adam optimizers
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from DataSet import *


class AeModel:

    def __init__(self, data_set: DataSet):
        
        self.data_set = data_set
        self.num_epochs = 75
        self.batch_size = 32
        self.learning_rate = 10 ** (-3)
        self.model = None
        self.model_type = 'large'
        self.checkpoint_dir = './checkpoints_%s' % self.data_set.scene_name

    def define_the_model(self):
        
        # defineste autoencoderul
        
        if self.model_type == 'tiny':
            self.model = tf.keras.models.Sequential([
                layers.InputLayer(input_shape=(self.data_set.network_input_size[0], self.data_set.network_input_size[1], 1)),
                layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
                layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu',  padding='same'),
                layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',  padding='same'),
                layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
                layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',  padding='same'),
                layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(filters=2, kernel_size=(3, 3), activation='tanh', padding='same')
            ])
            
        else:
            self.model = tf.keras.models.Sequential([
                layers.InputLayer(input_shape=(self.data_set.network_input_size[0], self.data_set.network_input_size[1], 1)),
                layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
                layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
                layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
                layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same'),
                layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same'),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same'),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same'),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(filters=2, kernel_size=(3, 3), activation='tanh', strides=(1, 1), padding='same'),
            ])
        
        
        
        # afiseaza arhitectura modelului
        self.model.summary()

    def compile_the_model(self):
        
        self.model.compile(optimizer='Adam', loss='mse')


    def train_the_model(self):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
            
        # definim callback-ul pentru checkpoint
        my_callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir + '/model.{epoch:05d}.hdf5') ]
        
        self.data_set.input_training_images  = self.data_set.input_training_images.reshape(self.data_set.training_length, self.data_set.network_input_size[0], self.data_set.network_input_size[1], 1)
        self.data_set.ground_truth_training_images = self.data_set.ground_truth_training_images.reshape(self.data_set.training_length, self.data_set.network_input_size[0], self.data_set.network_input_size[1], 2)
        
        self.model.fit(self.data_set.input_training_images, self.data_set.ground_truth_training_images, batch_size= self.batch_size , epochs=self.num_epochs, callbacks = my_callbacks)


    def evaluate_the_model(self):
        best_epoch = self.num_epochs  # puteti incerca si cu alta epoca de exemplu cu prima epoca,
                                      # sa vedeti diferenta dintre ultima epoca si prima
        # incarcam modelul
        best_model = load_model(os.path.join(self.checkpoint_dir, 'model.%05d.hdf5') % best_epoch)
        
        for i in range(len(self.data_set.input_test_images)):
            # prezicem canalele ab pe baza input_test_images[i]
            
            inp = self.data_set.input_test_images[i]
            inp = np.reshape(inp, (1, self.data_set.network_input_size[0], self.data_set.network_input_size[1], 1))

            output = self.model.predict(inp)
            output *= 128
        
            lab_image = cv.cvtColor(np.float32(self.data_set.ground_truth_bgr_test_images[i]) / 255, cv.COLOR_BGR2LAB)
            # reconstruim reprezentarea Lab                        
            output_color_lab = np.zeros((self.data_set.network_input_size[0], self.data_set.network_input_size[1], 3))
            output_color_lab[:,:,0] = lab_image[:, :, 0]
            output_color_lab[:,:,1:] = output[0]

            # convertim din Lab in BGR            
            output_color_bgr = cv.cvtColor(np.float32(output_color_lab), cv.COLOR_LAB2BGR)*255            
          
            # convertim imaginea de input din L in 'grayscale'
            
            gray = np.uint8(self.data_set.input_test_images[i] * 255)
            gray = np.reshape(gray, (self.data_set.network_input_size[0], self.data_set.network_input_size[1]))
            
            input_image = np.zeros((self.data_set.network_input_size[0], self.data_set.network_input_size[1], 3))
            input_image[:,:,0] = gray
            input_image[:,:,1] = gray
            input_image[:,:,2] = gray
            
            
            # imaginea ground-truth in format bgr
            gt_image = np.uint8(self.data_set.ground_truth_bgr_test_images[i])
            
            concat_images = self.concat_images(input_image, output_color_bgr, gt_image)
            
            cv.imwrite(os.path.join(self.data_set.dir_output_images, '%d.png' % i), concat_images)

    def concat_images(self, input_image, pred, ground_truth):
        """
        :param input_image: imaginea grayscale (canalul L din reprezentarea Lab).
        :param pred: imaginea prezisa.
        :param ground_truth: imaginea ground-truth.
        :return: concatenarea imaginilor.
        """
        h, w, _, = input_image.shape
        space_btw_images = int(0.2 * h)
        image = np.ones((h, w * 3 + 2 * space_btw_images, 3)) * 255
        
        #print(input_image)
        
        # add input_image
        image[:, :w] = input_image
        # add predicted
        offset = w + space_btw_images
        image[:, offset: offset + w] = pred
        # add ground truth
        
        print(image.shape)
        offset = 2 * (w + space_btw_images)
        image[:, offset: offset + w] = ground_truth
        
        return np.uint8(image)