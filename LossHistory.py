from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model, load_model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        self.batch_loss = []
        self.val_acc = []

    def on_batch_end(self, batch, logs={}):
        self.batch_loss.append(logs.get('loss'))