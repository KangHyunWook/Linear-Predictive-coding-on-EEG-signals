from keras.layers import LSTM, Dropout, Dense, Activation, Conv1D, Flatten, Layer
import keras.backend as K

def zero_loss(y_true, y_pred):
    return  K.sum(K.sqrt(y_true-y_pred),axis=0)

def get_model_output(x, num_classes):
    
    x=Conv1D(256, 5, strides=1, padding='same')(x)
    x=Conv1D(256, 5, strides=1, padding='same')(x)
    
    x=Conv1D(128, 5, strides=1, padding='same')(x)
    x=Conv1D(128, 5, strides=1, padding='same')(x)
    
    x=Conv1D(64, 3, strides=1, padding='same')(x)
    x=Conv1D(64, 3, strides=1, padding='same')(x)
    x=Flatten()(x)
    
    x=Dense(num_classes)(x)
    x=Activation('softmax')(x)
    
    return x

class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(3, 3),
                                       initializer='uniform',
                                       trainable=False)
                                       
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
     
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        
        new_centers = self.centers - self.alpha * delta_centers
        
        self.add_update((self.centers, new_centers), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) / K.dot(x[1], center_counts)
        
        return self.result 

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

"""
Code adapted from: https://github.com/handongfeng/MNIST-center-loss/blob/master/centerLoss_MNIST.py
"""    
def get_centermodel_output(x, labels, num_classes):  
    x=Conv1D(256, 5, strides=1, padding='same')(x)
    x=Conv1D(256, 5, strides=1, padding='same')(x)
    
    x=Conv1D(128, 5, strides=1, padding='same')(x)
    x=Conv1D(128, 5, strides=1, padding='same')(x)
    
    x=Conv1D(64, 3, strides=1, padding='same')(x)
    x=Conv1D(64, 3, strides=1, padding='same')(x)
    x=Flatten(name='side_out')(x)
    x=Dense(num_classes)(x)
    
    side=CenterLossLayer(alpha=0.9, name='centerlosslayer')([x, labels])
    y=Activation('softmax', name='main_out')(x)
    
    return y, side

def get_DNN(x, num_classes):

    x=Dense(256)(x)
    x=Dense(128)(x)
    x=Dense(64)(x)
    x=Dense(3)(x)
    x=Activation('softmax')(x)
        
    return x
    
__factory = {
    '1': get_model_output,  
    '3': get_DNN
}

def create(name, inputs, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
   
    return __factory[name](inputs, num_classes)    
    
