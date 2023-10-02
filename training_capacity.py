import argparse
import io
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from skimage.io import imread
from shutil import copyfile
#import tensorflow_probability as tfp
print(tf.__version__)
import matplotlib.pyplot as plt
plt.gray()
from sklearn.metrics import accuracy_score
#import tensorflow_probability as tfp
import random
import json
import datetime

adjective=['adorable','adventurous','aggressive','agreeable','angry','annoyed','anxious','arrogant','attractive','awful','bad','bloody','bored','brave','calm','dull']
colors= ['black','red','orange','purple','yellow','gray','green','white','pink','blue','magenta','gold','ocre','brown','silver']
words = ['sun','ball','moon','earth','grass','world','sea','chess','car','jupiter','chicken','river','table','moto','fontain','three']
number = ['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','quince','uno','tres','cuatro']

parser = argparse.ArgumentParser(description='Results in Neyman-Generated',epilog='Enjoy! S. Velasco-Forero')
parser.add_argument('--pathoutput',type=str, default='/gpfsstore/rech/qpj/uyn98cq/choquet/', help='path output')
parser.add_argument('--batch', type=float, default=32., help='batch_size')
parser.add_argument('--epochs',  type=float, default=512.,   help='epochs')
parser.add_argument('--nlayers', type=float, default=1, help='nlayers')
parser.add_argument('--nfilters', type=float, default=48, help='nfilters')
parser.add_argument('--ksize', type=float, default=13, help='ksize')
parser.add_argument('--featurespace', type=float, default=12, help='featurespace')
parser.add_argument('--generatedata', type=float, default=1, help='generate')

args = parser.parse_args()


PATHOUT=args.pathoutput #'/gpfsstore/rech/qpj/uyn98cq/TransformationInvariant/'

BATCH_SIZE = int(args.batch)
EPOCHS = int(args.epochs)

batch_size = BATCH_SIZE
epochs = EPOCHS
learning_rate=0.001
CHANNELS=1
NLAYERS=int(args.nlayers)
NFILTERS=int(args.nfilters)
KSIZE=int(args.ksize)
SUBSPACE=int(args.featurespace)
PATIENCE_ES=40
PATIENCE_RP=5
pathoutput=str(args.pathoutput)
output_dir_root =pathoutput

test_name=random.choice(number)+random.choice(adjective)+random.choice(colors)+random.choice(words)
#timeDir=get_time()
dir_name = output_dir_root + "_" + test_name
print('dir_name',dir_name)

LOG=True
if LOG:
   os.makedirs(dir_name)
   dir_autosave_model_weights = os.path.join(dir_name, "autosave_model_weights")
   dir_autosave_model_stat = os.path.join(dir_name, "accuracy")
   os.makedirs(dir_autosave_model_weights)
   this_file_name = os.path.basename(__file__)
   copyfile(__file__, os.path.join(dir_name, this_file_name))
   print('copyfile','ok')
   #print('dir_name plus parameter  json')
   #print(os.path.join(dir_name, "parameter.json"))
   #outfile=open(os.path.join(dir_name, "parameter.json"), 'w' )
   #jsondic=vars(args)
   #jsondic["dir_name"] = dir_name
   #json.dump(jsondic,outfile)
   #outfile.close()

class NeymanScott:
    """
    Neyman-Scott point process using a Poisson variable for the number of parent points, uniform for
    the number of daughter points and Pareto distribution for the distance from the daughter points to
    the parent.
    """
    def __init__(self,
                 poisson_mean: float,
                 daughter_max: int,
                 pareto_alpha: float,
                 pareto_scale: float,
                 size: (int, int)):
        """
        :param poisson_mean: mean of the number of parent points
        :param daughter_max: maximum number of daughters per parent points
        :param pareto_alpha: alpha parameter of the Pareto distribution
        :param pareto_scale: scale used in the Pareto distribution. This parameter is
            applied before resizing the points from the [0, 1] interval to the size of the image.
        :param size: rescale the output to this size
        """
        self.poisson_mean = poisson_mean
        self.daughter_max = daughter_max
        self.pareto_alpha = pareto_alpha
        self.pareto_scale = pareto_scale
        self.size = np.array([size])
        self.generator = np.random.Generator(np.random.PCG64())

    def __call__(self):
        num_parents = self.generator.poisson(lam=self.poisson_mean)
        parents = self.generator.random((num_parents, 2))
        num_daughters = self.generator.integers(1, self.daughter_max, num_parents)
        points = np.empty((0, 2))

        for i in range(num_parents):
            # normalizes the pareto II distribution
            dist = self.generator.pareto(self.pareto_alpha, (num_daughters[i], 1))
            dist = (dist + 1) * self.pareto_scale
            angle = self.generator.uniform(0., 2 * np.pi, (num_daughters[i],))
            positions = np.stack([np.cos(angle), np.sin(angle)], 1)
            positions *= dist
            positions += parents[i, np.newaxis, :]
            points = np.concatenate([points, positions])
        # remove points outside the set [0, 1] x [0, 1]
        valid_points = np.logical_and(
            np.logical_and(0. <= points[:, 0], points[:, 0] <= 1.),
            np.logical_and(0. <= points[:, 1], points[:, 1] <= 1.)
        )
        points = points[valid_points, :]
        # scale to the image size
        points = points * self.size
        return points



NSAMPLES_TRAINING=2024*2
IMG_SIZE=128
poisson_mean=100
daughter_max=50
pareto_scale=.02
pareto_alpha=1. #GENERATION ON IT
gen = NeymanScott(poisson_mean, daughter_max, pareto_alpha, pareto_scale, (IMG_SIZE, IMG_SIZE))

if args.generatedata==1:
    listIm=[]
    listY=[]
    for i in range(NSAMPLES_TRAINING):
        pareto_alpha=gen.generator.random(1)*10
        gen = NeymanScott(poisson_mean, daughter_max, pareto_alpha, pareto_scale, (IMG_SIZE, IMG_SIZE))
        points = gen()
        I=np.zeros([IMG_SIZE,IMG_SIZE])
        I[np.int64(np.floor(points[:, 0])), np.int64(np.floor(points[:, 1]))]=1
        listIm.append(I)
        listY.append(pareto_alpha)
    listIm=np.stack(listIm)
    listY=np.stack(listY)

    NSAMPLES_VALIDATION=512
    listImVal=[]
    listYVal=[]
    for i in range(NSAMPLES_VALIDATION):
        pareto_alpha=gen.generator.random(1)*10
        gen = NeymanScott(poisson_mean, daughter_max, pareto_alpha, pareto_scale, (IMG_SIZE, IMG_SIZE))
        points = gen()
        I=np.zeros([IMG_SIZE,IMG_SIZE])
        I[np.int64(np.floor(points[:, 0])), np.int64(np.floor(points[:, 1]))]=1
        listImVal.append(I)
        listYVal.append(pareto_alpha)

    listImVal=np.stack(listImVal)
    listYVal=np.stack(listYVal)
    np.save(args.pathoutput+'listIm.npy',listIm)
    np.save(args.pathoutput+'listImVal.npy',listImVal)
    np.save(args.pathoutput+'listY.npy',listY)
    np.save(args.pathoutput+'listYVal.npy',listYVal)
else:
    listIm=np.load(args.pathoutput+'listIm.npy')
    listImVal=np.load(args.pathoutput+'listImVal.npy')
    listY=np.load(args.pathoutput+'listY.npy')
    listYVal=np.load(args.pathoutput+'listYVal.npy')

print('listIm.shape',listIm.shape)
print('listImVal.shape',listImVal.shape)
print('listY.shape',listY.shape)
print('listYVal.shape',listYVal.shape)

listY=listY/9
listYVal=listYVal/9

@tf.function
def dilation2d(x, st_element, strides, padding,rates=(1, 1)):
    """

    From MORPHOLAYERS

    Basic Dilation Operator
    :param st_element: Nonflat structuring element
    :strides: strides as classical convolutional layers
    :padding: padding as classical convolutional layers
    :rates: rates as classical convolutional layers
    """
    x = tf.nn.dilation2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
    return x

class DepthwiseDilation2D(Layer):
    '''
    Depthwise Dilation 2D Layer: Depthwise Dilation for now assuming channel last
    '''
    def __init__(self, kernel_size,depth_multiplier=1, strides=(1, 1),padding='same', dilation_rate=(1,1), kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=0.),
    kernel_constraint=None,kernel_regularization=None,**kwargs):
        super(DepthwiseDilation2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.depth_multiplier= depth_multiplier
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim,self.depth_multiplier)
        self.kernel2D = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel2D',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        super(DepthwiseDilation2D, self).build(input_shape)

    def call(self, x):
        res=[]
        for di in range(self.depth_multiplier):
            H=tf.nn.dilation2d(x,self.kernel2D[:,:,:,di],strides=(1, ) + self.strides + (1, ),padding=self.padding.upper(),data_format="NHWC",dilations=(1,)+self.rates+(1,))
            res.append(H)
        return tf.concat(res,axis=-1)

    def compute_output_shape(self, input_shape):

        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.depth_multiplier,)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'depth_multiplier': self.depth_multiplier,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config



xinput = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
xconv=layers.Conv2D(NFILTERS,(KSIZE,KSIZE),use_bias=False,padding='same')(xinput)
for i in range(NLAYERS):
    xconv = layers.Conv2D(NFILTERS,(3,3),padding='same',activation='relu')(xconv)
xfeatures=layers.GlobalAveragePooling2D()(xconv)
xfeatures=layers.BatchNormalization()(xfeatures)
xfeatures=layers.Dense(SUBSPACE,'relu')(xfeatures)
xfeatures=layers.Dense(SUBSPACE)(xfeatures)
xend=layers.Dense(1,activation='sigmoid')(xfeatures)
modelConv=tf.keras.Model(xinput,xend)
modelConv.summary()
print(modelConv.count_params())

CB1=[tf.keras.callbacks.EarlyStopping(patience=PATIENCE_ES,restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=PATIENCE_RP,min_lr=1e-6),
    tf.keras.callbacks.CSVLogger(dir_autosave_model_stat+'Conv', separator=',', append=False)
   ]
modelConv.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=["mse","mae"])
histConv=modelConv.fit(listIm, listY, batch_size=batch_size, epochs=epochs,callbacks=CB1,validation_data=(listImVal, listYVal))


xinput = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
xconv = DepthwiseDilation2D((KSIZE,KSIZE),depth_multiplier=NFILTERS,padding='same')(xinput)
for i in range(NLAYERS):
    xconv = layers.Conv2D(NFILTERS,(3,3),padding='same',activation='relu')(xconv)
xfeatures=layers.GlobalAveragePooling2D()(xconv)
xfeatures=layers.BatchNormalization()(xfeatures)
xfeatures=layers.Dense(SUBSPACE,activation='relu')(xfeatures)
xfeatures=layers.Dense(SUBSPACE)(xfeatures)
xend=layers.Dense(1,activation='sigmoid')(xfeatures)
modelDil=tf.keras.Model(xinput,xend)
modelDil.summary()
print(modelDil.count_params())

CB2=[tf.keras.callbacks.EarlyStopping(patience=PATIENCE_ES,restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=PATIENCE_RP,min_lr=1e-6),
    tf.keras.callbacks.CSVLogger(dir_autosave_model_stat+'Dil', separator=',', append=False)
   ]
modelDil.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=["mse","mae"])
histDil=modelDil.fit(listIm, listY, batch_size=batch_size, epochs=epochs,callbacks=CB2,validation_data=(listImVal, listYVal))

if LOG:
   print('dir_name plus parameter  json')
   print(os.path.join(dir_name, "parameter.json"))
   outfile=open(os.path.join(dir_name, "parameter.json"), 'w' )
   jsondic=vars(args)
   jsondic["dir_name"] = dir_name
   jsondic["num_parameters_CNN"]=modelConv.count_params()
   jsondic["num_parameters_Morpho"]=modelDil.count_params()
   jsondic["train_capacity"]=1
   json.dump(jsondic,outfile)
   outfile.close()
