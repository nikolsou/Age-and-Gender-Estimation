
# Life is incomplete without this statement!
import tensorflow as tf

# And this as well!
import numpy as np

# To visualize results
import matplotlib.pyplot as plt

import os
import datetime # To download checkpoints, Keras models, TFLite models
from google.colab import files

# Image size for our model.
MODEL_INPUT_IMAGE_SIZE = [ 200 , 200 ]

# Fraction of the dataset to be used for testing.
TRAIN_TEST_SPLIT = 0.3

# Number of samples to take from dataset
N = 20000

# This method will be mapped for each filename in `list_ds`.
def parse_image( filename ):

    image_raw = tf.io.read_file( filename )
    image = tf.image.decode_jpeg( image_raw , channels=3 )
    image = tf.image.resize( image , MODEL_INPUT_IMAGE_SIZE ) / 255

    # Split the filename to get the age and the gender. Convert the age ( str ) and the gender ( str ) to dtype float32.
    parts = tf.strings.split( tf.strings.split( filename , '/' )[ -1 ] , '_' )
    age_str = parts[0]  # Get the first element, which is the age
    #print(age_str)
    # Convert age to float and normalize
    age = tf.strings.to_number(age_str, out_type=tf.float32) / 116.0
    #print(tf.strings.to_number(age_str, out_type=tf.float32))  # Normalize by 116
    return image , age

# List all the image files in the given directory.
list_ds = tf.data.Dataset.list_files( '/kaggle/input/utkface-new/UTKFace/*' , shuffle=True )

# Map `parse_image` method to all filenames.
dataset = list_ds.map( parse_image , num_parallel_calls=tf.data.AUTOTUNE )
dataset = dataset.take( N )


print(dataset)

num_examples_in_test_ds = int( dataset.cardinality().numpy() * TRAIN_TEST_SPLIT ) #TRAIN_TEST_SPLIT=0.3

test_ds = dataset.take( num_examples_in_test_ds )
train_ds = dataset.skip( num_examples_in_test_ds )

print( 'Num examples in train ds {}'.format( train_ds.cardinality() ) )
print( 'Num examples in test ds {}'.format( test_ds.cardinality() ) )

# Negative slope coefficient for LeakyReLU.
leaky_relu_alpha = 0.2

lite_model = False

# Define the conv block.
def conv( x , num_filters , kernel_size=( 3 , 3 ) , strides=1 ):
    if lite_model:
        x = tf.keras.layers.SeparableConv2D( num_filters ,
                                            kernel_size=kernel_size ,
                                            strides=strides,
                                            use_bias=False ,
                                            depthwise_initializer=tf.keras.initializers.HeNormal(),  # Use depthwise_initializer
                                            pointwise_initializer=tf.keras.initializers.HeNormal(),  # Use pointwise_initializer ,
                                            depthwise_regularizer=tf.keras.regularizers.L2(1e-5),  # Use depthwise_regularizer
                                            pointwise_regularizer=tf.keras.regularizers.L2(1e-5)
                                             )( x )
    else:
        x = tf.keras.layers.Conv2D( num_filters ,
                                   kernel_size=kernel_size ,
                                   strides=strides ,
                                   use_bias=False ,
                                   kernel_initializer=tf.keras.initializers.HeNormal() ,
                                   kernel_regularizer=tf.keras.regularizers.L2( 1e-5 )
                                    )( x )


    x = tf.keras.layers.BatchNormalization()( x )
    x = tf.keras.layers.LeakyReLU( leaky_relu_alpha )( x )
    return x

# Αkολουθεί ο ορισμός της συνάρτησης dense
def dense( x , filters , dropout_rate ):
    x = tf.keras.layers.Dense( filters , kernel_regularizer=tf.keras.regularizers.L2( 0.1 ) ,
                              bias_regularizer=tf.keras.regularizers.L2( 0.1 ) )( x )
    x = tf.keras.layers.LeakyReLU( alpha=leaky_relu_alpha )( x )
    x = tf.keras.layers.Dropout( dropout_rate )( x )
    return x

# No. of convolution layers to be added.
num_blocks = 6
# Num filters for each conv layer.
num_filters = [ 32 , 32 , 64 , 128 , 256 , 256 ]
# Kernel sizes for each conv layer.
kernel_sizes = [ 3 , 3 , 3 , 3 , 3 , 3 ]

# Init a Input Layer.
inputs = tf.keras.layers.Input( shape=MODEL_INPUT_IMAGE_SIZE + [ 3 ] )

# Add conv blocks sequentially
x = inputs
for i in range( num_blocks ):
    x = conv( x , num_filters=num_filters[ i ] , kernel_size=kernel_sizes[ i ] )
    x = tf.keras.layers.MaxPooling2D()( x )


# Flatten the output of the last Conv layer.


spatial_dim = x.shape[1] * x.shape[2]

x = tf.keras.layers.Reshape((spatial_dim, x.shape[-1]))(x)

x = tf.keras.layers.Flatten()( x )

conv_output = x
# Add Dense layers ( Dense -> LeakyReLU -> Dropout )
x = dense( conv_output , 128 , 0.6 )
x = dense( x , 64 , 0.4 )
x = dense( x , 32 , 0.2 )
outputs = tf.keras.layers.Dense( 1 , activation='relu' )( x )

# Build the Model
model = tf.keras.models.Model( inputs , outputs )

# Uncomment the below to view the summary of the model.
#model.summary()

# Initial learning rate
learning_rate = 0.001

num_epochs = 50 #@param {type: "number"}
batch_size = 128 #@param {type: "number"}

# Batch and repeat `train_ds` and `test_ds`.
train_ds = train_ds.batch( batch_size )
test_ds = test_ds.batch( batch_size )

# Init ModelCheckpoint callback
save_dir_ = 'model_1'  #@param {type: "string"}
save_dir = save_dir_ + '/{epoch:02d}-{val_mae:.2f}.keras'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    save_dir ,
    save_best_only=True ,
    monitor='val_mae' ,
    mode='min' ,
)

tb_log_name = 'model_1'  #@param {type: "string"}
# Init TensorBoard Callback
logdir = os.path.join( "tb_logs" , tb_log_name )
tensorboard_callback = tf.keras.callbacks.TensorBoard( logdir )

# Init LR Scheduler
def scheduler( epochs , learning_rate ):
    if epochs < num_epochs * 0.25:
        return learning_rate
    elif epochs < num_epochs * 0.5:
        return 0.0005
    elif epochs < num_epochs * 0.75:
        return 0.0001
    else:
        return 0.000095

lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler( scheduler )

# Init Early Stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping( monitor='val_mae' , patience=10 )

# Compile the model
from tensorflow.keras.losses import MeanAbsoluteError  # Import the class

model.compile(
    loss=MeanAbsoluteError(),  # Instantiate the class
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    metrics=['mae']
)

model.fit(
    train_ds,
    epochs=num_epochs,
    validation_data=test_ds,
    batch_size=batch_size,
    callbacks=[ checkpoint_callback , tensorboard_callback , lr_schedule_callback , early_stopping_callback ]
)

p = model.evaluate( test_ds )
print( p )

fig = plt.figure( figsize=( 10 , 15 ) )
rows = 5
columns = 2

i = 1
for image , label in test_ds.unbatch().take( 10 ):
    image = image.numpy()
    fig.add_subplot( rows , columns , i )
    plt.imshow( image )
    label_ = int( model.predict( np.expand_dims( image , 0 ) ) * 116 )
    plt.axis( 'off' )
    plt.title( 'Predicted age : {} , actual age : {}'.format( label_ , int( label.numpy() * 116 ) ) )
    i += 1