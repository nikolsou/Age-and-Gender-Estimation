# Life is incomplete without this statement!
import tensorflow as tf

# And this as well!
import numpy as np

# To visualize results
import matplotlib.pyplot as plt

import os
import datetime 

# Image size for our model.
MODEL_INPUT_IMAGE_SIZE = [ 128 , 128 ]

# Fraction of the dataset to be used for testing.
TRAIN_TEST_SPLIT = 0.3

# Number of samples to take from dataset
NUM_SAMPLES = 20000

# This method will be mapped for each filename in `list_ds`. 
def parse_image( filename ):

    # Read the image from the filename and resize it.
    image_raw = tf.io.read_file( filename )
    image = tf.image.decode_jpeg( image_raw , channels=3 ) 
    image = tf.image.resize( image , MODEL_INPUT_IMAGE_SIZE ) / 255

    parts = tf.strings.split( tf.strings.split( filename , '/' )[ -1 ] , '_' )
    #gender_str = parts[1]  # Get the second element, the gender

   # One-hot encode the label
    gender = tf.strings.to_number( parts[ 1 ] )

    return image , gender

# List all the image files in the given directory.
list_ds = tf.data.Dataset.list_files( '/kaggle/input/utkface-new/UTKFace/*' , shuffle=True )
# Map `parse_image` method to all filenames.
dataset = list_ds.map( parse_image , num_parallel_calls=tf.data.AUTOTUNE )
dataset = dataset.take( NUM_SAMPLES )

# Create train and test splits of the dataset.
num_examples_in_test_ds = int( dataset.cardinality().numpy() * TRAIN_TEST_SPLIT )

test_ds = dataset.take( num_examples_in_test_ds )
train_ds = dataset.skip( num_examples_in_test_ds )

print( 'Num examples in train ds {}'.format( train_ds.cardinality() ) )
print( 'Num examples in test ds {}'.format( test_ds.cardinality() ) )


def dense( x , filters , dropout_rate ):
    x = tf.keras.layers.Dense( filters , kernel_regularizer=tf.keras.regularizers.L2( 0.1 ) , bias_regularizer=tf.keras.regularizers.L2( 0.1 ) )( x )
    x = tf.keras.layers.LeakyReLU( alpha=leaky_relu_alpha )( x )
    x = tf.keras.layers.Dropout( dropout_rate )( x )
    return x

# Custom preprocessing layer to modify image brightness randomly.
class RandomBrightness( tf.keras.layers.Layer ):

    def __init__( self , max_delta ):
        super( RandomBrightness , self ).__init__()
        self.__max_delta = max_delta

    def call( self , inputs ):
        return tf.image.random_brightness( inputs , self.__max_delta )

    def get_config( self ):
        return { "max_delta" : self.__max_delta }
        
       
# Negative slope coefficient for LeakyReLU.
leaky_relu_alpha = 0.2

lite_model = True 

# Define the conv block.
def conv( x , num_filters , kernel_size=( 3 , 3 ) , strides=1 ):
    if lite_model:
        x = tf.keras.layers.SeparableConv2D( num_filters ,
                                            kernel_size=kernel_size ,
                                            strides=strides, 
                                            use_bias=False ,
                                            kernel_initializer=tf.keras.initializers.HeNormal() ,
                                            kernel_regularizer=tf.keras.regularizers.L2( 1e-5 )
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

def dense( x , filters , dropout_rate ):
    x = tf.keras.layers.Dense( filters , kernel_regularizer=tf.keras.regularizers.L2( 0.1 ) , bias_regularizer=tf.keras.regularizers.L2( 0.1 ) )( x )
  
    x = tf.keras.layers.LeakyReLU( negative_slope=leaky_relu_alpha )( x )
    x = tf.keras.layers.Dropout( dropout_rate )( x )
    return x


# No. of convolution layers to be added.
num_blocks = 5
# Num filters for each conv layer.
num_filters = [ 16 , 32 , 64 , 128 , 256 , 256 ]
# Kernel sizes for each conv layer.
kernel_sizes = [ 3 , 3 , 3 , 3 , 3 , 3 ]

# Init a Input Layer.
inputs = tf.keras.layers.Input( shape=MODEL_INPUT_IMAGE_SIZE + [ 3 ] )


preprocessing_module = [
    tf.keras.layers.RandomFlip( mode='vertical' ) , 
    RandomBrightness( max_delta=0.2 ) , 
    tf.keras.layers.RandomRotation( factor=0.1 )
]

x = inputs
for layer in preprocessing_module:
    x = layer( x )

# Add conv blocks sequentially
for i in range( num_blocks ):
    x = conv( x , num_filters=num_filters[ i ] , kernel_size=kernel_sizes[ i ] )
    x = tf.keras.layers.MaxPooling2D()( x )

# Flatten the output of the last Conv layer.
x = tf.keras.layers.Flatten()( x )
conv_output = x 

# Add Dense layers ( Dense -> LeakyReLU -> Dropout )
x = dense( conv_output , 256 , 0.6 )
x = dense( x , 64 , 0.4 )
x = dense( x , 32 , 0.2 )
outputs = tf.keras.layers.Dense( 2 , activation='softmax' )( x )

# Build the Model
model = tf.keras.models.Model( inputs , outputs )

# Uncomment the below to view the summary of the model.
model.summary()
# tf.keras.utils.plot_model( model , to_file='architecture.png' )



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
                                            kernel_initializer=tf.keras.initializers.HeNormal() ,
                                            kernel_regularizer=tf.keras.regularizers.L2( 1e-5 )
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

def dense( x , filters , dropout_rate ):
    x = tf.keras.layers.Dense( filters , kernel_regularizer=tf.keras.regularizers.L2( 0.1 ) , bias_regularizer=tf.keras.regularizers.L2( 0.1 ) )( x )
  
    x = tf.keras.layers.LeakyReLU( negative_slope=leaky_relu_alpha )( x )
    x = tf.keras.layers.Dropout( dropout_rate )( x )
    return x


# No. of convolution layers to be added.
num_blocks = 5
# Num filters for each conv layer.
num_filters = [ 16 , 32 , 64 , 128 , 256 , 256 ]
# Kernel sizes for each conv layer.
kernel_sizes = [ 3 , 3 , 3 , 3 , 3 , 3 ]

# Init a Input Layer.
inputs = tf.keras.layers.Input( shape=MODEL_INPUT_IMAGE_SIZE + [ 3 ] )


preprocessing_module = [
    tf.keras.layers.RandomFlip( mode='vertical' ) , 
    RandomBrightness( max_delta=0.2 ) , 
    tf.keras.layers.RandomRotation( factor=0.1 )
]

x = inputs
for layer in preprocessing_module:
    x = layer( x )

# Add conv blocks sequentially
for i in range( num_blocks ):
    x = conv( x , num_filters=num_filters[ i ] , kernel_size=kernel_sizes[ i ] )
    x = tf.keras.layers.MaxPooling2D()( x )

# Flatten the output of the last Conv layer.
x = tf.keras.layers.Flatten()( x )
conv_output = x 

# Add Dense layers ( Dense -> LeakyReLU -> Dropout )
x = dense( conv_output , 256 , 0.6 )
x = dense( x , 64 , 0.4 )
x = dense( x , 32 , 0.2 )
outputs = tf.keras.layers.Dense( 2 , activation='softmax' )( x )

# Build the Model
model = tf.keras.models.Model( inputs , outputs )

# Uncomment the below to view the summary of the model.
model.summary()
# tf.keras.utils.plot_model( model , to_file='architecture.png' )


learning_rate = 0.0001
num_epochs = 10 
batch_size = 128

train_ds = train_ds.batch( batch_size ).repeat( num_epochs )
test_ds = test_ds.batch( batch_size ).repeat( num_epochs )

save_dir = 'train-1/cp.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( save_dir )

logdir = os.path.join( "tb_logs" , datetime.datetime.now().strftime("%Y%m%d-%H%M%S") )
tensorboard_callback = tf.keras.callbacks.TensorBoard( logdir )

early_stopping_callback = tf.keras.callbacks.EarlyStopping( monitor='val_accuracy' , patience=3 )

model.compile( 
    loss=tf.keras.losses.sparse_categorical_crossentropy , 
    optimizer = tf.keras.optimizers.Adam( learning_rate ) , 
    metrics =[ 'accuracy' ]
)

model.fit( 
    train_ds, 
    epochs=num_epochs,  
    validation_data=test_ds, callbacks=[ checkpoint_callback , tensorboard_callback , early_stopping_callback ]
)

p = model.evaluate( test_ds )
print( 'loss is {} \n accuracy is {} %'.format( p[0] , p[1] * 100 ) )


fig = plt.figure( figsize=( 12 , 15 ) )
classes = [ 'Male' , 'Female' ]
rows = 5
columns = 2

i = 1
for image , label in test_ds.unbatch().take( 10 ):
    print(label)
    image = image.numpy()
    fig.add_subplot( rows , columns , i )
    plt.imshow( image )
    label_ = classes[ np.argmax( model.predict( np.expand_dims( image , 0 ) ) ) ]
    plt.axis( 'off' )
    plt.title( 'Predicted gender : {} , actual gender : {}'.format( label_ , classes[int(label) ] ) )
    i += 1









