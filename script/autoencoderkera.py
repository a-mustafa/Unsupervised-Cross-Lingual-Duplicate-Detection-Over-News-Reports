from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

'''
from keras.regularizers import activity_l1
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape
autoencoder.fit(x_train, x_train,epochs=50,batch_size=256,shuffle=True,validation_data=(x_test, x_test))
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


inputs = Input(shape=(data_l.shape[1],))
hencode = Dense(data_l.shape[1]/2, activation='relu')(inputs)
encoded = Dense(data_l.shape[1]/3, activation='relu', activity_regularizer=activity_l1(1e-5))(hencode)
hdecode = Dense(data_l.shape[1]/2, activation='relu')(encoded)
outputs = Dense(data_l.shape[1])(hdecode)
model = Model(input=inputs, output=outputs)

encoder = Model(inputs, encoded)
encoded_imgs = encoder.predict(x_test)


# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig(fname='/home/ahmad/duplicate-detection/script/mnst')
#plt.show()



'''




if __name__ == '__main__':
    
    if len(sys.argv) < 3:
    	print 'Usage: python pretrain_da.py datasetl datasetr learningRate'
    	print 'Example: python pretrain_da.py gauss basic 0.1'
    	sys.exit()
    
    
    #python autoencoderkeras.py /home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNEleftAttro2enes.txt /home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNErightAttro2enes.txt 0.1
    filepathl= str(sys.argv[1]) 
    filepathr = str(sys.argv[2])
    lr = float(sys.argv[3])
    
    ''' 
    log_folder = root_folder + '/logs/da_' + mm + '/layer'+str(input_ll+1)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)	
    logfile = open(os.path.join(log_folder, dd+',lr='+str(lr)+'.log'), 'w', 0)
    sys.stdout = logfile
    '''
    
    #filepathl='/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNEleftAttro2enes.txt'
    #filepathr='/home/ahmad/duplicate-detection/eventregistrydata/pairs/StanfordMatchingNErightAttro2enes.txt'
    leftNE=[]
    with open(filepathl,'r') as myfile:
      leftNE=myfile.readlines()
    
    rightNE=[]
    with open(filepathr,'r') as myfile:
      rightNE=myfile.readlines()
    
    leftNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in leftNE]
    rightNE=[item.replace("[u'","").replace("[","").replace("]","").replace("\', u\'"," ").replace("\'","").split() for item in rightNE]
    data_l=[]
    data_r=[]
    for _leftNE, _rightNE in zip(leftNE,rightNE):
      word1 = _leftNE[0]
      word2 = _rightNE[0]
      if '' !=word1.strip().lower() and '' !=word2.strip().lower():
        try:
          if type(word1)==type(''):
            word1=word1.strip().lower().decode('utf-8')
          else:
            word1=word1.strip().lower()
          
          if type(word2)==type(''):
            word2=word2.strip().lower().decode('utf-8')
          else:
            word2=word2.strip().lower()
          
          if embeddingsmodel['es'].__contains__(word2) and embeddingsmodel['en'].__contains__(word1):
            data_r.append(embeddingsmodel['es'][word2])
            data_l.append(embeddingsmodel['en'][word1])
        except:
          pass
    
    #del embeddingsmodel
    data_l=np.asarray(data_l)
    data_r=np.asarray(data_r)
    
    data_l = data_l / np.linalg.norm(data_l)
    data_r = data_r / np.linalg.norm(data_r)
    
    inputs = Input(shape=(data_l.shape[1],))
    hencode = Dense(data_l.shape[1]/2, activation='relu')(inputs)
    encoded = Dense(data_l.shape[1]/3, activation='relu', activity_regularizer=regularizers.l1(0.01))(hencode) #activity_l1(1e-5)
    hdecode = Dense(data_l.shape[1]/2, activation='relu')(encoded)
    outputs = Dense(data_l.shape[1])(hdecode)
    encoder = Model(inputs, encoded)
    model = Model(input=inputs, output=outputs)
    model.compile(optimizer='adam', loss='mse')
    model.fit(data_l, data_r, batch_size=64, nb_epoch=5)
    
    transformedmodel=dict()
    for lng in embeddingsmodel.keys():
      transformedmodel[lng]=dict()
      for word in embeddingsmodel[lng].id2word:
        transformedmodel[word]=encoder.predict(embeddingsmodel[lng][word])
    
    cPickle.dump(transformedmodel, open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/FastTextAttro2enesNormMse1000epcs2Layers.p', 'wb'))
