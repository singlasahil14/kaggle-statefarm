from utils import *

path = "data/"
#path = "data/state/sample/"
model_path = path + 'models/'

#test_batches = get_batches(path+'test', batch_size=batch_size*2, shuffle=False)
#(val_classes, trn_classes, val_labels, trn_labels, 
#    val_filenames, filenames, test_filenames) = get_classes(path)

#trn = get_data(path+'train')
#val = get_data(path+'valid')

#save_array(path+'results/val.dat', val)
#save_array(path+'results/trn.dat', trn)

trn = load_array(path+'results/trn.dat')
val = load_array(path+'results/val.dat')
print(trn.shape)
print(val.shape)

(val_classes, trn_classes, val_labels, trn_labels, val_filenames, filenames, test_filenames) = get_classes(path)
print(trn_labels.shape)
print(val_labels.shape)

vgg = Vgg16()
model=vgg.model
last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D][-1]
conv_layers = model.layers[:last_conv_idx+1]
conv_model = Sequential(conv_layers)


#batch_size=64
#batches = get_batches(path+'train', batch_size=batch_size, shuffle=False)
#val_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)

batch_size=64
gen = image.ImageDataGenerator()
batches = gen.flow(trn, trn_labels, batch_size=batch_size, shuffle=False)
val_batches = gen.flow(val, val_labels, batch_size=2*batch_size, shuffle=False)
conv_feat = conv_model.predict_generator(batches, trn.shape[0])
conv_val_feat = conv_model.predict_generator(val_batches, val.shape[0])
#conv_test_feat = conv_model.predict_generator(test_batches, test_batches.nb_sample)

save_array(path+'results/conv_feat.dat', conv_feat)
save_array(path+'results/conv_val_feat.dat', conv_val_feat)
#save_array(path+'results/conv_test_feat.dat', conv_test_feat)

#conv_feat = load_array(path+'results/conv_feat.dat')
#conv_val_feat = load_array(path+'results/conv_val_feat.dat')
print(conv_feat.shape)
print(conv_val_feat.shape)

def get_bn_layers(p):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dropout(p/2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
        ]

p=0.8
bn_model = Sequential(get_bn_layers(p))
bn_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=10, 
             validation_data=(conv_val_feat, val_labels))

bn_model.optimizer.lr=0.0001
bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=20, 
             validation_data=(conv_val_feat, val_labels))
bn_model.save_weights(model_path+'conv8.h5')
