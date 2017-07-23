from __future__ import print_function
import keras
from keras.models import Sequential
import keras.utils.np_utils as kutils
from keras.layers import Dense, Dropout, Flatten, Activation, TimeDistributed, Bidirectional
from keras.layers import Conv2D, MaxPooling2D, LSTM, Masking, Embedding, BatchNormalization
import os, sys
import numpy as np
import keras.backend as K
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
#from sklearn.metrics import roc_auc_score



os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


batch_size = 25
num_class = 2
epochs = 20
size = 300
time_steps = 2000
length = 19

def f1_score(y_true, y_pred):
    num_tp = K.sum(K.argmax(y_true)*K.argmax(y_pred))
    num_fn = K.sum(K.argmax(y_true)*(1.0-K.argmax(y_pred)))
    num_fp = K.sum((1.0-K.argmax(y_true))*K.argmax(y_pred))
    num_tn = K.sum((1.0-K.argmax(y_true))*(1.0-K.argmax(y_pred)))

    f1 = 2*num_tp/(2*num_tp + num_fn + num_fp)
    return f1

def Net_model():
    model = Sequential()
    model.add(Masking(mask_value = 0.0, input_shape = input_shape))
    model.add(TimeDistributed(Dense(128)))
#    model.add(Dropout(0.3))
#    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences = True)))
#    model.add(Dropout(0.3))
#    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences = True)))
#    model.add(Dropout(0.3))
#    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(2, activation = 'softmax')))
#    model.add(Dropout(0.3))

    print("Compiling model...")
    model.compile(optimizer = 'adam',
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy'],
    sample_weight_mode = "temporal")

    arch = model.to_json()
    open('LSTM_arch.json', 'w').write(arch)
    model.summary()
    
    
    return model

def Train_model(model, trainX, trainY, train_label):
    print("Training model...")
    #model.load_weights('weights_16-0.80.hdf5')
    checkPointer = ModelCheckpoint('weights_{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only = True)
    earlyStopper = EarlyStopping(monitor='val_loss', patience= 500)
    reduceLROnPalteau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    csvlogger = CSVLogger('loss_log.csv', separator=',', append=False)

    #tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size= batch_size, write_grads = True, write_images = True)

    sample_weights = train_label[:,:,0]
    sample_weights[sample_weights == 1] = 5
    sample_weights[sample_weights == 0] = 1

    model.fit(trainX, trainY, 
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.11,
    shuffle = True,
    #callbacks = [checkPointer, earlyStopper, reduceLROnPalteau, csvlogger, tensorboard],
    callbacks = [checkPointer, earlyStopper, reduceLROnPalteau, csvlogger],
    #sample_weight = sample_weights,
    initial_epoch = 0)
    model.save_weights('LSTM_weights.hdf5', overwrite = True)

def Test_model(model, test_X, test_Y):
#    model.load_weights('weights_00-0.00.hdf5')
    score = model.evaluate(test_X, test_Y, batch_size = batch_size)

    print(score)
    print(model.metrics_names)

def Predict_model(model, test_X, test_label):

    classes = model.predict_classes(test_X).flatten()
    pred = model.predict(test_X)[:, :, 1]
    np.savetxt('data\\data250_pred.txt', pred)
    testlabel = test_label.flatten()
    #ROC-AUC
#    print("roc_auc = ", roc_auc_score(testlabel, pred))
    #TP/FN/FP/TN
    y_true = testlabel
    y_pred = classes
    num_tp = np.sum(y_true*y_pred)
    num_fn = np.sum(y_true*(1.0-y_pred))
    num_fp = np.sum((1.0-y_true)*y_pred)
    num_tn = np.sum((1.0-y_true)*(1.0-y_pred))    

    print("tp = ", num_tp)
    print("fn = ", num_fn)
    print("fp = ", num_fp)
    print("tn = ", num_tn)
    #PREC/RECALL/F1
    test_prec = num_tp/(num_tp + num_fp)
    test_recall = num_tp/(num_tp + num_fn)
    test_f1 = 2*num_tp/(2*num_tp + num_fn + num_fp)

    print('acc = ', test_prec)
    print('recall = ', test_recall)
    print('f1 = ', test_f1)
    

def load_data(data_name):
    data = np.load(data_name)
    X = data['X']
    X = kutils.normalize(X, order = 2, axis = 0)
    label = data['Y']
    Y = np.zeros((label.shape[0], label.shape[1], num_class))
    for i in range(label.shape[0]):
        Y[i] = kutils.to_categorical(label[i], num_class)
    return (X, label, Y)

if __name__ == "__main__":
    mode = sys.argv[1]
    data_file = sys.argv[2]
    if len(sys.argv) == 4 :
        weights_name = sys.argv[3]
    print('Loading data...')
    data_X, data_label, data_Y = load_data(data_file)
    train_X = data_X[:225]
    train_label = data_label[:225]
    train_Y = data_Y[:225]
    test_X = data_X[-25:]
    test_label = data_label[-25:]
    test_Y = data_Y[-25:]
    
    input_shape = (time_steps, length)
    model = Net_model()

    if (mode == 'train'):
        Train_model(model, train_X, train_Y, train_label)
        Test_model(model, test_X, test_Y)
        Predict_model(model, test_X, test_label)
    elif (mode == 'test'):
        model.load_weights(weights_name)
        Test_model(model, test_X, test_Y)
        Predict_model(model, test_X, test_label)
    elif (mode == 'train_test'):
        model.load_weights(weights_name)
        Test_model(model, train_X, train_Y)
        Predict_model(model, train_X, train_label)
    elif (model == 'plot'):
        Net_model()
    else:
        print("usage: LSTM.py train/test")
