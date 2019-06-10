import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.python.keras.regularizers import L1L2
from tensorflow.python.keras.optimizers import  adam
from  sklearn.model_selection import  train_test_split
from sklearn.preprocessing import  LabelEncoder,normalize
import  pandas as pd
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from  tensorflow.python import  keras


import  os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def save_best_model(epoch, dir_path, num_ext, ext):
    tmp_file_name = os.listdir(dir_path)
    test = []
    num_element = -num_ext

    for x in range(0, len(tmp_file_name)):
        test.append(tmp_file_name[x][:num_element])
        float(test[x])

    highest = max(test)

    return str(highest) + ext

def train_test():
    data_frame = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Crops.csv'), sep=",")
    data_frame = data_frame.reindex(np.random.permutation(data_frame.index))

    selected_features = data_frame[
        ["Temperature",
         "Humidity"]]

    scaler = MinMaxScaler()
    scaler.fit(selected_features)
    selected_features = scaler.transform(selected_features)


    target_class = data_frame['Crop']

    label_encoder = LabelEncoder()
    label_encoder.fit(target_class)
    labels = label_encoder.transform(target_class)


    encoded_labels = np_utils.to_categorical(labels)

    X_train, X_test, y_train, y_test = train_test_split(np.asarray(selected_features), encoded_labels, test_size=0.33)

    checkpoint = ModelCheckpoint('output/{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,save_best_only=True, save_weights_only=True,
                                 mode='auto')



    model = Sequential()
    model.add(Dense(50, input_dim=2, activation='relu',name="dense_in"))
    model.add(Dense(100, activation='relu',name="dense_in_2"))
    model.add(Dense(3,
                    activation='softmax',name="dense_in_3"))


    optimizer = adam(lr=0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    tbCallBack = TensorBoard(log_dir="Graph", histogram_freq=0, write_graph=True, write_images=True)
    model.fit(X_train, y_train, epochs=100, batch_size=5, validation_data=(X_test, y_test), verbose=2,callbacks=[tbCallBack,checkpoint])

    keras.models.save_model(model,"saved_model.h5",overwrite=True)
    model.save("best.h5",overwrite=True,save_weights_only=True)
    predict = model.predict(np.asarray([0.6, 0.8]).reshape(1,-1))



def convert_to_lite():
    import tensorflow as tf

    # Convert to TensorFlow Lite model.
    keras_file="saved_model.h5"
    converter =tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    print ("Conversion successful")



if __name__ == '__main__':
    #convert_to_lite()
    train_test()
