import pathlib
import cv2
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras import layers, models 
from keras.utils import to_categorical

class ChestDiseaseClassification(object):
    """ Class For Chest Image Classification 
    """

    def __init__(self, Normal_TrainImage_path, Pneumonia_TrainImage_path,
                 TB_TrainImage_path, TestImage_path, epochs = 10, batch_size = 16):

        self.epochs = epochs 
        self.batch_size = batch_size 
        self.Normal_TrainImage_path = Normal_TrainImage_path
        self.Pneumonia_TrainImage_path = Pneumonia_TrainImage_path
        self.TB_TrainImage_path = TB_TrainImage_path
        self.TestImage_path = TestImage_path

        self.train_data = []
        self.train_labels = []
   
    def Convert_Preprocess_train(self, path, label):
        """ Convert the dataset in the required Format 

        """

        for image in path:
 
            image = cv2.imread(str(image))  
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)/255.
            self.train_data.append(image)
            self.train_labels.append(label)
    
        return self.train_data, self.train_labels 

    def Convert_Preprocess_test(self, path):
        """ Convert the dataset in the required Format 

        """
        data_list = []

        for image in path:
 
            image = cv2.imread(str(image))  
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)/255.
            data_list.append(image)
    
        return data_list 

    def Load_Data(self):
        """ 
        """

        # Load Normal Train Dataset into the Train data list
        Normal_TrainDir_path = pathlib.Path(self.Normal_TrainImage_path)
        normal_image_path = Normal_TrainDir_path.glob('*.jpg')
        label_normal = 0 
        normal_data, normal_labels = self.Convert_Preprocess_train(normal_image_path, label_normal)


        # Pneumonia Train Data 
        Pneumonia_TrainDir_path = pathlib.Path(self.Pneumonia_TrainImage_path)
        pneumonia_image_path = Pneumonia_TrainDir_path.glob('*.jpg')
        label_pneumonia = 1 
        pneumonia_data, pneumonia_labels = self.Convert_Preprocess_train(pneumonia_image_path, label_pneumonia)


        # TB Train Data 
        TB_TrainDir_path = pathlib.Path(self.TB_TrainImage_path)
        tb_image_path = TB_TrainDir_path.glob('*.jpg')
        label_tb = 2 
        tb_data, tb_labels = self.Convert_Preprocess_train(tb_image_path, label_tb)


        # Test Data 
        TestDir_path = pathlib.Path(self.TestImage_path)
        test_path = TestDir_path.glob('*.jpg')
        label_normal = 0 
        test_data = self.Convert_Preprocess_test(test_path)

        # Convert list to Numpy array 
        train_data_arr = np.array(self.train_data)
        train_labels_arr = np.array(self.train_labels)
        test_data_arr = np.array(test_data)

        return train_data_arr, test_data_arr, train_labels_arr 

    def CNN_Model(self): 
        """
        """

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(64, (3,3), activation='relu',padding='same', input_shape = (224, 224, 3)))
        self.model.add(layers.Conv2D(64, (3,3), activation= 'relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(64, (3,3), activation= 'relu'))
        self.model.add(layers.Flatten())
        # model.add(layers.Dense(1024, activation = 'relu'))
        self.model.add(layers.Dense(516, activation = 'relu'))
        # model.add(layers.Dense(256, activation = 'relu'))
        self.model.add(layers.Dense(128, activation = 'relu'))
        # model.add(layers.Dense(64, activation = 'relu'))
        self.model.add(layers.Dense(32, activation = 'relu'))
        self.model.add(layers.Dense(3, activation = 'softmax'))

        self.model.summary()

        return self 
    
    def train_CNN(self, X_train, y_train, X_val, y_val): 
        """
        """
        self.CNN_Model()
        self.model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        history = self.model.fit(X_train, y_train, batch_size = 10, epochs = 10, validation_data = (X_val, y_val))

        return history 
    
    def Model_loss_Evaluation(self, history_model, model_eva):
        """ Loss Plot for Model 
        Parameter: 
        -----------
        history_model: history of the model 
        model_eva: Name of the Model 

        """
        plt.plot(history_model.history['loss'], color = 'm', label = 'Training Loss')
        plt.plot(history_model.history['val_loss'], color = 'r', label = "Val Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Evaluation for {0}'.format(model_eva))
        plt.legend(loc = 'best')
        plt.show()

    def Model_Accuracy_Evaluation(self, history_model, model_eva):
        """ Accracy Plot for the Model
        Parameter: 
        -----------
        history_model: history for the model 
        model_eva: Name of the Model   
        """
        plt.plot(history_model.history['accuracy'], color = 'k', label = " Accuracy ")
        plt.plot(history_model.history['val_accuracy'], color = 'g', label = 'Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Score ')
        plt.title('Model Accuracy Evaluation for {0}'.format(model_eva))
        plt.legend(loc = 'best')
        plt.show()

if __name__ == "__main__":

    Normal_Train_imagedir = 'Dataset/train/Normal'
    Pneumonia_Train_imagedir = 'Dataset/train/Pneumonia'
    TB_Train_imagedir = 'Dataset/train/TB'
    
    Test_imagedir = 'Dataset/test'

    cic = ChestDiseaseClassification(Normal_Train_imagedir, Pneumonia_Train_imagedir, TB_Train_imagedir, Test_imagedir)
    data_train, data_test, train_labels = cic.Load_Data()

    # Check the Train and Test Dataset
    print("Shape of X_train Set: ", data_train.shape)
    print("Shape of y_train set: ", train_labels.shape)
    print("Shape of X_test Set: ", data_test.shape)
    # print("Shape of y_test set: ", test_labels.shape)

    plt.imshow(data_train[0])
    # plt.imshow(data_train[110])
    # plt.imshow(data_train[240])
    plt.show()

