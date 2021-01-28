import os
import glob
import torch
import cv2
import torch.optim as optim
from matplotlib import pyplot as plt

from siamese_net import *
from data_loader import Data_loader
from kjh_model import KJH_Model

def set_model():
    dataset_path = 'data'
    use_agumentation = False
    batch_size = 16
    epoch = 100

    data_loader = Data_loader(dataset_path=dataset_path, use_agumentation=use_agumentation)
    data_loader.split_train_datasets()

    pairs_of_images, labels = data_loader.load_data()

    embeddingNetSiamese = EmbeddingNetSiamese()

    optimizer = optim.Adam(embeddingNetSiamese.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0005, amsgrad=True)

    kjh_model = KJH_Model(data=pairs_of_images, label=labels, model=embeddingNetSiamese,
                          optimizer=optimizer, epochs=epoch,
                          batch_size=batch_size)

    return kjh_model

def train_model(model):
    # bring the paths of dataset
    drowsy_dataset_path = 'data/images_train/drowsy'
    normal_dataset_path = 'data/images_train/normal'

    # check how many data set were saved before
    file = open("data/record_of_num_of_dataset.txt", "r")
    is_empty = os.path.getsize("data/record_of_num_of_dataset.txt")

    latest_num_of_drowsy = 0
    latest_num_of_normal = 0

    if (is_empty != 0):
        latest_num_of_drowsy = int(file.readline().strip())
        latest_num_of_normal = int(file.readline().strip())

    file.close()

    # calculate current the amount of data set
    now_num_of_drowsy = len(os.listdir(drowsy_dataset_path))
    now_num_of_normal = len(os.listdir(normal_dataset_path))

    # if new drowsy data was saved, write how many data is
    if (now_num_of_drowsy > latest_num_of_drowsy
            or now_num_of_normal > latest_num_of_normal):

        with open("data/record_of_num_of_dataset.txt", "w") as file:
            file.write(str(now_num_of_drowsy) + "\n")
            file.write(str(now_num_of_normal))

        diff_drowsy = now_num_of_drowsy - latest_num_of_drowsy
        diff_normal = now_num_of_normal - latest_num_of_normal

        # if the difference between current data and previous data is greater than 16, train the model
        if ((diff_drowsy > 8 or diff_normal > 8)):
            model.fit()
            print("[INFO] train finished")
        else:
            print("[INFO] no train")
    else:
        print("[INFO] no train")


def histogram():
    img = cv2.imread("data/images_train/drowsy/1000.png")

    color = ("b", "g", "r")
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color = col)
        plt.xlim([0, 256])
    plt.show()

def main():

    # histogram()

    # if you want to make a model first
    #model = set_model()
    #model.fit()

    print("[INFO] System operation...")

    os.makedirs('checkpoints', exist_ok=True)

    model_path = 'checkpoints'

    kjh_model = set_model()

    num_of_models = len(os.listdir(model_path))

    if(num_of_models == 0):
        # if no model, try only blink detection
        kjh_model.try_camera(mode='blink detection')
    else:
        # bring latest model
        list_of_models = glob.glob('checkpoints/*')
        latest_model = max(list_of_models, key=os.path.getctime)
        checkpoint = torch.load(latest_model)
        kjh_model.try_camera(mode='behavior detection', checkpoint=checkpoint)

    new_model = set_model()
    train_model(new_model)

    print("[INFO] system down...")

if __name__ == '__main__':
    main()