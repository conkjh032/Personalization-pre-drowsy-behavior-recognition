import os
import math
import numpy as np
from PIL import Image

from image_augmentor import ImageAugmentor
from sklearn.model_selection import train_test_split

class Data_loader:

    def __init__(self, dataset_path, use_agumentation = False):
        self.dataset_path = dataset_path
        self.train_dictionary = {}
        self.evaluation_dictionary = {}
        self.image_width = 112
        self.image_height = 112
        self.use_agumentation = use_agumentation

        self._current_train_state_index = 0
        self._train_states = []
        self._validation_images = []
        self._evaluation_images = []
        self._train_image_paths = {}
        self._validation_image_paths = {}

        self.find_data_path()

        for state in ['drowsy', 'normal']:
            self._train_states.append(state)

        if (self.use_agumentation):
            self.image_augmentor = self.createAugmentor()
        else:
            self.use_agumentation = []

    # find all data path and gather
    def find_data_path(self):

        data_path = os.path.join(self.dataset_path, 'images_train')
        evaluation_path = os.path.join(self.dataset_path, 'images_evaluation')

        # state = drowsy / normal
        for state in os.listdir(data_path):
            train_list = []
            state_path = os.path.join(data_path, state)
            for image in os.listdir(state_path):
                image_path = os.path.join(state_path, image)
                train_list.append(image_path)
            self.train_dictionary[state] = train_list

        # state_path = data/drowsy, data/normal

        # evaluation_dictionary. I don't use it. I'd like to use validation
        # However, i don't have a lot of data
        for state in os.listdir(evaluation_path):
            evaluation_list = []
            state_path = os.path.join(evaluation_path, state)
            for image in os.listdir(state_path):
                image_path = os.path.join(state_path, image)
                evaluation_list.append(image_path)
            self.evaluation_dictionary[state] = evaluation_list

    # augmentation
    def createAugmentor(self):
        rotatation_range = [-10, 10]
        shear_range = [-0.3 * 180 / math.pi, 0.3 * 180 / math.pi]
        zoom_range = [0.8, 1.5]
        shift_range = [5, 5]

        return ImageAugmentor(0.5, shear_range, rotatation_range, zoom_range, shift_range)

    # split data into train and validation
    def split_train_datasets(self):
        """
        split the dataset in train and validation
        """

        # all drowsy images path
        drowsy_image_paths = self.train_dictionary['drowsy']

        # all normal images path
        normal_image_paths = self.train_dictionary['normal']

        # split
        drowsy_train_image_paths, drowsy_validation_image_paths = train_test_split(drowsy_image_paths, test_size=0.2, random_state=43)
        normal_train_image_paths, normal_validation_image_paths = train_test_split(normal_image_paths, test_size=0.2, random_state=43)

        self._train_image_paths['drowsy'] = drowsy_train_image_paths
        self._train_image_paths['normal'] = normal_train_image_paths
        self._validation_image_paths['drowsy'] = drowsy_validation_image_paths
        self._validation_image_paths['normal'] = normal_validation_image_paths

    # after collect all data path, it converts path list to image and label
    def _convert_path_list_to_images_and_labels(self, path_list, is_one_shot_task):

        num_of_pairs = len(path_list) // 2

        pairs_of_images = [np.zeros(
            (num_of_pairs, self.image_height, self.image_width, 3)) for i in range(2)]
        labels = np.zeros((num_of_pairs, ))

        for pair in range(num_of_pairs):
            # from the list AA AB AA AB, pick only front A
            image = Image.open(path_list[pair * 2])
            image = image.resize((self.image_height, self.image_width))
            # you might change data type to uint8 or others
            image = np.asarray(image).astype(np.float64)
            #image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - image.mean()) / image.std()
            pairs_of_images[0][pair, :, :, :] = image

            # from the list AA AB AA AB, pick rear A and B
            image = Image.open(path_list[pair * 2 + 1])
            image = image.resize((self.image_height, self.image_width))
            image = np.asarray(image).astype(np.float64)
            #image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - image.mean()) / image.std()
            pairs_of_images[1][pair, :, :, :] = image

            # if AA, 1, if AB, 0.
            if not is_one_shot_task:
                if (pair + 1) % 2 == 0:
                    labels[pair] = 0
                else:
                    labels[pair] = 1

            # is_one_shot_task 일 경우, paht_list는 AA AB AC AD AE AF ... 형식이다.
            # 첫 번째 AA에서만 1을 부여하고, 나머지는 0을 부여한다.
            else:
                if pair == 0:
                    labels[pair] = 1
                else:
                    labels[pair] = 0

        # shuffling
        if not is_one_shot_task:
            random_shuffle = np.random.permutation(num_of_pairs)
            labels = labels[random_shuffle]
            pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_shuffle, :, :, :]
            pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_shuffle, :, :, :]

        return pairs_of_images, labels

    def load_data(self):
        "make a list including AA AB AA AB ..."
        "make a matric with two channels which one for AAAA and another for ABABAB"
        "make label including 101010101010 ..."

        images_path = []

        current_state = self._train_states[self._current_train_state_index]
        num_of_images = len(self._train_image_paths[current_state])

        # In case the number of directory A's images is smaller than the number of directory B's images // 3
        # The number of directory A's images is smaller than the number of directory B's images // 3, then choose A, else B // 3
        if current_state == 'drowsy':
            tmp_state = 'normal'
            num_of_another_images = len(self._train_image_paths[tmp_state])
        else:
            tmp_state = 'drowsy'
            num_of_another_images = len(self._train_image_paths[tmp_state])

        if num_of_another_images < num_of_images // 3:
            max_num_of_images = num_of_another_images
        else:
            max_num_of_images = num_of_images // 3

        # drowsy : 0, normal : 1
        for count in range(max_num_of_images):

            image_paths = self._train_image_paths[current_state]
            image = image_paths[count * 3 + 0]
            images_path.append(image)
            image = image_paths[count * 3 + 1]
            images_path.append(image)
            image = image_paths[count * 3 + 2]
            images_path.append(image)

            # change 3A to 1B
            if current_state == 'drowsy':
                current_state = 'normal'
            else:
                current_state = 'drowsy'

            image_paths = self._train_image_paths[current_state]
            image = image_paths[count]
            images_path.append(image)

            # change 1B to 3A
            if current_state == 'drowsy':
                current_state = 'normal'
            else:
                current_state = 'drowsy'

        if self._current_train_state_index == 0:
            self._current_train_state_index = 1
        else:
            self._current_train_state_index = 0

        # check AA AB AA AB... or BB BA BB BA...
        # print("images_path: ", images_path)
        # print("images_paht size", np.shape(images_path))

        images, labels = self._convert_path_list_to_images_and_labels(images_path, is_one_shot_task=False)

        # check if label and image are correct
        # print("shape of image metrix: ", np.shape(images))
        # print("labels: ", labels[:7])
        # for i in range(7):
        #    img = np.array(images[1][i])
        #    print(img.shape)
        #    cv2.imshow('123', img)
        #    cv2.waitKey(0)
        #    cv2.destroyAllWindows()

        if self.use_agumentation:
            images = self.image_augmentor.get_random_transform(images)

        return images, labels

    def load_image(self, paths, state):

        len_of_image = len(paths)

        images = np.zeros((len_of_image, self.image_height, self.image_width, 3))
        labels = np.zeros((len_of_image,))

        for i in range(len_of_image):
            img = Image.open(paths[i])
            img = img.resize((self.image_height, self.image_width))
            img = np.asarray(img).astype(np.float64)
            img = (img - img.mean()) / img.std()
            images[i, :, :, :] = img

            if state == "drowsy": labels[i] = 1
            elif state == "normal": labels[i] = 0

        return images, labels

    def visualization_data_load(self):

        drowsy_image_paths = self.train_dictionary['drowsy']

        normal_image_paths = self.train_dictionary['normal']

        drowsy_images, drowsy_labels = self.load_image(drowsy_image_paths, "drowsy")
        normal_images, normal_labels = self.load_image(normal_image_paths, "normal")

        images = np.concatenate((drowsy_images, normal_images), axis=0)
        labels = np.concatenate((drowsy_labels, normal_labels), axis=0)

        return images, labels