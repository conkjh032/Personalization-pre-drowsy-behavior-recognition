import os
import gc
import cv2
import dlib
import time
import imutils
import datetime
import playsound
import numpy as np

from threading import Thread
from imutils import face_utils
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.spatial import distance as dist
from pynput.keyboard import Listener, Key

from siameseNet import *
from loss_functions import *

class KJH_Model:

    def __init__(self, data, label, model, optimizer, epochs, batch_size):
        self.data = data
        self.label = label
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.no_of_training_batches = len(data[0]) / batch_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def fit(self):

        print("[INFO] training started \n")

        train_loss_log = []

        for epoch in range(self.epochs):
            loss = self.run_epoch(self.data, self.label, self.model, self.optimizer, split='train')

            loss = loss / self.no_of_training_batches

            train_loss_log.append(loss)

            print('Loss after epoch ' + str(epoch + 1) + ' is:', loss)

            if (epoch + 1) % 50 == 0:
                torch.save({'model_state_dict': self.model.cpu().state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss},
                            'checkpoints/model_epoch_' + str(epoch + 1) + '.pth')

        x_axis = [x for x in range(self.epochs)]
        plt.plot(x_axis, train_loss_log)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    def run_epoch(self, data, label, model, optimizer, split='train'):

        model.to(self.device)
        gc.collect()

        if split == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0

        # 묶여진 데이터는 for문으로 해서 min_batch 마다 학습된다.
        # 데이터 형식을 tensor로 바꾸는 거 잊지말고

        pack_of_data = self.split_to_batch(data, label)

        # iterations
        for batch_id, (imgs1, imgs2, labels) in enumerate(pack_of_data):

            imgs1 = torch.from_numpy(imgs1)
            imgs2 = torch.from_numpy(imgs2)
            labels = torch.from_numpy(labels)

            imgs1 = imgs1.type(torch.FloatTensor)
            imgs2 = imgs2.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            imgs1 = imgs1.to(self.device)
            imgs2 = imgs2.to(self.device)
            labels = labels.to(self.device)
            embeddings1, embeddings2 = model.siamese_get_embeddings(imgs1, imgs2)
            batch_loss = contrastive_loss(embeddings1, embeddings2, labels, margin=1)
            optimizer.zero_grad()

            if split == 'train':
                batch_loss.backward()
                optimizer.step()

            running_loss = running_loss + batch_loss.item()

        return running_loss

    def split_to_batch(self, data, label):

        dataA = data[0]
        dataB = data[1]

        dataA = np.transpose(dataA, [0, 3, 2, 1])
        dataB = np.transpose(dataB, [0, 3, 2, 1])

        coupled_data = []
        iter_num = len(dataA) // self.batch_size
        remainer = len(dataA) % self.batch_size
        last_index = 0

        for i in range(iter_num):
            tmp_package = []
            tmp_package.append(dataA[i*self.batch_size : i*self.batch_size + self.batch_size, :, :, :])
            tmp_package.append(dataB[i*self.batch_size : i*self.batch_size + self.batch_size, :, :, :])
            tmp_package.append(label[i*self.batch_size : i*self.batch_size + self.batch_size])
            last_index = i
            coupled_data.append(tmp_package)

        if remainer != 0:
            tmp_package = []
            tmp_package.append(dataA[(last_index+1)*self.batch_size:, :, :, :])
            tmp_package.append(dataB[(last_index+1)*self.batch_size:, :, :, :])
            tmp_package.append(label[(last_index+1)*self.batch_size:])
            coupled_data.append(tmp_package)

        return coupled_data

    def behavior_detection(self, model, image, frame):

        # behavior detection function
        frame = cv2.resize(frame, (112, 112))
        frame = np.asarray(frame).astype(np.float64)
        frame = (frame - frame.mean()) / frame.std()

        frame = frame.tolist()
        frame = np.array([frame])
        frame = np.transpose(frame, [0, 3, 2, 1])

        frame = torch.from_numpy(frame)
        frame = frame.type(torch.FloatTensor)
        frame = frame.to(self.device)

        model.to(self.device)
        output1, output2 = model.siamese_get_embeddings(image, frame)
        euclidean_distance = torch.cdist(output1, output2).item()
        # euclidean_distance = F.pairwise_distance(output1, output2).item()
        print("similarity: {:.2f}".format(euclidean_distance))

        # If the distance is lower than threshold, alarm the driver to take some rest or to refresh
        if(euclidean_distance < 0.65):
            alarm = Thread(target=self.sound_alarm, args=('alarm.wav',))
            alarm.deamon = True
            alarm.start()

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

    def sound_alarm(self, path):
        playsound.playsound(path)

    def try_camera(self, mode='blink detection', checkpoint=None):

        cap = cv2.VideoCapture("data/video test/drowsy2.mp4")
        start_time = 0

        if mode == 'behavior detection':
            model = EmbeddingNetSiamese()
            model.to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.eval()

            image = self.data[0][0]
            image = image.tolist()
            image = np.array([image])
            image = np.transpose(image, [0, 3, 2, 1])

            image = torch.from_numpy(image)
            image = image.type(torch.FloatTensor)

            image = image.to(self.device)

        # define two constants, one for the eye aspect ratio to indicate
        # blink and then a second constant for the number of consecutive
        # frames the eye must be below the threshold for to set off the
        # alarm
        EYE_AR_THRESH = 0.3
        EYE_AR_CONSEC_FRAMES = 48

        # initialize the frame counter as well as a boolean used to
        # indicate if the alarm is going off
        COUNTER = 0
        BLINK_ALARM_ON = False

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        print("[INFO] starting video stream thread...")

        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale

        while(True):

            ret, frame = cap.read()

            if ret == False:
                break

            frame = imutils.resize(frame, width=500, height=500)

            if mode == 'behavior detection':
                b_frame = imutils.resize(frame, width=112, height=112)
                self.behavior_detection(model, image, b_frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)

            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                    # if the eyes were closed for a sufficient number of
                    # then sound the alarm
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:

                        start_time = time.time()

                        # if the alarm is not on, turn it on
                        if not BLINK_ALARM_ON:
                            BLINK_ALARM_ON = True

                            # check to see if an alarm file was supplied,
                            # and if so, start a thread to have the alarm
                            # sound played in the background
                            alarm = Thread(target=self.sound_alarm, args=('alarm.wav',))
                            alarm.deamon = True
                            alarm.start()

                        # draw an alarm on the frame
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                    COUNTER = 0
                    BLINK_ALARM_ON = False

                # draw the computed eye aspect ratio on the frame to help
                # with debugging and setting the correct eye aspect ratio
                # thresholds and frame counters
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # after 2 seconds drowsy driving detected, capture the behavior of driver
            # to try to wake him/her up
            end_time = time.time()
            gap_time = end_time - start_time
            if (gap_time > 2 and gap_time < 3):
                date = datetime.datetime.today()
                cv2.imwrite("data/images_train/drowsy_"
                            + str(date.year) + "_" + str(date.month) + "_"
                            + str(date.day) + "_" + str(date.second) + '.png', frame)

            # show the frame
            cv2.imshow("Driving", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # delete VideoCapture and shutdown window
        cap.release()
        cv2.destroyAllWindows()