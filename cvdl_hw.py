import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import yaml

import os

import sys, cvdl

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from PIL import Image

class MainWindow(QMainWindow, cvdl.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('cvdl_hw')
        self.setupUi(self)        
        self.resize(1000, 800)
        
        self.folder_path = None
        self.image_L = None
        self.image_R = None
        self.image_1 = None
        self.image_2 = None
        self.image = None
        self.mtx = None
        self.dist = None
        
        self.pushButton.clicked.connect(self.loadFolder)
        self.pushButton_2.clicked.connect(self.load_image_l)
        self.pushButton_3.clicked.connect(self.load_image_r)
        self.pushButton_6.clicked.connect(self.find_extrinsic)
        self.pushButton_5.clicked.connect(self.find_intrinsic)
        self.pushButton_4.clicked.connect(self.find_corners)
        self.pushButton_7.clicked.connect(self.find_distortion)
        self.pushButton_8.clicked.connect(self.show_result)
        self.pushButton_11.clicked.connect(self.show_words_on_board)
        self.pushButton_9.clicked.connect(self.show_words_vertical)
        self.pushButton_10.clicked.connect(self.stereo_diparity_map)
        self.pushButton_12.clicked.connect(self.load_image_1)
        self.pushButton_13.clicked.connect(self.load_image_2)
        self.pushButton_14.clicked.connect(self.keypoints)
        self.pushButton_15.clicked.connect(self.matched_keypoints)
        self.pushButton_18.clicked.connect(self.show_model_structure)
        self.pushButton_17.clicked.connect(self.show_augmentation_images)
        self.pushButton_19.clicked.connect(self.show_acc_n_loss)
        self.pushButton_20.clicked.connect(self.inference)
        self.pushButton_16.clicked.connect(self.load_image)     
        self.spinBox.setRange(1, 15)
        self.lineEdit.setMaxLength(6)
        
    #-------------------------#
    #        Model Net        #
    #-------------------------#       
    class VGG19(nn.Module):
        def __init__(self,num_classes):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
            self.block4 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
            self.block5  = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
            self.classifier = nn.Sequential(
                nn.Linear(512*1*1,4096),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(4096,4096),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(4096,num_classes)
            )
        def forward(self,x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            logits = self.classifier(x.view(-1,512*1*1))
            probas = F.softmax(logits,dim = 1)
            return logits,probas    
    
    #-------------------------#
    #        Load Image       #
    #-------------------------#  
    def loadFolder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.folder_path = QFileDialog.getExistingDirectory(self, "Select folder: ", options=options)
        
        if self.folder_path:
            print("Folder path: ", self.folder_path)
 
    #-------------------------#
    #       Load Image L      #
    #-------------------------#      
    def load_image_l(self):     
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        
        if file_name:
            print("File name: ", file_name)
            self.image_L = cv2.imread(file_name)
            
    #-------------------------#
    #       Load Image R      #
    #-------------------------#      
    def load_image_r(self):     
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        
        if file_name:
            print("File name: ", file_name)
            self.image_R = cv2.imread(file_name)   
            
    #-------------------------#
    #       Load Image 1      #
    #-------------------------#      
    def load_image_1(self):     
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        
        if file_name:
            print("File name: ", file_name)
            self.image_1 = cv2.imread(file_name)
            
    #-------------------------#
    #       Load Image 2      #
    #-------------------------#      
    def load_image_2(self):     
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        
        if file_name:
            print("File name: ", file_name)
            self.image_2 = cv2.imread(file_name)
              
    #-------------------------#
    #          Q 1.1          #
    #-------------------------#             
    def find_corners(self):
        image_files = [f for f in os.listdir(self.folder_path) if f.endswith('.bmp')]

        for image_file in image_files:
            image_path = os.path.join(self.folder_path, image_file)
            image = cv2.imread(image_path)

            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                width = 11
                height = 8
                
                ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
                if ret:
                    winSize = (5, 5)
                    zeroZone = (-1, -1)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)

                    cv2.drawChessboardCorners(image, (width, height), corners, ret)
                    
                    resized_image = cv2.resize(image, (1024, 1024))
                    
                    cv2.imshow('Chessboard Corners', resized_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print(f"No chessboard's corners in '{image_file}'")
            else:
                print(f"Cannot import '{image_file}', plz check the path.")
                
    #-------------------------#
    #          Q 1.2          #
    #-------------------------# 
    def find_intrinsic(self):
        self.find_corners() # Q1.1
        
        width = 11
        height = 8

        objpoints = []
        imgpoints = []

        objp = np.zeros((width * height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

        image_files = [f for f in os.listdir(self.folder_path) if f.endswith('.bmp')]

        for image_file in image_files:
            image_path = os.path.join(self.folder_path, image_file)

            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("Intrinsic matrix (camera matrix):\n", self.mtx)
        
    #-------------------------#
    #          Q 1.3          #
    #-------------------------#
    def find_extrinsic(self):
        self.find_intrinsic() # Q1.2
        
        image_number = self.spinBox.value()
        image_files = [f for f in os.listdir(self.folder_path) if f.endswith('.bmp')]
        
        if image_number <= len(image_files):
            image_path = os.path.join(self.folder_path, image_files[image_number - 1])
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            width = 11
            height = 8
            objp = np.zeros((width * height, 3), np.float32)
            objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
            ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

            if ret:
                _, rvec, tvec = cv2.solvePnP(objp, corners, self.mtx, self.dist)
    
                extrinsic_matrix = cv2.Rodrigues(rvec)[0]
                extrinsic_matrix = np.hstack((extrinsic_matrix, tvec))

                print("Extrinsic Matrix: ")
                print(extrinsic_matrix)   
                
    #-------------------------#
    #          Q 1.4          #
    #-------------------------#
    def find_distortion(self):
        self.find_extrinsic() # Q1.3
        
        width = 11
        height = 8
        objpoints = []
        imgpoints = []

        objp = np.zeros((width * height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        image_files = [f for f in os.listdir(self.folder_path) if f.endswith('.bmp')]
        for image_file in image_files:
            image_path = os.path.join(self.folder_path, image_file)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("Distortion coefficients: ", self.dist)
    
    #-------------------------#
    #          Q 1.5          #
    #-------------------------#  
    def show_result(self):
        self.find_distortion() # Q1.4
        
        image_number = self.spinBox.value()
        
        image_files = [f for f in os.listdir(self.folder_path) if f.endswith('.bmp')]
        
        if 1 <= image_number <= 15:
            image_path = os.path.join(self.folder_path, image_files[image_number - 1])
            image = cv2.imread(image_path)
            
            if image is not None:
                undistorted_image = cv2.undistort(image, self.mtx, self.dist)

                undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
                
                image = cv2.resize(image, (700, 700))
                undistorted_image = cv2.resize(undistorted_image, (700, 700))

                cv2.imshow('Distorted Chessboard', image)
                cv2.imshow('Undistorted Chessboard', undistorted_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
    #-------------------------#
    #          Q 2.1          #
    #-------------------------# 
    def draw(self, img, imgpts):
        img = np.copy(img)
        print(imgpts)
        for i in range(0, len(imgpts), 2):
            img = cv2.line(img, pt1=(int(imgpts[i][0][0]), int(imgpts[i][0][1])), pt2=(int(imgpts[i+1][0][0]), int(imgpts[i+1][0][1])), thickness=10, color=(255, 0, 0))
        return img
    
    def show_words_on_board(self):                
        width = 11
        height = 8
        objpoints = []
        imgpoints = []

        objp = np.zeros((width * height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        
        image_files = [f for f in os.listdir(self.folder_path) if f.endswith('.bmp')]
        for image_file in image_files:
            image_path = os.path.join(self.folder_path, image_file)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        library = r'Dataset_CvDl_Hw1\Q2_Image\Q2_lib\alphabet_lib_onboard.yaml'

        with open(library, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        init_pos_offset = [[7, 5], [4, 5], [1, 5], [7, 2], [4, 2], [1, 2]]
        img_files = os.listdir(self.folder_path)[:-1]

        alpha_dict = {}
        for key in data.keys():
            vals = data[key]
            pos_infos = vals['data']
            alpha_dict[key] = np.array(pos_infos, dtype=np.float32).reshape(-1, 3)

        word = self.lineEdit.text()
        
        points = []
        for index, w in enumerate(word):
            alpha = np.copy(alpha_dict[w])
            for index_pos in range(len(alpha)):
                alpha[index_pos][0] += init_pos_offset[index][0]
                alpha[index_pos][1] += init_pos_offset[index][1]
                points.append(alpha[index_pos])
        
        points = np.array(points, dtype=np.float32)

        for img_index, img_file in enumerate(img_files):
            img = cv2.imread(os.path.join(self.folder_path, img_file))

            for index, w in enumerate(word):        
                imgpts, jac = cv2.projectPoints(points, rvecs[img_index], tvecs[img_index], mtx, dist)
                imgpts = np.array(imgpts)
                img = self.draw(img, imgpts)

            plt.imshow(img)
            plt.draw()
            plt.axis('off')
            plt.pause(1)

    #-------------------------#
    #          Q 2.2          #
    #-------------------------#
    def show_words_vertical(self):                
        width = 11
        height = 8
        objpoints = []
        imgpoints = []

        objp = np.zeros((width * height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        
        image_files = [f for f in os.listdir(self.folder_path) if f.endswith('.bmp')]
        for image_file in image_files:
            image_path = os.path.join(self.folder_path, image_file)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        library = r'Dataset_CvDl_Hw1\Q2_Image\Q2_lib\alphabet_lib_vertical.yaml'

        with open(library, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        init_pos_offset = [[7, 5], [4, 5], [1, 5], [7, 2], [4, 2], [1, 2]]
        img_files = os.listdir(self.folder_path)[:-1]

        alpha_dict = {}
        for key in data.keys():
            vals = data[key]
            pos_infos = vals['data']
            alpha_dict[key] = np.array(pos_infos, dtype=np.float32).reshape(-1, 3)

        word = self.lineEdit.text()
        
        points = []
        for index, w in enumerate(word):
            alpha = np.copy(alpha_dict[w])
            for index_pos in range(len(alpha)):
                alpha[index_pos][0] += init_pos_offset[index][0]
                alpha[index_pos][1] += init_pos_offset[index][1]
                points.append(alpha[index_pos])
        
        points = np.array(points, dtype=np.float32)

        for img_index, img_file in enumerate(img_files):
            img = cv2.imread(os.path.join(self.folder_path, img_file))

            for index, w in enumerate(word):        
                imgpts, jac = cv2.projectPoints(points, rvecs[img_index], tvecs[img_index], mtx, dist)
                imgpts = np.array(imgpts)
                img = self.draw(img, imgpts)

            plt.imshow(img)
            plt.draw()
            plt.axis('off')
            plt.pause(1)

    #-------------------------#
    #           Q 3           #
    #-------------------------# 
    def stereo_diparity_map(self):        
        num_disparities = 256
        block_size = 25        
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

        left_gray = cv2.cvtColor(self.image_L, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.image_R, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(left_gray, right_gray)
        disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        left_image_with_dots = self.image_L.copy()
        right_image_with_dots = self.image_R.copy()

        def draw_dot(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                disparity_value = disparity_norm[y, x]    
                print(disparity_value)         
                if disparity_value != 0:
                    cv2.circle(left_image_with_dots, (x, y), 5, (255, 0, 255), -1)
                    x_offset = x - disparity_value
                    cv2.circle(right_image_with_dots, (x_offset, y), 5, (0, 0, 255), -1)

        cv2.namedWindow('Image L', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image L', 720, 480)
        cv2.imshow('Image L', left_image_with_dots)
        cv2.setMouseCallback('Image L', draw_dot)
        
        cv2.namedWindow('Image R', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image R', 720, 480)
        cv2.imshow('Image R', right_image_with_dots)
        
        cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Disparity', 720, 480)
        cv2.imshow("Disparity", disparity_norm)

        while True:
            cv2.namedWindow('Image L', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image L', 720, 480)
            cv2.imshow('Image L', left_image_with_dots)
            
            cv2.namedWindow('Image R', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image R', 720, 480)
            cv2.imshow('Image R', right_image_with_dots)
            
            cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Disparity', 720, 480)
            cv2.imshow("Disparity", disparity_norm)
            
            key = cv2.waitKey(1)
            if key == 27 & 0xFF:
                break
        
        cv2.destroyAllWindows()
        
    #-------------------------#
    #          Q 4.1          #
    #-------------------------#
    def keypoints(self):
        gray = cv2.cvtColor(self.image_1, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, _ = sift.detectAndCompute(gray, None)
        image_with_keypoints = cv2.drawKeypoints(self.image_1, kp, None, color=(0, 255, 0))

        cv2.namedWindow('Image with SIFT Keypoints', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image with SIFT Keypoints', 900, 900)
        cv2.imshow('Image with SIFT Keypoints', image_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    #-------------------------#
    #          Q 4.2          #
    #-------------------------#
    def matched_keypoints(self):
        gray_1 = cv2.cvtColor(self.image_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(self.image_2, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(gray_1, None)
        kp2, des2 = sift.detectAndCompute(gray_2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        matched_image = cv2.drawMatchesKnn(gray_1, kp1, gray_2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.namedWindow('Matched Keypoints', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Matched Keypoints', 1200, 600)
        cv2.imshow('Matched Keypoints', matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    #-------------------------#
    #          Q 5.1          #
    #-------------------------#
    def show_augmentation_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.folder_path = QFileDialog.getExistingDirectory(self, "Select folder: ", options=options)
        
        if self.folder_path:
            print("Folder path: ", self.folder_path)
            
        image_paths = [os.path.join(self.folder_path, filename) for filename in os.listdir(self.folder_path) if filename.endswith(".png")]
        image_labels = [os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths]
        loaded_images = [Image.open(image_path) for image_path in image_paths]

        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
            transforms.RandomCrop(size=(22, 22)), 
            transforms.RandomAdjustSharpness(sharpness_factor=2)
        ])
        
        augmented_images = [data_transforms(image) for image in loaded_images]

        plt.figure("Augmented Images", figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[i])
            plt.title(image_labels[i])

        plt.tight_layout()
        plt.show()                
                
    #-------------------------#
    #          Q 5.2          #
    #-------------------------#                
    def show_model_structure(self):
        vgg19 = self.VGG19(10)
        vgg19.to('cuda')
        summary(vgg19, (3, 32, 32))        
    
    #-------------------------#
    #          Q 5.3          #
    #-------------------------#
    def show_acc_n_loss(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select photos: ", options=options, filter="Images (*.png *.jpg *.jpeg *.bmp)")
        if file_paths:
            images = [cv2.imread(path) for path in file_paths]
            if len(images) >= 2:
                cv2.imshow("Acc", images[0])
                cv2.imshow("Loss", images[1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                   
    #-------------------------#
    #        Load Image       #
    #-------------------------#        
    def convert_cvimage_to_pixmap(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(128, 128, aspectRatioMode=Qt.KeepAspectRatio)

        return scaled_pixmap
      
    def load_image(self):     
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        
        if file_name:
            print("File name: ", file_name)
            self.image = cv2.imread(file_name)

            pixmap = self.convert_cvimage_to_pixmap(self.image)

            scene = QGraphicsScene(self)
            item = QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.InferenceImage.setScene(scene)
                
    #-------------------------#
    #          Q 5.4          #
    #-------------------------#       
    def inference(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pth);;All Files (*)", options=options)
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        if model_path:
            if os.path.exists(model_path):
                print("Loading model from:", model_path)

            model = self.VGG19(10)
            model.load_state_dict(torch.load(model_path))

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            model.eval()
            with torch.no_grad():
                logits, probas = model(transform(self.image).reshape(1, 3, 32, 32))
                print("Predicted class:", classes[np.argmax(logits.detach().numpy())])
                pred = classes[np.argmax(logits.detach().numpy())]
                distribution = probas.detach().numpy()

                self.label.setText(f'predict = {pred}')
                x = np.arange(10)

                plt.bar(x, distribution.reshape(10, ))
                plt.xticks(x, classes)
                plt.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())