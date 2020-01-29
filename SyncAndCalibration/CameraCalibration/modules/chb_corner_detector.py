import numpy as np
import threading
import os
import cv2
import glob
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

class CornerDetector:
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self, parent, idx, list, chb_param, calib_configs, is_gui=False):
        self.is_gui = is_gui

        self.thread = None
        self.run_thread = False
        self.parent = parent
        self.index = idx
        self.path_list = list
        self.chb_dim = chb_param['chb_dim']
        self.calib_configs = calib_configs
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.chb_dim[0] * self.chb_dim[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chb_dim[0], 0:self.chb_dim[1]].T.reshape(-1, 2)

        self.num_success = 0
        self.num_fail = 0


    def _load_image(self, index, image_path, scale):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # scale
        h, w = gray.shape
        h = float(h) * float(scale)
        w = float(w) * float(scale)
        gray_smaller = cv2.resize(gray, (int(w), int(h)))
        return gray, gray_smaller

    def _detect_corners(self):
        if self.is_gui:
            self.parent.print_to_text_widget(self.index, 'Camera [{}]\n'.format(self.index + 1))

        # make output folder if not exists
        if self.is_gui:
            vs = self.path_list[0].split('/')
        else:
            vs = self.path_list[0].split('\\')
        now = datetime.now()
        folder_name = now.strftime("%Y%m%d")

        output_path = '\\'.join(vs[:-1]) + '\\Corners_' + folder_name

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # load image and detect
        num_imgs = len(self.path_list)
        for i, image_path in enumerate(self.path_list):
            if self.run_thread is False:
                break

            if self.is_gui:
                img_name_str = image_path.split('/')[-1]
            else:
                img_name_str = image_path.split('\\')[-1]
            img_name_int = int(img_name_str.split('.')[0].split('_')[-1])

            if img_name_int % 5 is not 0:
                print('(skipping: {})'.format(img_name_str))
                continue

            # verbose
            if self.is_gui:
                path_piece = image_path.split('/')
            else:
                path_piece = image_path.split('\\')
            image_name = path_piece[-1].split('.')[0]
            now = datetime.now()
            time = now.strftime("%H:%M:%S")
            if self.is_gui:
                self.parent.print_to_text_widget(self.index, '({}/{}) [{}]\n'.format(i + 1, num_imgs, time))
            else:
                print('({}/{}) [{}]'.format(i + 1, num_imgs, time))

            # load an image
            if 'scale' in self.calib_configs:
                scale = self.calib_configs['scale']
            else:
                print('[ERROR] Scale not set. Setting it to 1.0')
                scale = 1.0

            image, image_smaller = self._load_image(i, image_path, scale)
            if self.is_gui:
                self.parent.print_to_text_widget(self.index, '  - Image loaded: {}\n'.format(image_path))
            else:
                print('  - Image loaded: {}'.format(image_path))


            # find corners
            if self.is_gui:
                self.parent.print_to_text_widget(self.index, '  - Detecting corners..')
            else:
                print('  - Detecting corners.. ', end='')

            ret, corners = cv2.findChessboardCorners(image_smaller, self.chb_dim, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

            output_txt_path = ''
            str_ret = ''
            output_str_list = []
            # If found, add object points, image points (after refining them)
            if ret:
                ret, corners = cv2.findChessboardCorners(image, self.chb_dim, cv2.CALIB_CB_ADAPTIVE_THRESH)
                scale_used = 1.0

                if not ret:
                    ret, corners = cv2.findChessboardCorners(image_smaller, self.chb_dim, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
                    scale_used = scale

                if ret:
                    str_ret = 'SUCCESS: scale={}'.format(scale_used)
                    corners /= scale_used
                    self.num_success += 1

                    corners2 = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), self.criteria)
                    output_str_list.append('True')

                    # corner object to image points string
                    for uv in corners2:
                        uv_str = '\n' + str(uv[0][0]) + ' ' + str(uv[0][1])
                        output_str_list.append(uv_str)
            else:
                str_ret = 'FAIL'
                self.num_fail += 1
                output_str_list.append('False')

            # export output to .txt file
            output_txt_path = output_path + '\\' + image_name + '.txt'
            with open(output_txt_path, 'w') as output_txt:
                output_txt.writelines(output_str_list)
                output_txt.close()

            str_ret += ' | saved as: ' + image_name + '.txt'
            # display status to text_widget
            if self.is_gui:
                self.parent.print_to_text_widget(self.index, '  ' + str_ret + '\n')
            else:
                print('| {}'.format(str_ret))

            # clean up
            del image

        if self.run_thread:
            self.run_thread = False

            # display status to text_widget
            if self.is_gui:
                self.parent.print_to_text_widget(self.index, '\n* Complete! Output saved to:\n  {}\n'.format(output_path))
            else:
                print()
                print('* Complete! Outputs saved to:\n  {}\n'.format(output_path))
                input("Press enter to quit.")

    def stop_work(self):
        print('thread[{}] stop work'.format(self.index))
        self.run_thread = False
        self.thread.join()


    def run_corner_detection_thread(self):
        self.thread = threading.Thread(target=self._detect_corners)
        self.run_thread = True
        self.thread.start()