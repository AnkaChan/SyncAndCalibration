import tkinter as tk
from frames.frames import *


class MainApp(tk.Tk):
    win_w = 800
    win_h = 600
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(StartFrame)
        self.geometry(str(self.win_w) + 'x' + str(self.win_h))
        self.winfo_toplevel().title("Camera Calibration (2019.11)")

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()
        return new_frame

def main_GUI():
    app = MainApp()
    app.mainloop()
    print('done')

def main_CONSOLE():
    root_path = input("Image directory: ")
    img_scale = input("Image scale [0, 1]: ")

    file_list_raw = glob.glob(root_path + '\\*.pgm')
    i0 = input("Start frame [0, {}]: ".format(len(file_list_raw)))
    i1 = input("End frame [0, {}]: ".format(len(file_list_raw)))
    i0 = int(i0)
    i1 = int(i1)
    print()
    chb_param = {'chb_dim': (11, 8), 'sqr_size': 60}
    calib_configs = {'scale': img_scale}

    file_list = []
    for file in file_list_raw[i0:i1 + 1]:
        img_name_str = file.split('\\')[-1]
        img_name_int = int(img_name_str.split('.')[0].split('_')[-1])
        if img_name_int % 5 == 0:
            file_list.append(file)

    file_list = sorted(file_list)

    corner_detector = CornerDetector(None, 0, file_list, chb_param, calib_configs, False)

    # load image and run corner detection
    corner_detector.run_corner_detection_thread()


if __name__ == '__main__':
    # main_GUI()
    main_CONSOLE()