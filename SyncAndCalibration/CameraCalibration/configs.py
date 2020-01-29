import numpy as np

class Configs:
    large_font = ("Verdana", 12)
    image_shape = (2160, 4000)
    cam_sensor_shape = (22, 11.88)  # w, h [mm]
    num_cams = 0

    def __init__(self):
        self.num_stereo_imgs = 0
        self.num_single_calib_imgs = 0
        self.center_cam_idx = 0
        self.center_img_name = '00000'
        self.frame_range = (0, 0)
        # chb
        s = self.Chb.size
        for r in range(self.Chb.row):
            for c in range(self.Chb.col):
                i = r*self.Chb.col + c
                self.Chb.obj_points[i, 0] = -c * s
                self.Chb.obj_points[i, 1] = r * s

    # checkerboard
    class Chb:
        row = 8
        col = 11
        num_corners = 88
        size = 60
        obj_points = np.zeros((row * col, 3), np.float32)

