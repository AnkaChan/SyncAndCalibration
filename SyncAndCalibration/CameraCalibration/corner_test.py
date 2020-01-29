from modules.parser import *
import matplotlib.pyplot as plt
import os
from PIL import Image
import datetime
from configs import *
import shutil
import matplotlib
from modules.chb_corner_detector import *
import math




def __ChESS(image, mask):
    I = np.array(image, copy=True).astype('float32')
    gb = np.zeros((I.shape[0], I.shape[1], 2))
    image = np.dstack((I, gb))

    score = 0
    sample_indices = range(int(0.25 * len(mask)))
    half = int(len(sample_indices) * 2)
    quarter = int(len(sample_indices))
    for idx0 in sample_indices:
        idx1 = idx0 + half
        ortho_idx0 = idx0 + quarter
        ortho_idx1 = ortho_idx0 + half

        image[mask[idx0]] = [0, 0, 255]
        image[mask[idx1]] = [0, 255, 0]
        image[mask[ortho_idx0]] = [0, 255, 255]
        image[mask[ortho_idx1]] = [255, 255, 0]

        # sum response (SR)
        score += np.abs((I[mask[idx0]] + I[mask[idx1]]) - (I[mask[ortho_idx0]] + I[mask[ortho_idx1]]))

        # diff response (DR)
        score -= np.abs(I[mask[idx0]] - I[mask[idx1]]) + np.abs(I[mask[ortho_idx0]] - I[mask[ortho_idx1]])

    #     score /= 255.0
    return image, score


def __NotChESS(image):
    hh, ww = image.shape
    hh = int(hh / 2.0)
    ww = int(ww / 2.0)

    imgTL = image[0:hh, 0:ww]
    imgTR = image[0:hh, ww:2 * ww]
    imgBR = np.flip(image[hh:2 * hh, ww:2 * ww])
    imgBL = np.flip(image[hh:2 * hh, 0:ww])

    score = (np.sum((imgTL - imgBR)**2) + np.sum((imgTR - imgBL)**2)) / 255.0
    return score


def corner_test():

    root_path = r'D:\Pictures\2019_08_09_AllPoseCapture\Converted\_CornerCrop'
    output_path = root_path + '\\Scored'
    cams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    mask = np.zeros((16, 16)).astype('uint8')
    cx = 8.0
    cy = 8.0
    r = int(mask.shape[0] * 0.25) - 0.5
    mask_vu = []
    mask_type = 'circle'
    if mask_type == 'circle':
        for theta in np.linspace(1, int(np.pi * 2 + 1), 360):
            v = int(cy + r * math.sin(theta))
            u = int(cx + r * math.cos(theta))
            if mask[v, u] == 0:
                mask[v, u] = 1
                mask_vu.append((v, u))
    elif mask_type == 'square':
        sy = int(cy - r)
        sx = int(cx - r)
        ey = int(cy + r) + 1
        ex = int(cx + r) + 1
        for v in range(sy + 1, ey):
            if mask[v, ex] == 0:
                mask[v, ex] = 1
                mask_vu.append((v, ex))

        for u in range(ex - 1, sx, -1):
            if mask[ey, u] == 0:
                mask[ey, u] = 1
                mask_vu.append((ey, u))

        for v in range(ey - 1, sy, -1):
            if mask[v, sx] == 0:
                mask[v, sx] = 1
                mask_vu.append((v, sx))
        for u in range(sx + 1, ex):
            if mask[sy, u] == 0:
                mask[sy, u] = 1
                mask_vu.append((sy, u))

    dpi = 300
    sz = 10
    x = np.linspace(0, 15, 17)
    y = np.linspace(0, 15, 17)
    for cam_idx in range(8):
        print('Camera ', str(cam_idx))

        path = root_path + '\\' + str(cam_idx)
        folders = sorted(next(os.walk(path))[1])

        for folder in folders:
            p = path + '\\' + folder
            corners_list = glob.glob(p + '\\*.jpg')

            for p_idx, corner in enumerate(corners_list):
                corner_img = cv2.imread(corner)
                gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)

                # ChESS
                _, score1 = __ChESS(blur, mask_vu)

                if score1 < 300:
                    # non-chess
                    score2 = __NotChESS(blur)

                    img_title = '{}_{}_{}_{:.4f}_{:.4f}'.format(cam_idx, folder, p_idx, score1, score2)

                    fig = plt.figure()
                    fig.set_size_inches(16 * 20 / dpi, 16 * 20 / dpi, forward=False)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(corner_img, cmap='gray')

                    plt.scatter(7.5, 7.5, s=sz, c='r')
                    plt.scatter([7.5] * len(y), y, s=sz * 0.2, c='r')
                    plt.scatter(x, [7.5] * len(x), s=sz * 0.2, c='r')

                    plt.savefig(output_path + '\\' + img_title + '.jpg', dpi=dpi)
                    plt.close()


if __name__ == '__main__':
    corner_test()