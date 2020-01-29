import vtk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import glob
import k3d
import matplotlib as mpl
from cycler import cycler
from matplotlib.gridspec import GridSpec
import itertools
from operator import itemgetter
import time

from mpl_toolkits.mplot3d import Axes3D
import json


class Analyzer:
    cams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    @classmethod
    def load_bundle_adj_output(cls, path, num_cams):
        cam_parameters = {}
        frames = {}
        g_chb_corners = None
        with open(path, 'r') as f:
            lines = f.readlines()
            num_frames = int(lines[0].split()[0])
            line_start = num_frames + 2

            cam_line = lines[line_start]
            cam_params = cam_line.split()
            for cam_idx in range(num_cams):
                param = cam_params[cam_idx * 22:(cam_idx + 1)*22 + 1]
                ax = np.array([param[0], param[1], param[2]]).astype(np.float32)
                t = np.array([param[3], param[4], param[5]]).astype(np.float32)
                fx = float(param[6])
                fy = float(param[7])
                cx = float(param[8])
                cy = float(param[9])
                k = np.array([param[10], param[11], param[12], param[13], param[14], param[15]]).astype(np.float32)
                p = np.array([param[16], param[17], param[18], param[19], param[20], param[21]]).astype(np.float32)
                out = {}
                out['ax'] = ax
                out['t'] = t
                out['fx'] = fx
                out['fy'] = fy
                out['cx'] = cx
                out['cy'] = cy
                out['k1'] = k[0]
                out['k2'] = k[1]
                out['k3'] = k[2]
                out['k4'] = k[3]
                out['k5'] = k[4]
                out['k6'] = k[5]

                out['p1'] = p[0]
                out['p2'] = p[1]
                out['p3'] = p[2]
                out['p4'] = p[3]
                out['p5'] = p[4]
                out['p6'] = p[5]
                cam_parameters[cam_idx] = out

            wp_start = line_start + 1
            line_indices = range(wp_start, wp_start + num_frames)

            chb_lines_exist = len(lines[wp_start + num_frames].split()) > 2
            if chb_lines_exist:
                print(" - 6-DoF checkerboard orientations exist")
            for line_idx in line_indices:
                wp_line = lines[line_idx]

                v = wp_line.split()
                frame_idx = int(v[0])
                img_name = v[1]
                detections = np.zeros((num_cams, 1))
                for cam_idx in range(num_cams):
                    detections[cam_idx, :] = int(v[2 + cam_idx])
                wp = np.array(v[18:]).astype(np.float).reshape((88, 3))
                frame = {}
                frame['frame_idx'] = frame_idx
                frame['img_name'] = img_name
                frame['detections'] = detections
                frame['chb_corners'] = wp

                if chb_lines_exist:
                    chb_line = lines[line_idx + num_frames]
                    c = chb_line.split()
                    chb_rvec = np.array([float(c[18]), float(c[19]), float(c[20])])
                    chb_tvec = np.array([float(c[21]), float(c[22]), float(c[23])])
                    frame['chb_rvec'] = chb_rvec
                    frame['chb_tvec'] = chb_tvec
                frames[int(v[0])] = frame

            chb_points_start = wp_start + num_frames*2
            chb_points_exist = False
            chb_points_exist_1 = len(lines) > chb_points_start
            if chb_points_exist_1:
                chb_points_exist_2 = len(lines[chb_points_start].split()) > 2
                chb_points_exist = chb_points_exist_2

            if chb_points_exist:
                chb = lines[chb_points_start].split()
                g_chb_corners = np.array(chb).reshape((88, 3)).astype(np.float)
                print(' - Global checkerboard points exist : {}'.format(g_chb_corners.shape))
            final_cost = float(lines[-1].split()[-1])
            f.close()
        return cam_parameters, frames, g_chb_corners, final_cost

    @classmethod
    def reproject_bundle_output(cls, cam_params, frames, configs, g_chb_corners=None):
        if g_chb_corners is not None:
            defaut_corners = g_chb_corners

        else:
            col = 11
            row = 8
            S = 60
            x = np.linspace(0, -(col - 1) * S, col)
            y = np.linspace(0, (row - 1) * S, row)
            X, Y = np.meshgrid(x, y)
            X = X.ravel()
            Y = Y.ravel()
            defaut_corners = np.column_stack((X, Y, np.zeros(len(X))))

        # one for each frame
        all_image_points = {}
        curr_idx = 0
        for k, v in frames.items():
            if curr_idx % 100 == 0:
                print('  {}/{}'.format(curr_idx, len(frames.keys())), end='')

            image_name = v['img_name']
            if 'chb_rvec' in v:
                ax = v['chb_rvec']
                R, _ = cv2.Rodrigues(ax)
                t = v['chb_tvec'].reshape(3, )

                chb_3d_corners = np.zeros((88, 3))
                for i in range(defaut_corners.shape[0]):
                    p = np.array(defaut_corners[i, :], copy=True)
                    chb_3d_corners[i, :] = R.dot(p) + t
            else:
                chb_3d_corners = v['chb_corners']

            # reproject
            image_points = {}
            detections = v['detections']
            for cam_idx, detected in enumerate(detections):
                if detected:
                    img_points = cls.reproject_world_points(cam_params[cam_idx], chb_3d_corners, configs)
                    image_points[cam_idx] = img_points
            all_image_points[image_name] = image_points

            curr_idx += 1
        return all_image_points

    @classmethod
    def reproject_world_points(cls, param, world_points, configs):
        max_k = configs['max_k']
        max_p = configs['max_p']
        radial_model = configs['radial_model']
        # world_points = np.array((88, 3))
        # max_k, max_p = [2, 6]
        image_points = []
        k = [param['k1'], param['k2'], param['k3'], param['k4'], param['k5'], param['k6']]
        p = [param['p1'], param['p2'], param['p3'], param['p4'], param['p5'], param['p6']]

        fx = param['fx']
        fy = param['fy']
        cx = param['cx']
        cy = param['cy']

        ax = param['ax']
        R, _ = cv2.Rodrigues(ax)
        t = param['t']
        E = np.array(
            [[R[0, 0], R[0, 1], R[0, 2], t[0]], [R[1, 0], R[1, 1], R[1, 2], t[1]], [R[2, 0], R[2, 1], R[2, 2], t[2]],
             [0, 0, 0, 1]])

        img_points = np.zeros((world_points.shape[0], 2))
        for p_idx in range(world_points.shape[0]):
            wp = world_points[p_idx, :]

            cp = E.dot(np.array([wp[0], wp[1], wp[2], 1]).astype('double'))
            xp = cp[0] / cp[2]
            yp = cp[1] / cp[2]

            if radial_model == 0:
                r2 = xp * xp + yp * yp
                r2_radials = 1.0
                radial_dist = 1.0
                for ki in range(0, max_k):
                    r2_radials *= r2
                    radial_dist += k[ki] * r2_radials
            else:
                r2 = xp ** 2 + yp ** 2
                r4 = r2 ** 2
                r6 = r2 ** 3
                radial_dist = (1.0 + k[0] * r2 + k[1] * r4 + k[2] * r6) / (1.0 + k[3] * r2 + k[4] * r4 + k[5] * r6)

            tan_post = 1.0
            r2_tangentials = 1.0
            for pi in range(2, max_p):
                r2_tangentials *= r2
                tan_post += p[pi] * r2_tangentials

            tan_x = (p[1] * (r2 + 2.0 * xp * xp) + 2.0 * p[0] * xp * yp) * tan_post
            tan_y = (p[0] * (r2 + 2.0 * yp * yp) + 2.0 * p[1] * xp * yp) * tan_post

            xp = xp * radial_dist + tan_x
            yp = yp * radial_dist + tan_y

            x_pred = fx * xp + cx
            y_pred = fy * yp + cy

            img_points[p_idx, 0] = x_pred
            img_points[p_idx, 1] = y_pred
        return img_points

    @classmethod
    def load_bundle_adj_input(cls, path):
        # key = image_name, value = dictionary with key=cam_idx, val=2d points list
        output = {}
        with open(path, 'r') as f:
            j = json.load(f)
            configs = j['configs']
            num_frames = configs['num_frames']
            frames = j['frames']
            if len(frames) != num_frames:
                print('[ERROR] num frames sanity check fail: {} != {}'.format(len(frames), num_frames))

            for frame in frames:
                out = {}
                img_name = frame['img_name']
                img_pts = frame['img_pts']
                for cam_idx, v in img_pts.items():
                    wp = np.array(v).reshape((88, 2))
                    out[int(cam_idx)] = wp
                output[img_name] = out

            f.close()
        return output

    @classmethod
    def compute_reprojection_errors(cls, preds_in, opencvs_in, configs, mins=None, maxs=None):
        img_names_2_skip = {}
        if mins is not None:
            min_u = mins[0]
            min_v = mins[1]
            max_u = maxs[0]
            max_v = maxs[1]
            for img_name, v in preds_in.items():
                img_names_2_skip[img_name] = False
                opencvs = opencvs_in[img_name]
                for cam_idx in v.keys():
                    mea = opencvs[cam_idx]

                    for p_idx in range(mea.shape[0]):
                        mask = (min_u[cam_idx] < mea[p_idx, 0]) and (mea[p_idx, 0] < max_u[cam_idx]) and (
                                    min_v[cam_idx] < mea[p_idx, 1]) and (mea[p_idx, 1] < max_v[cam_idx])
                        if not mask:
                            img_names_2_skip[img_name] = True
                            break
                    if img_names_2_skip[img_name]:
                        print('skip: {}'.format(img_name))
                        break

        all_errs = []
        err_data = {}
        sanity_err = 0
        for img_name, v in preds_in.items():
            if img_name in img_names_2_skip:
                if img_names_2_skip[img_name]:
                    print(' - skip: {}'.format(img_name))
                    continue

            opencvs = opencvs_in[img_name]
            curr_errs = {}
            for cam_idx in v.keys():
                pred = v[cam_idx]
                mea = opencvs[cam_idx]

                dxdy = pred - mea
                dxdx_dydy_sum = np.sum(dxdy ** 2, axis=1)
                if configs['loss_type'] == 'huber':
                    delta = configs['loss_huber_delta']
                    for ei in range(dxdx_dydy_sum.shape[0]):
                        if dxdx_dydy_sum[ei] > delta:
                            sanity_err += 2 * delta * np.sqrt(dxdx_dydy_sum[ei]) - 1 * delta ** 2
                        else:
                            sanity_err += dxdx_dydy_sum[ei]
                else:
                    sanity_err += np.sum(dxdx_dydy_sum)

                all_errs.extend(np.sqrt(dxdx_dydy_sum).tolist())
                curr_errs[cam_idx] = np.sqrt(dxdx_dydy_sum)
            err_data[img_name] = curr_errs

        sanity_err *= 0.5
        return all_errs, err_data, sanity_err

    @classmethod
    def render_projections(cls, img_name_str, scale, pred, mea, reproj_errs, img_path, mins=None, maxs=None):
        # image name format: A00000.pgm
        """
        :param pred: 8,88,2
        :param mea:  8,88,2
        """
        if mins is not None and maxs is not None:
            min_u = mins[0]
            min_v = mins[1]
            max_u = maxs[0]
            max_v = maxs[1]

        num_detected = 0
        max_err = 0
        min_err = np.inf
        sum_err = 0
        num_err = 0
        for cam_idx in range(16):
            if cam_idx in pred:
                num_detected += 1
                max_err = max(max_err, np.max(reproj_errs[cam_idx]))
                min_err = min(min_err, np.min(reproj_errs[cam_idx]))
                sum_err += np.sum(reproj_errs[cam_idx])
                num_err += reproj_errs[cam_idx].shape[0]
        mean_err = sum_err / num_err

        fig = plt.figure(figsize=(12, num_detected * 8))
        gs = GridSpec(num_detected, 1, figure=fig)
        idx = 0
        sz = 2
        for cam_idx in range(16):
            if cam_idx in pred:
                ax = fig.add_subplot(gs[idx, 0])

                path = img_path + cls.cams[cam_idx] + '\\' + cls.cams[cam_idx] + img_name_str + '.pgm'
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print('[ERROR] No image at:', path)
                ax.imshow(img, cmap='gray')

                # mea
                m = mea[cam_idx]
                ax.scatter(m[:, 0], m[:, 1], s=sz * 2.0, color='r')

                # pred
                p = pred[cam_idx]
                cmap = matplotlib.cm.get_cmap('plasma')
                derr = max_err - min_err
                colors = [cmap(err / derr) for err in reproj_errs[cam_idx]]
                ax.scatter(p[:, 0], p[:, 1], s=sz, vmin=min_err, vmax=max_err, color=colors)

                if mins is not None:
                    v1 = [min_u[cam_idx], min_v[cam_idx]]
                    v2 = [max_u[cam_idx], min_v[cam_idx]]
                    v3 = [max_u[cam_idx], max_v[cam_idx]]
                    v4 = [min_u[cam_idx], max_v[cam_idx]]
                    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color='g')
                    ax.plot([v2[0], v3[0]], [v2[1], v3[1]], color='g')
                    ax.plot([v3[0], v4[0]], [v3[1], v4[1]], color='g')
                    ax.plot([v4[0], v1[0]], [v4[1], v1[1]], color='g')

                ax.set_title('{}.pgm | Cam[{}] | max. err={:.2f} | point: {}'.format(img_name_str, cam_idx, np.max(reproj_errs[cam_idx]),
                                                                            np.argmax(reproj_errs[cam_idx])), fontsize=24)

                ax.set_xticks([]), ax.set_yticks([])
                ax.legend(['OpenCv', 'Predictions'])
                idx += 1
        plt.suptitle('{}.pgm | mean={:.2f} | max={:.2f}'.format(img_name_str, mean_err, max_err), fontsize=32)

    @classmethod
    def render_projections_old_naming_convention(cls, img_name_str, scale, pred, mea, reproj_errs, img_path, mins=None, maxs=None):
        # image name format: 00000.pgm
        """
        :param pred: 8,88,2
        :param mea:  8,88,2
        """
        if mins is not None and maxs is not None:
            min_u = mins[0]
            min_v = mins[1]
            max_u = maxs[0]
            max_v = maxs[1]

        num_detected = 0
        max_err = 0
        min_err = np.inf
        sum_err = 0
        num_err = 0
        for cam_idx in range(16):
            if cam_idx in pred:
                num_detected += 1
                max_err = max(max_err, np.max(reproj_errs[cam_idx]))
                min_err = min(min_err, np.min(reproj_errs[cam_idx]))
                sum_err += np.sum(reproj_errs[cam_idx])
                num_err += reproj_errs[cam_idx].shape[0]
        mean_err = sum_err / num_err

        fig = plt.figure(figsize=(12, num_detected * 8))
        gs = GridSpec(num_detected, 1, figure=fig)
        idx = 0
        sz = 2
        for cam_idx in range(16):
            if cam_idx in pred:
                ax = fig.add_subplot(gs[idx, 0])

                img = cv2.imread(img_path + cls.cams[cam_idx] + '\\' + img_name_str + '.pgm', cv2.IMREAD_GRAYSCALE)
                ax.imshow(img, cmap='gray')

                # mea
                m = mea[cam_idx]
                ax.scatter(m[:, 0], m[:, 1], s=sz * 2.0, color='r')

                # pred
                p = pred[cam_idx]
                cmap = matplotlib.cm.get_cmap('plasma')
                derr = max_err - min_err
                colors = [cmap(err / derr) for err in reproj_errs[cam_idx]]
                ax.scatter(p[:, 0], p[:, 1], s=sz, vmin=min_err, vmax=max_err, color=colors)

                if mins is not None:
                    v1 = [min_u[cam_idx], min_v[cam_idx]]
                    v2 = [max_u[cam_idx], min_v[cam_idx]]
                    v3 = [max_u[cam_idx], max_v[cam_idx]]
                    v4 = [min_u[cam_idx], max_v[cam_idx]]
                    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color='g')
                    ax.plot([v2[0], v3[0]], [v2[1], v3[1]], color='g')
                    ax.plot([v3[0], v4[0]], [v3[1], v4[1]], color='g')
                    ax.plot([v4[0], v1[0]], [v4[1], v1[1]], color='g')

                ax.set_title('cam[{}] | max. err={:.2f} | point: {}'.format(cam_idx, np.max(reproj_errs[cam_idx]),
                                                                            np.argmax(reproj_errs[cam_idx])))
                ax.set_xticks([]), ax.set_yticks([])
                idx += 1
        plt.suptitle('{}.pgm | mean={:.2f} | max={:.2f}'.format(img_name_str, mean_err, max_err))

    @classmethod
    def load_triangulation_output(cls, trian_path, configs):
        cam_parameters = {}
        frames = []
        img_pts_path = ''
        with open(trian_path, 'r') as f:
            lines = f.readlines()
            f_line = lines[0].split()
            num_cams = int(f_line[1])
            img_pts_path = f_line[2]
            num_frames = len(lines) - num_cams - 1

            i = 1
            for cam_idx in range(num_cams):
                line = lines[i + cam_idx]
                param = line.split()
                ax = np.array([param[0], param[1], param[2]]).astype(np.float32)
                t = np.array([param[3], param[4], param[5]]).astype(np.float32)
                fx = float(param[6])
                fy = float(param[7])
                cx = float(param[8])
                cy = float(param[9])
                k = np.array([param[10], param[11], param[12], param[13], param[14], param[15]]).astype(np.float32)
                p = np.array([param[16], param[17], param[18], param[19], param[20], param[21]]).astype(np.float32)
                out = {}
                out['ax'] = ax
                out['t'] = t
                out['fx'] = fx
                out['fy'] = fy
                out['cx'] = cx
                out['cy'] = cy
                out['k1'] = k[0]
                out['k2'] = k[1]
                out['k3'] = k[2]
                out['k4'] = k[3]
                out['k5'] = k[4]
                out['k6'] = k[5]

                out['p1'] = p[0]
                out['p2'] = p[1]
                out['p3'] = p[2]
                out['p4'] = p[3]
                out['p5'] = p[4]
                out['p6'] = p[5]
                cam_parameters[cam_idx] = out

            i += num_cams
            for frame_idx in range(num_frames):
                line = lines[i + frame_idx]
                v = line.split()
                frame = {}
                frame['frame_idx'] = frame_idx
                frame['img_name'] = v[0]
                detections = np.array(v[1:1 + num_cams]).astype(int).tolist()
                frame['detections'] = detections
                wp = np.array(v[2 + num_cams:2 + num_cams + 88 * 3]).astype('float32')
                wp = wp.reshape((88, 3))
                frame['world_points'] = wp
                costs = np.array(v[2 + num_cams + 88 * 3:2 + num_cams + 88 * 3 + 88]).astype('float32')
                frame['costs'] = costs
                frames.append(frame)
            f.close()
        return cam_parameters, frames, img_pts_path

    @classmethod
    def reproject_triangulation_output(cls, cam_params, frames, configs):
        # one for each frame
        all_image_points = {}
        num_frames = len(frames)

        for frame_idx, frame in enumerate(frames):
            if frame_idx % 100 == 0:
                print('  {}/{}'.format(frame_idx, num_frames), end='')
            image_name = frame['img_name']
            world_points = frame['world_points']

            # reproject
            image_points = {}
            detections = frame['detections']
            for cam_idx, detected in enumerate(detections):
                if detected:
                    img_points = cls.reproject_world_points(cam_params[cam_idx], world_points, configs)
                    image_points[cam_idx] = img_points
            if len(image_points.keys()) < 2:
                print(image_name)
            all_image_points[image_name] = image_points
        return all_image_points

    @classmethod
    def load_triangulation_inputs(cls, path):
        image_points = {}
        with open(path, 'r') as data:
            j = json.load(data)
            configs = j['configs']
            num_frames = configs['num_frames']
            num_cams = configs['num_cams']
            num_corners = configs['chb']['num_corners']
            frames = j['frames']
            for frame_idx, frame in enumerate(frames):
                img_pt_cur = {}
                img_name = frame['img_name']
                img_pts = frame['img_pts']
                for cam_idx in range(num_cams):
                    if str(cam_idx) in img_pts:
                        img_pt = img_pts[str(cam_idx)]
                        img_pt_cur[cam_idx] = np.array(img_pt).astype('float32').reshape((num_corners, 2))
                image_points[img_name] = img_pt_cur
            data.close()
        return image_points

    @classmethod
    def compute_triangulatoin_reproj_errs(cls, preds_in, opencvs_in, configs):
        all_errs = []
        err_data = {}
        sanity_err = 0
        for img_name, pts_preds in preds_in.items():
            curr_errs = {}
            for cam_idx in range(16):
                if cam_idx in pts_preds:
                    pts_pred = pts_preds[cam_idx]
                    pts_mea = opencvs_in[img_name][cam_idx]
                    dudv = pts_pred - pts_mea
                    dudu_dvdv_sum = np.sum(dudv ** 2, axis=1)

                    if configs['loss_type'] == 'huber':
                        delta = configs['loss_huber_delta']
                        for ei in range(dudu_dvdv_sum.shape[0]):
                            if dudu_dvdv_sum[ei] > delta:
                                sanity_err += 2 * delta * np.sqrt(dudu_dvdv_sum[ei]) - 1 * delta ** 2
                            else:
                                sanity_err += dudu_dvdv_sum[ei]
                    else:
                        sanity_err += np.sum(dudu_dvdv_sum)

                    all_errs.extend(np.sqrt(dudu_dvdv_sum).tolist())
                    curr_errs[cam_idx] = np.sqrt(dudu_dvdv_sum)
            err_data[img_name] = curr_errs

        sanity_err *= 0.5
        return all_errs, err_data, sanity_err
