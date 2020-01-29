import glob
import json
import cv2
import os
import numpy as np
from modules.parser import *
from modules.calibrator import *

class Generator:
    cams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    @classmethod
    def generate_bund_adj_initial_values_16cams(cls, root_path, work_path, configs):
        print()
        print('Generate bundle adjustment initial values')
        output_path = work_path + '\\BundleAdjustment\\input'

        """
        # camera parameters
        """
        param_path = work_path + '\\SingleCalibrations'
        keys = ['rvec_ext', 'tvec_ext', 'M', 'd']
        num_cams = configs.num_cams
        cam_params = {}
        print('Cameras:', end='')
        for cam_idx in range(num_cams):
            print(' [{}/{}]'.format(cam_idx + 1, num_cams), end='')
            xml_path = param_path + '\\cam_param_' + cls.cams[cam_idx] + '.xml'
            params = Parser.load_xml(xml_path, keys)
            rvec = params['rvec_ext'].flatten()
            tvec = params['tvec_ext'].flatten()
            M = params['M']
            d = params['d'][0]
            p = {
                'rvec': list(rvec),
                'tvec': list(tvec),
                'fx': M[0, 0], 'fy': M[1, 1],
                'cx': M[0, 2], 'cy': M[1, 2],
                'k1': d[0], 'k2': d[1], 'k3': d[4], 'k4': 0, 'k5': 0, 'k6': 0,
                'p1': d[2], 'p2': d[3], 'p3': 0, 'p4': 0, 'p5': 0, 'p6': 0
            }
            cam_params[cam_idx] = p

        """
        # initial world points
        """
        print()
        chb = {}
        world_points = Calibrator.compute_initial_worldpoints_using_PnP(root_path, work_path, configs)
        img_names = sorted(world_points.keys())
        print('Chb world points to 6dof')
        for frame_idx, img_name in enumerate(img_names):
            if frame_idx % 100 == 0:
                print('  {}/{}'.format(frame_idx, len(img_names)), end='')
            wp = np.array(world_points[img_name])
            p0 = wp[0, :]
            p1 = wp[10, :]
            p2 = wp[77, :]

            dx = -(p1 - p0)
            dx /= np.linalg.norm(dx)
            dy = p2 - p0
            dy /= np.linalg.norm(dy)
            dz = np.cross(dx, dy)
            dy = np.cross(dz, dx)
            dx = np.cross(dy, dz)

            Rchb = np.array([[dx[0], dy[0], dz[0]],
                             [dx[1], dy[1], dz[1]],
                             [dx[2], dy[2], dz[2]]])
            rvec, _ = cv2.Rodrigues(Rchb)
            tvec = p0

            chb_data = {
                'world_pts': list(world_points[img_name]),
                'frame_idx': frame_idx,
                'rvec': rvec.flatten().tolist(),
                'tvec': tvec.tolist()
            }
            chb[img_name] = chb_data
        print()

        """
        # configs
        """
        configs_out = {
            'num_cams': configs.num_cams,
            'num_cam_params': 22,
            'num_corners': configs.Chb.num_corners,
            'num_frames': len(img_names)
        }

        output_data = {
            'configs': configs_out,
            'cam_params': cam_params,
            'chb': chb
        }

        with open(output_path + '\\bund_adj_inital_params.json', 'w+') as f:
            json.dump(output_data, f, indent=4)
            f.close()


        print('Done. Saved to: {}'.format(output_path))

    @classmethod
    def generate_bund_adj_initial_values_8cams(cls):
        print('generate_bund_adj_initial_values')
        # input path
        input_path = r'C:\Users\joont\Documents\PycharmProjects\CameraCalibration\data\BundleAdjustment\input\previous_ba_output\bundleadjustment_output_828frames.txt'
        output_path = r'C:\Users\joont\Documents\PycharmProjects\CameraCalibration\data\BundleAdjustment\input'

        start_idx = 1659
        num_frames = 828
        frames = {}
        with open(input_path, 'r') as f:
            lines = f.readlines()

            configs = {}
            # configs
            configs['num_cams'] = 8
            configs['num_cam_params'] = 22
            configs['num_corners'] = 88
            configs['num_frames'] = 828

            # extract camera parameters
            line = lines[830]
            v = line.split()
            cam_dic = {}
            for cam_idx in range(8):
                i = cam_idx * 22
                rvec = np.array(v[i:i+3]).astype('float32').tolist()
                i += 3
                tvec = np.array(v[i:i+3]).astype('float32').tolist()
                i += 3
                fx = float(v[i + 0])
                fy = float(v[i + 1])
                cx = float(v[i + 2])
                cy = float(v[i + 3])
                i += 4
                k1 = float(v[i + 0])
                k2 = float(v[i + 1])
                k3 = float(v[i + 2])
                k4 = float(v[i + 3])
                k5 = float(v[i + 4])
                k6 = float(v[i + 5])
                i += 6
                p1 = float(v[i + 0])
                p2 = float(v[i + 1])
                p3 = float(v[i + 2])
                p4 = float(v[i + 3])
                p5 = float(v[i + 4])
                p6 = float(v[i + 5])
                cam_dic[cam_idx] = {
                    'rvec': rvec, 'tvec': tvec, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                    'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6,
                    'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6
                }

            # extract chb world points
            chb = {}
            frame_idx = 0
            for i in range(831, 831 + num_frames):
                v = lines[i].split()
                img_name = v[1]
                wps = []
                for p_idx in range(88):
                    p = [float(v[10 + p_idx * 3]), float(v[10 + p_idx * 3 + 1]), float(v[10 + p_idx * 3 + 2])]
                    wps.append(p)
                if img_name in chb:
                    chb[img_name].update({'world_pts': wps, 'frame_idx': frame_idx})
                else:
                    chb[img_name] = {'world_pts': wps, 'frame_idx': frame_idx}
                frame_idx += 1

            # extract 6dof chb
            for i in range(start_idx, start_idx + num_frames):
                v = lines[i].split()
                img_name = v[1]
                rvec = [float(v[10]), float(v[11]), float(v[12])]
                tvec = [float(v[13]), float(v[14]), float(v[15])]

                if img_name in chb:
                    chb[img_name].update({'rvec': rvec, 'tvec': tvec})
                else:
                    chb[img_name] = {'rvec': rvec, 'tvec': tvec}

            frames['configs'] = configs
            frames['cam_params'] = cam_dic
            frames['chb'] = chb
            f.close()

        with open(output_path + '\\' + 'bund_adj_inital_params.json', 'w+') as outfile:
            json.dump(frames, outfile, indent=4)
            outfile.close()


        print('saved to: {}'.format(output_path + '\\' + 'bund_adj_inital_params.json'))

    @classmethod
    def generate_cam_params_from_bund_adj(cls, root_path, work_path, configs):
        input_path = work_path + '\\BundleAdjustment\\output\\bundle_adjustment_6dof\\bundleadjustment_output.txt'
        output_path = work_path + '\\Triangulation\\input'
        if not os.path.exists(work_path + '\\Triangulation'):
            os.mkdir(work_path + '\\Triangulation')
        if not os.path.exists(work_path + '\\Triangulation\\output'):
            os.mkdir(work_path + '\\Triangulation\\output')

        num_cams = configs.num_cams
        cam_dic = {}
        with open(input_path, 'r') as f:
            lines = f.readlines()
            num_frames = int(lines[0])
            line = lines[num_frames + 2]
            v = line.split()
            for cam_idx in range(num_cams):
                i = cam_idx * 22
                rvec = np.array(v[i:i+3]).astype('float32').tolist()
                i += 3
                tvec = np.array(v[i:i+3]).astype('float32').tolist()

                """
                # Raising 1m higher
                R_ext, _ = cv2.Rodrigues(rvec)
                R_se3 = R_ext.T
                t_se3 = -R_ext.T.dot(tvec)
                print('old t_se3=', t_se3)
                t_se3[2] += 1000
                print('new t_se3=', t_se3)
                print()
                tvec = -R_se3.T.dot(t_se3)
                tvec = tvec.tolist()
                rvec = rvec.tolist()
                """

                i += 3
                fx = float(v[i + 0])
                fy = float(v[i + 1])
                cx = float(v[i + 2])
                cy = float(v[i + 3])
                i += 4
                k = np.array(v[i:i+6]).astype('float32').tolist()
                i += 6
                p = np.array(v[i:i+6]).astype('float32').tolist()
                cam_dic[cam_idx] = {
                    'rvec': rvec, 'tvec': tvec, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                    'k1': k[0], 'k2': k[1], 'k3': k[2], 'k4': k[3], 'k5': k[4], 'k6': k[5],
                    'p1': p[0], 'p2': p[1], 'p3': p[2], 'p4': p[3], 'p5': p[4], 'p6': p[5]
                }
            f.close()

        configs = {'num_cams': num_cams, 'num_cam_params': 22}

        # export
        output_json = {
            'configs': configs,
            'cam_params': cam_dic
        }

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        with open(output_path + '\\cam_params.json', 'w+') as outfile:
            json.dump(output_json, outfile, indent=4)
        print('Saved to: {}'.format(output_path))

    @classmethod
    def generate_bund_adj_input(cls, root_path, work_path, configs):
        print('generate_bund_adj_input')
        output_path = work_path + '\\BundleAdjustment\\output'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = work_path + '\\BundleAdjustment\\output\\bundle_adjustment_6dof'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = work_path + '\\BundleAdjustment\\input'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # configs
        num_cams = configs.num_cams
        num_corners = 88
        num_rows = 8
        num_cols = 11
        chb_sqr_size = 60  # mm

        # initial parameters
        input_cam_params = work_path + '\\SingleCalibrations'

        # corner image points
        image_points_dic = {}  # frame_name : {cam_idx: [img_pts]}
        img_names_int = []  # list used for sorting image names when exporting to json
        for cam_idx in range(num_cams):
            print('  camera[{}]'.format(cam_idx))
            print('  ', end='')
            corner_path = root_path + '\\' + cls.cams[cam_idx] + '\\Corners\*.txt'
            frame_txts = glob.glob(corner_path)
            for idx, frame_txt in enumerate(frame_txts):
                if idx % 100 == 0:
                    print('  [{}/{}]'.format(idx, len(frame_txts)), end='')
                idx += 1

                with open(frame_txt, 'r') as f:
                    lines = f.readlines()
                    if lines[0][0] == 'T' or lines[0][0] == 't':
                        img_name = frame_txt.split('\\')[-1].split('.')[0]
                        img_name = img_name[1:]
                        img_pts = []
                        for p_idx in range(num_corners):
                            p = lines[p_idx + 1].split()
                            img_pts.append([float(p[0]), float(p[1])])
                        if img_name in image_points_dic:
                            image_points_dic[img_name].update({cam_idx: img_pts})
                        else:
                            image_points_dic[img_name] = {cam_idx: img_pts}
                            img_names_int.append(int(img_name))
            print()

        img_names_int = sorted(img_names_int)
        frame_data = []  # [image_name: string, 'img_pts': {cam_idx: [img_pts]}, 'initial_vals': {'chb_rvecs': 3}, {'chb_tvecs': 3}]
        for frame_idx, img_name_int in enumerate(img_names_int):
            # only write if >= 2 cameras saw chb
            num_detected = 0
            for cam_idx in range(num_cams):
                if cam_idx in image_points_dic[img_name]:
                    num_detected += 1
            if num_detected < 2:
                continue


            img_name = ''
            for i in range(5 - len(str(img_name_int))):
                img_name += '0'
            img_name += str(img_name_int)

            dic = {}

            # image name
            dic['img_name'] = img_name

            # image points
            img_pts = {}
            for cam_idx in range(num_cams):
                if cam_idx in image_points_dic[img_name]:
                    img_pts[cam_idx] = image_points_dic[img_name][cam_idx]
            dic['img_pts'] = img_pts
            frame_data.append(dic)

        num_frames = len(img_names_int)

        # cam params
        cam_parmas_dic = {}
        for cam_idx in range(num_cams):
            cam_path = input_cam_params + '\\cam_param_' + cls.cams[cam_idx] + '.xml'
            cv_file = cv2.FileStorage(cam_path, cv2.FILE_STORAGE_READ)
            M = cv_file.getNode("M").mat()
            d = cv_file.getNode("d").mat()[0]
            rvec = cv_file.getNode("rvec_ext").mat().reshape((1, 3)).tolist()[0]
            tvec = cv_file.getNode("tvec_ext").mat().reshape((1, 3)).tolist()[0]
            cam_parmas_dic[cam_idx] = {'fx': M[0, 0], 'fy': M[1, 1], 'cx': M[0, 2], 'cy': M[1, 2], 'k1': d[0], 'k2': d[1], 'p1': d[2], 'p2': d[3], 'k3': d[4], 'rvec': rvec, 'tvec': tvec}

        configs = {'num_cams': num_cams, 'num_cam_params': 22, 'num_frames': num_frames,  'chb': {'num_corners': num_corners, 'num_rows': num_rows, 'num_cols': num_cols, 'chb_sqr_size': chb_sqr_size}}

        # export
        output_json = {
            'configs': configs,
            'cam_params': cam_parmas_dic,
            'frames': frame_data
        }

        with open(output_path + '\\bund_adj_input.json', 'w+') as outfile:
            json.dump(output_json, outfile, indent=4)

        print('exported to: {}'.format(output_path))

    @classmethod
    def generate_chb_image_points_input(cls, root_path, work_path, configs, options):
        min_u = [1513.4609375, 1201.5557861328125, 1374.582763671875, 962.49951171875, 1266.4234619140625, 1118.775634765625, 803.3776245117188, 984.3426513671875]
        min_v = [87.19129180908203, 142.5582275390625, 77.34129333496094, 113.58232879638672, 95.41093444824219, 92.93460845947266, 98.31956481933594, 116.86554718017578]
        max_u = [3147.451904296875, 2979.2890625, 3328.78271484375, 3092.56689453125, 3146.521728515625, 3303.06982421875, 3059.278076171875, 2932.908935546875]
        max_v = [2140.17333984375, 2141.763427734375, 2141.37744140625, 2141.578125, 2141.6328125, 2141.581787109375, 2141.571044921875, 2141.691162109375]

        exclude_outliers = False
        if 'exclude_outliers' in options:
            exclude_outliers = options['exclude_outliers']

        center_region = False
        if 'center_region' in options:
            center_region = options['center_region']
        frame_range = configs.frame_range

        print('generate_chb_image_points_input')
        output_path = work_path + '\\\BundleAdjustment\\input'

        # configs
        num_cams = configs.num_cams
        num_corners = 88
        num_rows = 8
        num_cols = 11
        chb_sqr_size = 60  # mm

        outliers = Parser.load_outliers(work_path, num_cams)

        # corner image points
        image_points_dic = {}  # frame_name : {cam_idx: [img_pts]}
        img_names_int = []  # list used for sorting image names when exporting to json
        for cam_idx in range(num_cams):
            print('  Camera[{}]'.format(cam_idx), end='')
            corner_path = root_path + '\\' + cls.cams[cam_idx] + '\\Corners\*.txt'
            frame_txts = sorted(glob.glob(corner_path))

            skipped = 0
            used = 0
            for frame_txt in frame_txts:
                img_name_curr = frame_txt.split('\\')[-1].split('.')[0]
                img_name_curr = int(img_name_curr[1:])
                if frame_range is not None:
                    if not (frame_range[0] <= img_name_curr and img_name_curr <= frame_range[1]):
                        continue

                with open(frame_txt, 'r') as f:
                    lines = f.readlines()
                    if lines[0][0] == 'T' or lines[0][0] == 't':
                        img_name = frame_txt.split('\\')[-1].split('.')[0]
                        img_name = img_name[1:]
                        if exclude_outliers:
                            if img_name in outliers[str(cam_idx)]:
                                skipped += 1
                                f.close()
                                continue
                        used += 1
                        img_pts = []
                        for p_idx in range(num_corners):
                            p = lines[p_idx + 1].split()
                            img_pts.append([float(p[0]), float(p[1])])

                        if center_region:
                            within_region = True
                            for p in img_pts:
                                if not ((min_u[cam_idx] < p[0]) and (p[0] < max_u[cam_idx]) and (min_v[cam_idx] < p[1]) and (p[1] < max_v[cam_idx])):
                                    within_region = False
                                    break
                            if not within_region:
                                f.close()
                                continue

                        if img_name in image_points_dic:
                            image_points_dic[img_name].update({cam_idx: img_pts})
                        else:
                            image_points_dic[img_name] = {cam_idx: img_pts}
                            img_names_int.append(int(img_name))
                    f.close()
            print(' | {} frames used, {} frames w/ outliers skipped'.format(used, skipped))

        img_names_int = sorted(img_names_int)
        frame_data = []  # [image_name: string, 'img_pts': {cam_idx: [img_pts]}, 'initial_vals': {'chb_rvecs': 3}, {'chb_tvecs': 3}]

        frame_idx = 0
        for img_name_int in img_names_int:
            img_name = ''
            for i in range(5 - len(str(img_name_int))):
                img_name += '0'
            img_name += str(img_name_int)

            num_detected = 0
            for cam_idx in range(num_cams):
                if cam_idx in image_points_dic[img_name]:
                    num_detected += 1

            if not (num_detected >= 2):
                continue

            dic = {}

            # image name
            dic['img_name'] = img_name
            dic['num_detected'] = num_detected
            # image points
            img_pts = {}
            for cam_idx in range(num_cams):
                if cam_idx in image_points_dic[img_name]:
                    img_pts[cam_idx] = image_points_dic[img_name][cam_idx]

            dic['img_pts'] = img_pts
            dic['frame_idx'] = frame_idx
            frame_idx += 1
            frame_data.append(dic)

        num_frames = len(frame_data)
        configs = {'num_cams': num_cams, 'num_frames': num_frames,  'chb': {'num_corners': num_corners, 'num_rows': num_rows, 'num_cols': num_cols, 'chb_sqr_size': chb_sqr_size}}

        # export
        output_json = {
            'configs': configs,
            'frames': frame_data
        }

        with open(output_path + '\\image_points.json', 'w+') as outfile:
            json.dump(output_json, outfile, indent=4)

        print('{} frames. Exported to: {}'.format(num_frames, output_path))

    @classmethod
    def generate_detection_results(cls, root_path, work_path, configs):
        output_path = work_path + '\\detection_results'
        num_cams = configs.num_cams

        detection_results = {}
        for cam_idx in range(num_cams):
            print('Camera[{}]'.format(cam_idx))
            corners_path = glob.glob(root_path + '\\Converted\\' + cls.cams[cam_idx] + '\\Corners\\*.txt')
            for idx, corner_path in enumerate(corners_path):
                if idx % 100 == 0:
                    print('  {}/{}'.format(idx, len(corners_path)), end='')
                img_name = corner_path.split('\\')[-1].split('.')[0]
                with open(corner_path, 'r') as f:
                    line = f.readline()
                    detected = int(line[0][0] == 'T' or line[0][0] == 't')
                    if img_name in detection_results:
                        detection_results[img_name].update({cam_idx: detected})
                    else:
                        detection_results[img_name] = {cam_idx: detected}
                    f.close()
            print()

        with open(output_path + '.json', 'w+') as f:
            json.dump(detection_results, f, indent=4)
            f.close()

        print('Done. Saved to: {}'.format(output_path))

    @classmethod
    def cam_params_json_to_txt(cls, input_path, output_path, configs):
        num_cams = configs.num_cams
        output_str = []
        with open(input_path, 'r') as f:
            j = json.load(f)

            p = j['cam_params']
            for cam_idx in range(num_cams):
                line = []
                c = p[str(cam_idx)]
                rvec = c['rvec']
                tvec = c['tvec']
                fx = c['fx']
                fy = c['fy']
                cx = c['cx']
                cy = c['cy']
                k1 = c['k1']
                k2 = c['k2']
                k3 = c['k3']
                k4 = c['k4']
                k5 = c['k5']
                k6 = c['k6']
                p1 = c['p1']
                p2 = c['p2']
                p3 = c['p3']
                p4 = c['p4']
                p5 = c['p5']
                p6 = c['p6']
                line = [str(cam_idx), ' ', str(rvec[0]), ' ', str(rvec[1]), ' ', str(rvec[2]), ' ', str(tvec[0]), ' ', str(tvec[1]), ' ', str(tvec[2]), ' ', str(fx), ' ', str(fy), ' ', str(cx), ' ', str(cy), ' ', str(k1), ' ', str(k2), ' ', str(k3), ' ', str(k4), ' ', str(k5), ' ', str(k6), ' ', str(p1), ' ', str(p2), ' ', str(p3), ' ', str(p4), ' ', str(p5), ' ', str(p6), '\n']
                output_str.extend(line)
            f.close()
        with open(output_path, 'w+') as f:
            f.writelines(output_str)
            f.close()
        print('Saved to:', output_path)
    @classmethod
    def export_cam_params_to_txt(cls, work_path, configs, param_path):
        output_path = work_path + '\\FinalCamParams'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_path += '\\cam_params.txt'

        num_cams = configs.num_cams
        num_cam_params = 22

        strs = []
        with open(param_path, 'r') as f:
            lines = f.readlines()
            num_frames = int(lines[0])
            cam_line = lines[num_frames + 2]
            v = cam_line.split()
            for cam_idx in range(num_cams):
                p = v[cam_idx * num_cam_params:(cam_idx + 1) * num_cam_params]
                strs.append(' '.join(p))
                if cam_idx < num_cams - 1:
                    strs.append('\n')
                print(p)
            f.close()


        with open(output_path, 'w+') as f:
            f.writelines(strs)
            f.close()
        print('Done: {}'.format(output_path))

