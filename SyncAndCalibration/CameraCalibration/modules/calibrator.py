from modules.parser import *
import matplotlib.pyplot as plt
import os
from PIL import Image
import datetime
from configs import *
import shutil
import matplotlib
from modules.chb_corner_detector import *
from distutils.dir_util import copy_tree

class Calibrator:
    cams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    @classmethod
    def now_str(cls):
        now = datetime.now()
        output = '*[{}/{} {}:{}:{}]*'.format(now.month, now.day, now.hour, now.minute, now.second)
        return output

    @classmethod
    def detect_opencv_corners_multiprocess(cls, root_path, work_path, configs):
        """
        root_path = input("Image root directory: ")
        cam_indices_str = input("Camera index range, comma-separated (e.g., 0,15): ").split(',')
        cam_indices = range(int(cam_indices_str[0]), int(cam_indices_str[1]) + 1)
        frame_indices_str = input('Frame index range, comma-separated (e.g., 0,4400): ').split(',')
        frame_indices = range(int(frame_indices_str[0]), int(frame_indices_str[1]) + 1)
        img_scale = float(input("Image scale [0, 1]: "))
        """
        root_log_path = work_path + '\\CornerDetectLog'
        if not os.path.exists(root_log_path):
            os.mkdir(root_log_path)

        cam_indices = configs.cam_range
        img_scale = 0.5


        calib_configs = {'scale': img_scale, 'chb_dim': (Configs.Chb.col, Configs.Chb.row), 'sqr_size': Configs.Chb.size}
        frame_indices = configs.frame_range
        i0 = frame_indices[0]
        i1 = frame_indices[-1] + 1

        threads = []
        for cam_idx in range(cam_indices[0], cam_indices[1] + 1):
            log_path = root_log_path + '\\corner_log_' + cls.cams[cam_idx] + '.txt'
            print('Camera {} | Log: {}'.format(cls.cams[cam_idx], log_path))
            file_list_raw = glob.glob(root_path + '\\' + cls.cams[cam_idx] + '\\*.pgm')

            file_list = []
            for file in file_list_raw[i0:i1]:
                img_name_str = file.split('\\')[-1]
                img_name_str = (img_name_str.split('.')[0].split('_')[-1])[1:]
                img_name_int = int(img_name_str)
                if img_name_int % 5 == 0:
                    file_list.append(file)

            file_list = sorted(file_list)

            # load image and run corner detection
            threads.append(threading.Thread(target=cls.__detect_corners, args=(cam_idx, file_list, calib_configs, log_path)))

        for thread in threads:
            thread.start()

    @classmethod
    def __output_log(cls, path, strs):

        if not os.path.exists(path):
            with open(path, 'w+') as f:
                f.write(strs)
                f.close()
        else:
            with open(path, 'a+') as f:
                f.write(strs)
                f.close()

    @classmethod
    def __load_image(cls, index, image_path, scale):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # scale
        h, w = gray.shape
        h = float(h) * float(scale)
        w = float(w) * float(scale)
        gray_smaller = cv2.resize(gray, (int(w), int(h)))
        return gray, gray_smaller

    @classmethod
    def __detect_corners(cls, cam_idx, path_list, calib_configs, log_path):
        chb_dim = calib_configs['chb_dim']

        # make output folder if not exists
        vs = path_list[0].split('\\')

        now = datetime.now()
        datetime_str = now.strftime("%Y%m%d")

        output_path = '\\'.join(vs[:-1]) + '\\Corners_' + datetime_str

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # load image and detect
        num_success = 0
        num_fail = 0
        num_imgs = len(path_list)
        for i, image_path in enumerate(path_list):
            img_name_str = image_path.split('\\')[-1]
            img_name_str = img_name_str.split('.')[0].split('_')[-1]
            img_name_int = int(img_name_str[1:])


            if img_name_int % 5 is not 0:
                cls.__output_log(log_path, '(skipping: {})\n'.format(img_name_str))
                continue

            # verbose
            path_piece = image_path.split('\\')
            image_name = path_piece[-1].split('.')[0]
            now = datetime.now()
            time = now.strftime("%H:%M:%S")
            cls.__output_log(log_path, '({}/{}) [{}]\n'.format(i + 1, num_imgs, time))

            # load an image
            if 'scale' in calib_configs:
                scale = calib_configs['scale']
            else:
                print('[ERROR] Scale not set. Setting it to 1.0')
                scale = 1.0

            image, image_smaller = cls.__load_image(i, image_path, scale)
            cls.__output_log(log_path, '  - Image loaded: {}\n'.format(image_path))


            # find corners
            cls.__output_log(log_path, '  - Detecting corners.. ')
            ret, corners = cv2.findChessboardCorners(image_smaller, chb_dim, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

            output_txt_path = ''
            str_ret = ''
            output_str_list = []
            # If found, add object points, image points (after refining them)
            if ret:
                ret, corners = cv2.findChessboardCorners(image, chb_dim, cv2.CALIB_CB_ADAPTIVE_THRESH)
                scale_used = 1.0

                if not ret:
                    ret, corners = cv2.findChessboardCorners(image_smaller, chb_dim, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
                    scale_used = scale

                if ret:
                    str_ret = 'SUCCESS: scale={}'.format(scale_used)
                    corners /= scale_used
                    num_success += 1

                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
                    output_str_list.append('True')

                    # corner object to image points string
                    for uv in corners2:
                        uv_str = '\n' + str(uv[0][0]) + ' ' + str(uv[0][1])
                        output_str_list.append(uv_str)
            else:
                str_ret = 'FAIL'
                num_fail += 1
                output_str_list.append('False')

            # export output to .txt file
            image_name_numbers = image_name.split(cls.cams[cam_idx])[-1]
            with open(output_path + '\\' + image_name_numbers + '.txt', 'w+') as f:
                f.writelines((output_str_list))

            str_ret += ' | saved as: ' + image_name_numbers + '.txt'
            # display status to text_widget
            cls.__output_log(log_path, ' | {}\n'.format(str_ret))

            # clean up
            del image

        cls.__output_log(log_path, '\n* Complete! Outputs saved to:\n  {}'.format(output_path))
        print('* Camera ' + cls.cams[cam_idx] + ' Complete! Outputs saved to:\n  {}'.format(output_path))


    @classmethod
    def save_opencv_corners_on_images(cls, root_path, work_path, configs):
        matplotlib.use('Agg')

        cam_range = configs.cam_range

        color = np.arange(0, 88)
        for cam_idx in range(cam_range[0], cam_range[1] + 1):
            print('Camera {}'.format(cls.cams[cam_idx]))
            img_path = root_path + '\\' + cls.cams[cam_idx]
            corner_path = img_path + '\\Corners'
            print(corner_path)
            output_path = work_path + '\\ChbImages\\' + cls.cams[cam_idx]
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            corner_files = sorted(glob.glob(corner_path + '\\*.txt'))
            sz = 1
            dpi = 300
            for i, corner_file in enumerate(corner_files):
                if i % 100 == 0:
                    print('  [{}/{}]'.format(i, len(corner_files)), end='')
                img_points = Parser.load_corner_txt(corner_file)
                img_name = corner_file.split('\\')[-1].split('.')[0]
                img_path_curr = img_path + '\\' + img_name + '.pgm'
                img = cv2.imread(img_path_curr)
                plt.figure()
                plt.imshow(img)
                if len(img_points) > 0:
                    img_points = np.array(img_points).astype(np.float)
                    plt.scatter(img_points[:, 0], img_points[:, 1], s=sz, c=color, cmap='prism')

                plt.title(img_path_curr)

                mng = plt.get_current_fig_manager()
                mng.full_screen_toggle()

                plt.xticks([]), plt.yticks([])
                img_output = output_path + '\\' + img_name + '.jpg'
                plt.savefig(img_output, bbox_inches='tight', dpi=dpi)
                plt.close('all')
                del img, img_points
            print()
        print('* Complete!')

    @classmethod
    def compute_initial_worldpoints_using_PnP(cls, root_path, work_path, configs):
        """
        # assuming intrinsics/extrinsics of all cameras are accurate, solve PnP using any 1 camera for each frame
        """
        print('compute_initial_worldpoints_using_PnP')
        output_path = work_path + '\\BundleAdjustment'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_path += '\\input'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # load detection result
        detect_res_path = work_path + '\\detection_result.json'
        print(detect_res_path)
        j = {}
        with open(detect_res_path, 'r') as f:
            j = json.load(f)
            f.close()

        num_cams = configs.num_cams
        world_points = {}
        idx = 0
        for img_name, detections in j['detections'].items():
            if idx % 100 == 0:
                print('  {}/{}'.format(idx, len(j['detections'].keys())))
            idx += 1
            num_detected = 0
            for cam_idx in range(num_cams):
                detected = detections[str(cam_idx)]
                if detected == 1:
                    # solve PnP
                    wps = cls.__solve_PnP(cam_idx, img_name, root_path, work_path, configs)
                    if wps is None:
                        print('[ERROR] unexpected None.')
                        input('Press Enter to continue.')
                        continue
                    world_points[img_name] = wps
                    num_detected += 1
                    break
            if num_detected == 0:
                print('No detection for: {}'.format(img_name + '.pgm'))
        print()


        with open(output_path + '\\initial_world_points.json', 'w+') as f:
            json.dump(world_points, f, indent=4)
            f.close()
        print('Saved to: {}'.format(output_path))
        return world_points


    @classmethod
    def __solve_PnP(cls, cam_idx, img_name, root_path, work_path, configs):
        # load intrinsics
        xml_path = work_path + '\\SingleCalibrations\\deleted\\intrinsics_' + cls.cams[cam_idx] + '.xml'
        keys = ['M', 'd']
        intrinsics = Parser.load_xml(xml_path, keys)
        # load extrinsics
        xml_path = work_path + '\\SingleCalibrations\\deleted\\extrinsics_' + cls.cams[cam_idx] + '.xml'
        keys = ['rvec', 'tvec']
        extrinsics = Parser.load_xml(xml_path, keys)
        Rext_cam, _ = cv2.Rodrigues(extrinsics['rvec'])
        text_cam = extrinsics['tvec']
        Rse3 = Rext_cam.T
        tse3 = -Rext_cam.T.dot(text_cam).reshape(3, )

        # load image points
        corner_path = root_path + '\\' + cls.cams[cam_idx] + '\\Corners\\' + cls.cams[cam_idx] + img_name + '.txt'
        _2d_points = Parser.load_corner_txt(corner_path)
        if len(_2d_points) > 0:
            _3d_points = configs.Chb.obj_points

            # calibrate extrinsics: rvecs, tvecs maps from world coord. sys. to camera coord. sys.
            ret, rvec, tvec = cv2.solvePnP(_3d_points, np.array(_2d_points), intrinsics['M'], intrinsics['d'])

            if ret:
                img_name = corner_path.split('\\')[-1].split('.')[0]

                # rvec is extrinsics of cam_idx
                Rext, _ = cv2.Rodrigues(rvec)
                text = tvec.reshape(3, )
                world_pts = []

                for p_idx in range(configs.Chb.num_corners):
                    chb_default = configs.Chb.obj_points[p_idx, :]
                    wp = Rse3.dot(Rext.dot(chb_default) + text) + tse3
                    world_pts.append(wp.tolist())
                return world_pts
        return None



    @classmethod
    def __solve_PnPs_single_cam(cls, root_path, work_path, configs):
        """

        Not part of the calibration framework

        """
        print('\nsolve_PnPs')
        """
        # assuming intrinsics/extrinsics of Camera A is accurate, obtain initial world chb points using PnP for all chb frames
        """
        cam_idx = configs.center_cam_idx
        print('  Camera[{}]'.format(cam_idx))
        output_path = work_path + '\\PnPOutput'

        # load intrinsics
        xml_path = work_path + '\\SingleCalibrations\\intrinsics_' + cls.cams[cam_idx] + '.xml'
        keys = ['M', 'd']
        intrinsics = Parser.load_xml(xml_path, keys)
        # load extrinsics
        xml_path = work_path + '\\SingleCalibrations\\extrinsics_' + cls.cams[cam_idx] + '.xml'
        keys = ['rvec', 'tvec']
        extrinsics = Parser.load_xml(xml_path, keys)
        Rext_cam, _ = cv2.Rodrigues(extrinsics['rvec'])
        text_cam = extrinsics['tvec']
        Rse3 = Rext_cam.T
        tse3 = -Rext_cam.T.dot(text_cam).reshape(3,)

        # load image points
        corner_path_list = glob.glob(root_path + '\\' + cls.cams[cam_idx] + '\\Corners\\*.txt')
        world_points = {}
        for idx, corner_path in enumerate(corner_path_list):
            if idx % 100 == 0:
                print('  [{}/{}]'.format(idx, len(corner_path_list)))
            _2d_points = Parser.load_corner_txt(corner_path)
            if len(_2d_points) > 0:
                _3d_points = configs.Chb.obj_points

                # calibrate extrinsics: rvecs, tvecs maps from world coord. sys. to camera coord. sys.
                ret, rvec, tvec = cv2.solvePnP(_3d_points, np.array(_2d_points), intrinsics['M'], intrinsics['d'])

                if ret:
                    img_name = corner_path.split('\\')[-1].split('.')[0]

                    # rvec is extrinsics of cam_idx
                    Rext, _ = cv2.Rodrigues(rvec)
                    text = tvec.reshape(3,)
                    world_pts = []

                    for p_idx in range(configs.Chb.num_corners):
                        chb_default = configs.Chb.obj_points[p_idx, :]
                        wp = Rse3.dot(Rext.dot(chb_default) + text) + tse3
                        world_pts.append(wp.tolist())
                    world_points[img_name] = world_pts
        out_json = output_path + '\\chb_pnp_' + cls.cams[cam_idx]
        with open(out_json + '.json', 'w+') as f:
            json.dump(world_points, f, indent=4)
            f.close()

        print('Done. Saved: {}'.format(out_json))

    @classmethod
    def copy_intial_camera_parameters(cls, from_path, to_path):
        copy_tree(from_path, to_path)

    @classmethod
    def compute_initial_camera_parameters(cls, root_path, work_path, intrinsics_cam_idx, configs):
        print('\nCompute initial camera parameters')

        calib_intrinsics = 1
        calib_extrinsics = 1
        calib_stereos = 1
        merge_int_ext = 1
        # root_path = r'D:\Utah\2019_08_09_AllPoseCapture\Converted'

        # get inputs first
        if calib_stereos:
            center_cam_idx = configs.center_cam_idx
            num_cams = configs.num_cams
            pairs = []
            for cam_idx in range(num_cams):
                if cam_idx != center_cam_idx:
                    pair = (center_cam_idx, cam_idx)
                    pairs.append(pair)
            """
            print()
            print('>> {} pairs loaded | index = 0-14'.format(len(pairs)))
            print('Pair index')
            pair_index0 = int(input('  from: '))
            pair_index1 = int(input('  to (including): '))
            """
            pair_index0 = 0
            pair_index1 = 14

        if calib_intrinsics:
            _2d_points_dic = {}
            cam_indices = range(configs.num_cams)

            if intrinsics_cam_idx < 0:
                # compute intrinsics for all cameras
                for cam_idx in cam_indices:
                    _2d_points_dic[cam_idx] = cls.__calib_intrinsics(root_path, work_path, cam_idx, configs)
                del _2d_points_dic
            else:
                # load intrinsics, if exists. If not, calibrate new
                xml_path = work_path + '\\SingleCalibrations\\intrinsics_' + cls.cams[intrinsics_cam_idx] + '.xml'

                if not os.path.exists(xml_path):
                    _ = cls.__calib_intrinsics(root_path, work_path, intrinsics_cam_idx, configs)

                keys = ['M', 'd']
                intrinsics = Parser.load_xml(xml_path, keys)

                for cam_idx in cam_indices:
                    if cam_idx == intrinsics_cam_idx:
                        continue
                    output_xml_path = work_path + '\\SingleCalibrations\\intrinsics_' + cls.cams[cam_idx] + '.xml'
                    cam_params = {'M': intrinsics['M'], 'd': intrinsics['d']}
                    Parser.export_xml(output_xml_path, cam_params)

        if calib_extrinsics:
            center_cam_idx = configs.center_cam_idx
            center_img_name = cls.cams[center_cam_idx] + configs.center_img_name
            # cam_indices = [center_cam_idx]
            cam_indices = range(0, 16)
            cls.__calib_extrinsics(root_path, work_path, cam_indices, configs, center_img_name)

        if calib_stereos:
            for pair_idx in range(pair_index0, pair_index1 + 1):
                pair = pairs[pair_idx]
                cls.__calib_stereo(root_path, work_path, pair, configs)

        if merge_int_ext:
            num_cams = configs.num_cams
            for cam_idx in range(num_cams):
                Parser.merge_intrinsics_extrinsics_xml(work_path, cam_idx)

    @classmethod
    def __calib_stereo(cls, root_path, work_path, pair, configs):
        print('Stereo between cameras [{}] & [{}]'.format(cls.cams[pair[0]], cls.cams[pair[1]]))
        _2d_points0, _2d_points1 = Parser.load_pair_image_points(root_path, pair, configs)
        print('  ', len(_2d_points0), 'frames loaded.')
        _3d_points = []
        for i in range(len(_2d_points0)):
            _3d_points.append(configs.Chb.obj_points)

        # load intrinsics
        xml_path0 = work_path + '\\SingleCalibrations\\intrinsics_' + cls.cams[pair[0]] + '.xml'
        xml_path1 = work_path + '\\SingleCalibrations\\intrinsics_' + cls.cams[pair[1]] + '.xml'
        keys = ['M', 'd']
        cam0 = Parser.load_xml(xml_path0, keys)
        cam1 = Parser.load_xml(xml_path1, keys)

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC  # we already have intrinsics (initial values)
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS # optmize intrinsics
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        #     flags |= cv2.CALIB_FIX_K1
        #     flags |= cv2.CALIB_FIX_K2
        #     flags |= cv2.CALIB_FIX_K3
        #     flags |= cv2.CALIB_FIX_K4
        #     flags |= cv2.CALIB_FIX_K5
        #     flags |= cv2.CALIB_FIX_K6

        # termination criteria for the iterative optimization algorithm.
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-5)

        # stereo calibrate
        ret, mtx0, dist0, mtx1, dist1, R10, t10, E, F = cv2.stereoCalibrate(_3d_points, _2d_points0, _2d_points1,
                                                                        cam0['M'], cam0['d'], cam1['M'], cam1['d'],
                                                                        configs.image_shape, criteria=criteria, flags=flags)

        if ret:
            print("Stereo calibration SUCCESS: cam[{}] & cam[{}]".format(pair[0], pair[1]))
            print("  Mean proj. error = {:.3f}; inter-distance = {:.3f} [m]".format(ret, np.linalg.norm(t10) * 0.001))

            xml_path = work_path + '\\SingleCalibrations\\extrinsics_' + cls.cams[pair[0]] + '.xml'
            keys = ['rvec', 'tvec']
            cam_param0 = Parser.load_xml(xml_path, keys)
            R0, _ = cv2.Rodrigues(cam_param0['rvec'])
            t0 = cam_param0['tvec'].reshape(3,)
            tw0 = -R0.T.dot(t0)
            Rw0 = R0.T

            # SE3 of camera w.r.t. global coordinates
            R01 = R10.T
            t01 = -R10.T.dot(t10)
            Rse3 = Rw0.dot(R01)
            tse3 = Rw0.dot(t01.reshape(3,)) + tw0

            # back to extrinsics
            t = -Rse3.T.dot(tse3)
            R = Rse3.T

            rvec1, _ = cv2.Rodrigues(R)
            tvec1 = t
            output_xml_path = work_path + '\\SingleCalibrations\\extrinsics_' + cls.cams[pair[1]] + '.xml'
            cam_params = {'rvec': rvec1, 'tvec': tvec1}
            Parser.export_xml(output_xml_path, cam_params)
        else:
            print("[ERROR] Stereo calibration FAILED")

    @classmethod
    def __calib_extrinsics(cls, root_path, work_path, cam_indices, configs, img_name):
        outliers = Parser.load_outliers(work_path, configs.num_cams)

        for cam_idx in cam_indices:
            print('Camera[{}]'.format(cam_idx))
            # load image points
            corner_path = root_path + '\\' + cls.cams[cam_idx] + '\\Corners'
            _2d_points = Parser.sample_image_points(root_path, cam_idx, configs, outliers, img_name)

            # load 3d points
            _3d_points = configs.Chb.obj_points

            # load intrinsics
            xml_path = work_path + '\\SingleCalibrations\\intrinsics_' + cls.cams[cam_idx] + '.xml'
            keys = ['M', 'd']
            intrinsics = Parser.load_xml(xml_path, keys)
            # calibrate extrinsics: rvecs, tvecs maps from world coord. sys. to camera coord. sys.
            ret, rvec, tvec = cv2.solvePnP(_3d_points, _2d_points, intrinsics['M'], intrinsics['d'])

            if ret:
                print('Extrinsics calibration complete for camera [{}]'.format(cam_idx))
                output_xml_path = work_path + '\\SingleCalibrations\\extrinsics_' + cls.cams[cam_idx] + '.xml'
                cam_params = {'rvec': rvec, 'tvec': tvec}
                Parser.export_xml(output_xml_path, cam_params)

    @classmethod
    def __calib_intrinsics(cls, root_path, work_path, cam_idx, configs):
        outliers = Parser.load_outliers(work_path, configs.num_cams)

        # load image points
        _2d_points = Parser.sample_image_points(root_path, cam_idx, configs, outliers)
        # load 3d points
        _3d_points = [configs.Chb.obj_points for i in range(len(_2d_points))]

        print('Calibrating intrinsics: Camera[{}], {} frames.'.format(cam_idx, len(_2d_points)))

        # calibrate intrinsics
        cameraMatrix = None
        distCoeffs = None
        print('  {} start | cv2.calibrateCamera'.format(cls.now_str()))
        rms_err, M, d, _, _ = cv2.calibrateCamera(_3d_points, _2d_points, configs.image_shape, cameraMatrix, distCoeffs)
        print('  {} end | cv2.calibrateCamera'.format(cls.now_str()))
        if rms_err:
            print('  Complete: {} images, RMS error={}'.format(len(_2d_points), rms_err))
            output_xml_path = work_path + '\\SingleCalibrations\\intrinsics_' + cls.cams[cam_idx] + '.xml'
            cam_params = {'rms_err': rms_err, 'M': M, 'd': d}
            Parser.export_xml(output_xml_path, cam_params)
        else:
            print('  Single calibration FAIL')

        return _2d_points



    @classmethod
    def __ChESS(cls, image, mask):
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

    @classmethod
    def __NotChESS(cls, image):
        hh, ww = image.shape
        hh = int(hh / 2.0)
        ww = int(ww / 2.0)

        imgTL = image[0:hh, 0:ww]
        imgTR = image[0:hh, ww:2 * ww]
        imgBR = np.flip(image[hh:2 * hh, ww:2 * ww])
        imgBL = np.flip(image[hh:2 * hh, 0:ww])

        score = (np.sum(np.abs(imgTL - imgBR)) + np.sum(np.abs(imgTR - imgBL))) / 255.0
        return score

    @classmethod
    def __dist(cls, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    @classmethod
    def __crop_square(cls, img, center, length):
        # create mask
        length_half = int(0.5 * length)
        start_point = (center[0] - length_half + 1, center[1] - length_half + 1)
        end_point = (center[0] + length_half, center[1] + length_half)
        crop = img[start_point[1]:end_point[1] + 1, start_point[0]:end_point[0] + 1, :]

        return crop

    @classmethod
    def save_ranked_corner_crops(cls, root_path, output_path, configs):
        """
        # Debug in jupyter notebook, then copy & paste here for deployment
        """
        cam_range = configs.cam_range

        print('Running for cameras [{}]-[{}]'.format(cam_range[0], cam_range[1]))
        # output_path = root_path + '\\..\\CroppedCorners'
        # output_path = work_path + '\\CroppedCorners'
        cams = cls.cams

        """
        Load corners
        """
        # num_cams = 16
        # root_path = r'D:\Pictures\2019_12_03_capture\Converted'

        # num_cams x {img_name : [corner points]}
        corner_data = {}
        num_frames = {}
        img_names_set = set()
        print('Loading: ')
        for cam_idx in range(cam_range[0], cam_range[1] + 1):
            print('  {} | '.format(cams[cam_idx]), end='')

            input_path = root_path + '\\' + cams[cam_idx] + '\\Corners\\*.txt'
            print(input_path)
            file_paths = glob.glob(input_path)

            for f_path in file_paths:
                f_name = f_path.split('\\')[-1]
                img_name = f_name.split('.')[0]
                img_name = img_name[1:]
                img_names_set.add(img_name)
                corners = Parser.load_corner_txt(f_path)
                if len(corners) > 0:
                    if cam_idx in corner_data:
                        corner_data[cam_idx].update({img_name: corners})
                    else:
                        corner_data[cam_idx] = {img_name: corners}

                    if cams[cam_idx] in num_frames:
                        num_frames[cams[cam_idx]] += 1
                    else:
                        num_frames[cams[cam_idx]] = 1

        print('Number of frames loaded:', num_frames)
        img_names_list = sorted(list(img_names_set))


        """
        Create neighbors
        """
        neighbor_indices = []
        for r in range(8):
            for c in range(11):
                neighbors = {}
                if r == 7:
                    s = -1
                else:
                    s = (r + 1) * 11 + c

                if r == 0:
                    n = -1
                else:
                    n = (r - 1) * 11 + c

                if c == 0:
                    w = -1
                else:
                    w = r * 11 + c - 1

                if c == 10:
                    e = -1
                else:
                    e = r * 11 + c + 1

                neighbors['n'] = n
                neighbors['s'] = s
                neighbors['e'] = e
                neighbors['w'] = w
                neighbor_indices.append(neighbors)



        """
        Find min dist
        """
        # compute distance between neighbors for each point
        # {'cam_idx': 'image_name' : {'min': [88], 'max': [88], 'avg': [88]}]}
        neighbor_dists = {}
        for cam_idx in range(cam_range[0], cam_range[1] + 1):
            print()
            print('  Camera[{}] |'.format(cam_idx), end='')
            print('  ', end='')
            dist_dics = {}
            for frame_idx, img_name in enumerate(img_names_list):
                if frame_idx % 100 == 0:
                    print('  {}/{}'.format(frame_idx, len(img_names_list)), end='')

                if img_name not in corner_data[cam_idx]:
                    continue

                corners = corner_data[cam_idx][img_name]
                mins = []
                maxs = []
                avgs = []
                for p_idx, point in enumerate(corners):
                    neigh = neighbor_indices[p_idx]

                    min_dist = np.inf
                    max_dist = -1
                    avg_dist = 0
                    num_neigh = 0
                    for k, neigh_idx in neigh.items():
                        if neigh_idx > -1:
                            num_neigh += 1
                            neigh_point = corners[neigh_idx]
                            dist_curr = cls.__dist(point, neigh_point)
                            avg_dist += dist_curr
                            if dist_curr < min_dist:
                                min_dist = dist_curr
                            if dist_curr > max_dist:
                                max_dist = dist_curr

                    avg_dist /= num_neigh
                    mins.append(min_dist)
                    maxs.append(max_dist)
                    avgs.append(avg_dist)
                curr_frame = {'min': mins, 'max': maxs, 'avg': avgs}
                dist_dics[frame_idx] = curr_frame
            neighbor_dists[cam_idx] = dist_dics
        print()

        """
        ChESS
        """
        import math
        mask = np.zeros((16, 16)).astype('uint8')
        cx = 8.0
        cy = 8.0
        r = int(mask.shape[0] * 0.25) - 0.5
        mask_vu = []
        print('radius =',r)
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

        # fig = plt.figure(figsize=(2, 2))
        # fig.patch.set_facecolor('lightgray')
        # plt.title('{} sample points'.format(np.sum(mask)))
        # plt.imshow(mask)
        # plt.xticks([]), plt.yticks([])
        # plt.show()
        """
        Crop squares
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print()
        print('Cropping starts.')
        print('  Saving to:', output_path)

        # load image and run corner detection
        # threads = []
        export_type = 'json'
        for cam_idx in range(cam_range[0], cam_range[1] + 1):
            # cls.__export_corner_crop_thread(export_type, root_path, output_path, cam_idx, img_names_list, corner_data, mask_vu)
            cls.__export_corner_crop_thread2(export_type, root_path, output_path, cam_idx, img_names_list, corner_data, mask_vu)
        #     threads.append(threading.Thread(target=cls.__export_corner_crop_thread, args=(root_path, output_path, cam_idx, img_names_list, corner_data, mask_vu)))
        #
        # for thread in threads:
        #     thread.start()
        #
        # for thread in threads:
        #     thread.join()
        print('\nAll complete!')
        print(cls.now_str())

    @classmethod
    def __export_corner_crop_thread2(cls, export_type, root_path, output_path, cam_idx, img_names_list, corner_data, mask_vu):
        print('  Camera[{}]'.format(cam_idx))
        h, w = Configs.image_shape
        scale = 1.0
        sz = 20.0
        length = 16.0
        dpi = 150
        x = np.linspace(0, 16, 17)
        y = np.linspace(0, 16, 17)

        for frame_idx, img_name in enumerate(img_names_list):
            if frame_idx % 100 == 0:
                print('  {}/{}'.format(frame_idx, len(img_names_list)), end='')
            if img_name not in corner_data[cam_idx]:
                continue

            img_dir = root_path + '\\' + cls.cams[cam_idx] + '\\' + cls.cams[cam_idx] + img_name + '.pgm'
            img_w = int(w * scale)
            img_h = int(h * scale)

            corners = corner_data[cam_idx][img_name]
            img = cv2.imread(img_dir)
            img = cv2.resize(img, (img_w, img_h))
            for p_idx, corner in enumerate(corners):

                xc = int(scale * corner[0])
                yc = int(scale * corner[1])

                crop = cls.__crop_square(img, (xc, yc), length)
                crop = np.array(crop).astype('uint8')[:, :, 0]
                #             crop = cv2.equalizeHist(crop)
                blur = cv2.GaussianBlur(crop, (3, 3), 0)

                blur_masked, score = cls.__ChESS(blur, mask_vu)
                if score < 500:
                    # score2 = __NotChESS(blur)
                    img_title = '{}_{}_{}_{}'.format(cam_idx, img_name, p_idx, score)
                    # img_title = '{}_{}_{}'.format(cam_idx, img_name, p_idx)
                    # fig = plt.figure()
                    # fig.set_size_inches(16 / dpi, 16 / dpi, forward=False)
                    # ax = plt.Axes(fig, [0., 0., 1., 1.])
                    # ax.set_axis_off()
                    # fig.add_axes(ax)
                    # ax.imshow(crop, cmap='gray')
                    # plt.savefig(output_path + '\\' + img_title + '.jpg', dpi=dpi)
                    # plt.close()


                    fig = plt.figure(figsize=(16 * 320 / dpi, 16 * 320 / dpi), dpi=dpi, tight_layout=True)
                    # fig.patch.set_facecolor('lightgray')
                    plt.axis('off')
                    # plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                    #                 labeltop='off', labelright='off', labelbottom='off')
                    # fig.suptitle(img_title, fontsize=10)
                    plt.grid(False)

                    # visualize mask
                    # for m in mask_vu:
                    #     plt.scatter(m[1], m[0], c='r', marker=',', s=36)
                    plt.imshow(crop, cmap='gray')
                    #                 plt.show()
                    plt.rcParams['savefig.facecolor'] = 'lightgray'
                    plt.savefig(output_path + '\\' + img_title + '.jpg', dpi=dpi)
                    plt.close('all')
                del crop, blur, blur_masked
            del img
        print()

    @classmethod
    def __export_corner_crop_thread(cls, export_type, root_path, output_path, cam_idx, img_names_list, corner_data, mask_vu):
        print('  Camera[{}]'.format(cam_idx))
        h, w = Configs.image_shape
        scale = 1.0
        sz = 20.0
        length = 16.0
        dpi = 150
        x = np.linspace(0, 16, 17)
        y = np.linspace(0, 16, 17)

        for frame_idx, img_name in enumerate(img_names_list):
            if frame_idx % 100 == 0:
                print('  {}/{}'.format(frame_idx, len(img_names_list)), end='')
            if img_name not in corner_data[cam_idx]:
                continue

            img_dir = root_path + '\\' + cls.cams[cam_idx] + '\\' + cls.cams[cam_idx] + img_name + '.pgm'
            img_w = int(w * scale)
            img_h = int(h * scale)

            corners = corner_data[cam_idx][img_name]
            img = cv2.imread(img_dir)
            img = cv2.resize(img, (img_w, img_h))
            for p_idx, corner in enumerate(corners):

                xc = int(scale * corner[0])
                yc = int(scale * corner[1])

                crop = cls.__crop_square(img, (xc, yc), length)
                crop = np.array(crop).astype('uint8')[:, :, 0]
                #             crop = cv2.equalizeHist(crop)
                blur = cv2.GaussianBlur(crop, (3, 3), 0)

                blur_masked, score = cls.__ChESS(blur, mask_vu)
                if score < 1000:
                    # score2 = __NotChESS(blur)
                    img_title = '{}_{}_{}_{}'.format(cam_idx, img_name, p_idx, score)
                    # img_title = '{}_{}_{}'.format(cam_idx, img_name, p_idx)
                    # fig = plt.figure()
                    # fig.set_size_inches(16 / dpi, 16 / dpi, forward=False)
                    # ax = plt.Axes(fig, [0., 0., 1., 1.])
                    # ax.set_axis_off()
                    # fig.add_axes(ax)
                    # ax.imshow(crop, cmap='gray')
                    # plt.savefig(output_path + '\\' + img_title + '.jpg', dpi=dpi)
                    # plt.close()


                    fig = plt.figure(figsize=(16 * 20 / dpi, 16 * 20 / dpi), dpi=dpi)
                    fig.patch.set_facecolor('lightgray')
                    plt.axis('off')
                    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                                    labeltop='off', labelright='off', labelbottom='off')
                    fig.suptitle(img_title, fontsize=10)
                    plt.grid(False)

                    # red lines
                    plt.scatter(7.5, 7.5, s=sz, c='r')
                    plt.scatter([7.5] * len(y), y, s=sz * 0.25, c='r')
                    plt.scatter(x, [7.5] * len(x), s=sz * 0.25, c='r')
                    plt.imshow(crop, cmap='gray')
                    #                 plt.show()
                    plt.rcParams['savefig.facecolor'] = 'lightgray'
                    plt.savefig(output_path + '\\' + img_title + '.jpg', dpi=dpi)
                    plt.close('all')
                del crop, blur, blur_masked
            del img
        print()

    @classmethod
    def determine_corner_outliers(cls, root_path, input_path):
        """
        Determine outliers
        """
        print(cls.now_str())
        # input_path = root_path + '\\..\\CroppedCorners'
        output_path = input_path + '\\Ranked'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        crop_paths = glob.glob(input_path + '\\*.jpg')

        crop_data = {}
        scores = {}
        for c_idx, crop_path in enumerate(crop_paths):
            name = crop_path.split('\\')[-1].split('.jpg')[0]
            names = name.split('_')
            cam_idx = int(names[0])
            img_name = names[1]
            p_idx = int(names[2])
            score = float(names[3])
            scores[c_idx] = score
            crop_data[c_idx] = {'cam_idx': cam_idx, 'img_name': img_name, 'p_idx': p_idx, 'score': score,
                                'path': crop_path, 'file_name': name}

        print('{} cropped corners ranked'.format(len(scores.keys())))
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])

        rank = 0
        i = 0
        for c_idx, score in sorted_scores:
            if i % 500 == 0:
                print(' [{}/{}]'.format(i, len(sorted_scores)), end='')
            i += 1
            c_data = crop_data[c_idx]

            # sanity check
            if c_data['score'] != score:
                print('[ERROR] crop score mismatch!')

            # rename cropped image
            src_dir = c_data['path']
            dir_extract = src_dir.split('\\')[0:-1]
            dst_dir = '\\'.join(dir_extract) + '\\Ranked\\' + str(rank) + '_' + c_data['file_name'] + '.jpg'
            shutil.copy(src_dir, dst_dir)
            rank += 1

        print('\nAll complete!')
        print(cls.now_str())

    @classmethod
    def generate_outliers_txt(cls, root_path, work_path, score_thres, configs, export=False):
        output_json = work_path + '\\Outliers'

        if not os.path.exists(output_json):
            os.mkdir(output_json)


        # fetch ranked crops
        outliers_data = {}
        # ranked_crop_path = root_path + '\\..\\CroppedCorners\\Ranked\\*.jpg'
        ranked_crop_path = work_path + '\\CroppedCorners\\Ranked\\*.jpg'
        print('Loading:', ranked_crop_path)
        ranked_crops = sorted(glob.glob(ranked_crop_path))
        print('  {} cropped images loaded.'.format(len(ranked_crops)))

        for crop_path in ranked_crops:
            file_name = crop_path.split('\\')[-1]
            name = file_name.split('.jpg')[0]
            v = name.split('_')
            score = float(v[4])
            if score < score_thres:
                rank = int(v[0])
                cam_idx = int(v[1])
                img_name = v[2]
                p_idx = int(v[3])
                outliers_data[rank] = {'cam_idx': cam_idx, 'img_name': img_name, 'p_idx': p_idx, 'score': score, 'file_path': crop_path, 'file_name': file_name}

                if img_name == '00605':
                    print(rank,' |', cam_idx, '-',img_name, '-',score)

        print('  {} outliers determined.'.format(len(outliers_data.keys())))

        # save to json
        outlier_imgs_by_cam = {}
        for rank, outlier in outliers_data.items():
            cam_idx = outlier['cam_idx']
            img_name = outlier['img_name']
            if cam_idx in outlier_imgs_by_cam:
                if not (img_name in outlier_imgs_by_cam[cam_idx]):
                    outlier_imgs_by_cam[cam_idx].append(img_name)
            else:
                outlier_imgs_by_cam[cam_idx] = [img_name]
            if img_name == '00605' or rank == 90:
                print(rank,' |', cam_idx, '-',img_name)

        with open(output_json + '\\outliers.json', 'w+') as f:
            json.dump(outlier_imgs_by_cam, f, indent=4)
            f.close()


        print('Done. Saved to:', output_json)

        if export == 'y' or export == 'Y':
            print()
            print('Exporting outlier images for debugging.')
            idx = 0
            for rank, outlier in outliers_data.items():
                if idx % 100 == 0:
                    print(' [{}/{}]'.format(idx, len(outliers_data.keys())))
                src = outlier['file_path']
                folder_path = work_path + '\\Outliers'
                dst = folder_path + '\\' + outlier['file_name']
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                shutil.copyfile(src, dst)
                idx += 1
            print()
            print('Done.')