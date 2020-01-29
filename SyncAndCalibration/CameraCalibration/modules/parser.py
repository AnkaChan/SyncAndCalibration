import numpy as np
import glob
import json
import cv2
import os
import shutil
import random
class Parser:
    cams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    @classmethod
    def load_outliers(cls, work_path, num_cams):
        outliers = {}  # cam_idx : [img_name]
        input_outliers_path = work_path + '\\Outliers\\outliers.json'
        with open(input_outliers_path, 'r') as f:
            outliers = json.load(f)
            f.close()
        # convert list to set
        for cam_idx in range(num_cams):
            if str(cam_idx) in outliers:
                img_list = outliers[str(cam_idx)]
            else:
                img_list = []
            outliers[str(cam_idx)] = set(img_list)
        return outliers


    @classmethod
    def load_corner_txt(cls, path):
        pts = []
        with open(path, 'r') as f:
            ret = f.readline()
            if (ret[0] is 't') or (ret[0] is 'T'):
                lines = f.readlines()
                for corner_idx, line in enumerate(lines):
                    uv = line.split()
                    pts.append(np.array([uv[0], uv[1]], dtype=np.float32))
            f.close()
        return pts

    @classmethod
    def sample_image_points(cls, root_path, cam_idx, configs, outliers, target_img_name=None):

        full_path = root_path + '\\' + cls.cams[cam_idx] + '\\Corners\\*.txt'
        img_points = []
        file_list = glob.glob(full_path)
        file_list_no_outliers = []
        outlier_images = outliers[str(cam_idx)]

        for file in file_list:
            image_name = file.split('\\')[-1]
            img_name = image_name[1:]
            if img_name not in outlier_images:
                file_list_no_outliers.append(file)

        if target_img_name is not None:
            for file_path in file_list:
                curr_img_name = file_path.split('\\')[-1].split('.')[0][1:]
                if curr_img_name != target_img_name[1:]:
                    continue
                pts = cls.load_corner_txt(file_path)
                if len(pts) > 0:
                    img_points.append(pts)
        else:
            rand_indices = random.sample(range(len(file_list_no_outliers)), len(file_list_no_outliers))
            for i in rand_indices:
                file_path = file_list_no_outliers[i]
                pts = cls.load_corner_txt(file_path)
                if len(pts) > 0:
                    img_points.append(pts)
                if len(img_points) >= configs.num_single_calib_imgs:
                    break

        return np.array(img_points)

    @classmethod
    def merge_detection_results(cls, root_path, work_path, configs_in):
        print('Merge detection results')

        # 'img_name' : {'cam_idx' : 0/1}
        detections = {}
        num_cams = configs_in.num_cams
        num_detections = {}
        for cam_idx in range(num_cams):
            corner_path = root_path + '\\' + cls.cams[cam_idx] + '\\Corners'
            corners_list = sorted(glob.glob(corner_path + '\\*.txt'))
            print(cls.cams[cam_idx], ':', len(corners_list), 'corners')
            for corner_path in corners_list:
                img_name = corner_path.split('\\')[-1].split('.')[0]
                img_name_num = img_name[1:]
                corners = cls.load_corner_txt(corner_path)
                res = int(len(corners) > 0)
                if img_name_num in detections:
                    detections[img_name_num].update({cam_idx: res})
                else:
                    detections[img_name_num] = {cam_idx: res}

                # num of detections
                if res:
                    if cam_idx in num_detections:
                        num_detections[cam_idx] += 1
                    else:
                        num_detections[cam_idx] = 1

        # sort image names
        results = {'num_frames': len(detections.keys()), 'num_cams': num_cams, 'num_detections': num_detections, 'detections': detections}
        output = work_path + '\\detection_result.json'
        with open(output, 'w+') as f:
            json.dump(results, f, indent=4)
            f.close()
        print('Saved to:\n  {}'.format(output))

    @classmethod
    def load_pair_image_points(cls, root_path, pair, configs):
        output_0 = []
        output_1 = []

        cam0_corners = root_path + '\\' + cls.cams[pair[0]] + '\\Corners\*.txt'
        cam0_corner_paths = glob.glob(cam0_corners)
        rand_indices = random.sample(range(len(cam0_corner_paths)), len(cam0_corner_paths))
        count = 0
        for i in rand_indices:
            path0 = cam0_corner_paths[i]
            pts0 = cls.load_corner_txt(path0)
            if len(pts0) > 0:
                img_name = cls.cams[pair[1]] + path0.split('\\')[-1][1:]
                path1 = root_path + '\\' + cls.cams[pair[1]] + '\\Corners\\' + img_name
                pts1 = cls.load_corner_txt(path1)
                if len(pts1) > 0:
                    # pair image points
                    output_0.append(pts0)
                    output_1.append(pts1)
                    count += 1
                    if count >= configs.num_stereo_imgs:
                        break
        return np.array(output_0), np.array(output_1)


    @classmethod
    def export_xml(cls, path, dic):
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        for k, v in dic.items():
            fs.write(k, v)
        fs.release()
        print('  Saved to: {}'.format(path))

    @classmethod
    def load_xml(cls, path, keys):
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        output = {}
        for key in keys:
            output[key] = fs.getNode(key).mat()
        fs.release()
        return output

    @classmethod
    def merge_intrinsics_extrinsics_xml(cls, work_path, cam_idx):
        int_path = work_path + '\\SingleCalibrations\\intrinsics_' + cls.cams[cam_idx] + '.xml'
        ext_path = work_path + '\\SingleCalibrations\\extrinsics_' + cls.cams[cam_idx] + '.xml'
        print(ext_path)
        print(int_path)
        intrinsics = cls.load_xml(int_path, ['M', 'd'])
        extrinsics = cls.load_xml(ext_path, ['rvec', 'tvec'])

        Rext, _ = cv2.Rodrigues(extrinsics['rvec'])
        text = extrinsics['tvec'].reshape(3, )
        Rse3 = Rext.T
        rvec_se3, _ = cv2.Rodrigues(Rse3)
        tvec_se3 = -Rext.T.dot(text)

        output = {}
        output['M'] = intrinsics['M']
        output['d'] = intrinsics['d']
        output['rvec_ext'] = extrinsics['rvec']
        output['tvec_ext'] = extrinsics['tvec']
        output['rvec_se3'] = rvec_se3
        output['tvec_se3'] = tvec_se3

        output_path = work_path + '\\SingleCalibrations\\cam_param_' + cls.cams[cam_idx] + '.xml'
        cls.export_xml(output_path, output)

        int_path_deleted = work_path + '\\SingleCalibrations\\deleted\intrinsics_' + cls.cams[cam_idx] + '.xml'
        ext_path_deleted = work_path + '\\SingleCalibrations\\deleted\extrinsics_' + cls.cams[cam_idx] + '.xml'

        if not os.path.exists(work_path + '\\SingleCalibrations\\deleted'):
            os.makedirs(work_path + '\\SingleCalibrations\\deleted')

        if os.path.exists(int_path_deleted):
            os.remove(int_path_deleted)
        if os.path.exists(ext_path_deleted):
            os.remove(ext_path_deleted)

        os.rename(int_path, int_path_deleted)
        os.rename(ext_path, ext_path_deleted)

    @classmethod
    def change_corner_folder_name(cls, root_path, folder_name):
        for cam_idx in range(12, 13):
            print('Camera {}'.format(cls.cams[cam_idx]))
            src = root_path + '\\' + cls.cams[cam_idx] + '\\' + folder_name
            trg = root_path + '\\' + cls.cams[cam_idx] + '\\Corners'
            if os.path.exists(src):
                os.rename(src, trg)
                print('  {}\n  ->{}'.format(src, trg))

            corner_list = glob.glob(trg + '\\*.txt')

            i = 0
            # [DEFAULT] Add camera letter in front of .txt file name
            for corner in corner_list:
                v = corner.split('\\')
                new_name = '\\'.join(v[0:-1]) + '\\' + cls.cams[cam_idx] + v[-1]
                os.rename(corner, new_name)
                if i == 0:
                    print('  {}\n  ->{}'.format(corner, new_name))
                    i += 1

            # Remove camera letter in front of .txt file name
            # for src_path in corner_list:
            #     img_name0 = src_path.split('\\')[-1].split('.')[0]
            #     if cls.cams[cam_idx] in img_name0:
            #         img_name = img_name0.split(cls.cams[cam_idx])[-1]
            #         trg_path = '\\'.join(src_path.split('\\')[:-1]) + '\\' + img_name + '.txt'
            #         os.rename(src_path, trg_path)
            #         if i == 0:
            #             print('  {}\n  ->{}'.format(src_path, trg_path))
            #             i += 1
            #     else:
            #         break
