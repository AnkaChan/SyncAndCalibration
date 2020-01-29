from modules.parser import *
from configs import Configs
from modules.calibrator import *
from modules.renderer import *
from modules.generator import *
from modules.chb_corner_detector import *


def load_user_inputs(path):
    out = {}
    print('Loading user inputs: {}'.format(path))
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            v = line.split()
            key = v[0]
            type = v[1]
            val = v[2]
            if type == 'int':
                val = int(val)
            elif type == 'double' or type == 'float':
                val = float(val)
            out[key] = val
            print('  {} : {}'.format(key, val))
        f.close()
    return out


def main_calibration(run):
    user_inputs = load_user_inputs('user_inputs.txt')

    configs = Configs()
    configs.num_cams = user_inputs['num_cams']

    # run = 2
    if run == 0:
        root_path = user_inputs['root_path']
        work_path = user_inputs['work_path']
        Calibrator.detect_opencv_corners_multiprocess(root_path, work_path, configs)
    elif run == 1:
        Calibrator.save_opencv_corners_on_images()
    elif run == 2:
        # merge detection results into one .txt
        # root_path = r'D:\Pictures\2019_12_03_capture'
        # output_path = r'D:\Pictures\2019_12_03_capture'
        root_path = user_inputs['root_path']
        output_path = root_path
        Parser.merge_detection_results(root_path, output_path, configs)
    elif run == 3:
        cam0 = input('Camera start index: ')
        cam1 = input('Camera end index: ')
        cam_range = (int(cam0), int(cam1))
        # root_path = input('Root path (e.g., r"D:/Pictures/2019_12_03_capture/Converted)": ')
        root_path = user_inputs['root_path']
        # Calibrator.save_ranked_corner_crops(cam_range, root_path)
        Calibrator.save_ranked_corner_crops(cam_range, root_path)
    elif run == 4:
        # root_path = input('Root path (e.g., r"D:/Pictures/2019_12_03_capture/Converted)": ')
        root_path = user_inputs['root_path']
        Calibrator.determine_corner_outliers(root_path)
    elif run == 5:
        root_path = user_inputs['root_path']
        export = input('\nSave outlier images as well? [y/n]: ')
        Calibrator.generate_outliers_txt(root_path, export, configs)
    elif run == 6:
        configs.num_stereo_imgs = user_inputs['num_stereo_imgs']
        configs.center_cam_idx = user_inputs['center_cam_idx']
        configs.center_img_name = user_inputs['center_img_name']
        configs.num_single_calib_imgs = user_inputs['num_single_calib_imgs']
        # root_path = input('Root path (e.g., r"D:/Pictures/2019_12_03_capture"): ')
        root_path = user_inputs['root_path']
        work_path = user_inputs['work_path']

        # if -1: obtain intrinsics for all cameras
        # if c > 0: copy&paste intrinsic from given camera index, c.
        standard_intrinsics_cam = 0
        Calibrator.compute_initial_camera_parameters(root_path, work_path, standard_intrinsics_cam, configs)
    elif run == 7:
        root_path = user_inputs['root_path']
        work_path = user_inputs['work_path']
        Calibrator.compute_initial_worldpoints_using_PnP(root_path, work_path, configs)
    elif run == 8:
        # root_path = input('Root path (e.g., r"D:/Pictures/2019_12_03_capture"): ')
        work_path = user_inputs['work_path']
        Renderer.render_camera_scene(work_path, configs)
    elif run == 9:
        root_path = user_inputs['root_path']
        Generator.generate_bund_adj_input(root_path, configs)
    elif run == 10:
        root_path = user_inputs['root_path']
        work_path = user_inputs['work_path']
        Generator.generate_bund_adj_initial_values_16cams(root_path, work_path, configs)
    elif run == 11:
        options = {'exclude_outliers': True, 'center_region': False, 'frame_range': (0, 4400)}
        root_path = user_inputs['root_path']
        work_path = user_inputs['work_path']
        Generator.generate_chb_image_points_input(root_path, work_path, configs, options)
    elif run == 12:
        # after bundle adjustment
        root_path = user_inputs['root_path']
        work_path = user_inputs['work_path']
        Generator.generate_cam_params_from_bund_adj(root_path, work_path, configs)
    elif run == 13:
        work_path = user_inputs['work_path']
        cam_param_path = work_path + r'\BundleAdjustment\output\bundle_adjustment_6dof\bundleadjustment_output.txt'
        Generator.export_cam_params_to_txt(work_path, configs, cam_param_path)
    elif run == 14:
        work_path = user_inputs['work_path']
        root_path = user_inputs['root_path']
        root_root_path = '\\'.join(root_path.split('\\')[0:-1])
        input_path = work_path + r'\\Triangulation\input\cam_params.json'
        output_path = root_root_path + r'\\CameraParameters'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        # copy json to output_path
        out_shared = output_path + '\\cam_params.json'
        out_local = work_path + '\\FinalCamParams\cam_params.json'
        shutil.copyfile(input_path, out_shared)
        shutil.copyfile(input_path, out_local)

        print('1. Saved to: {}'.format(out_local))
        print('2. Saved to: {}'.format(out_shared))
        output_path += '\\cam_params.txt'
        Generator.cam_params_json_to_txt(input_path, output_path, configs)
    else:
        print("[ERROR] invalid input integer! {}".format(run))


def main_one_stop():
    # merge detection results into one .txt
    root_path = r'\\DATACHEWER\shareZ\2020_01_01_KateyCapture\Converted'
    work_path = r'D:\CalibrationData\CameraCalibration\2020_01_01_KateyCapture'
    if not os.path.exists(work_path):
        os.mkdir(work_path)

    configs = Configs()
    configs.num_cams = 16
    configs.frame_range = (0, 3060)
    configs.cam_range = (0, 15)

    Calibrator.detect_opencv_corners_multiprocess(root_path, work_path, configs)

    # run only if a folder name has to be changed
    # Parser.change_corner_folder_name(root_path, 'Corners_20200102')

    # for visualizing detected corners
    # Calibrator.save_opencv_corners_on_images(root_path, work_path, configs)

    """
    detection results
    """
    Parser.merge_detection_results(root_path, work_path, configs)

    # """
    # outliers
    # """
    output_path = work_path + '\\CroppedCorners_2'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    Calibrator.save_ranked_corner_crops(root_path, output_path, configs)
    # Calibrator.determine_corner_outliers(root_path, output_path)
    score_thres = 500
    Calibrator.generate_outliers_txt(root_path, work_path, score_thres, configs, export=True)


    """
    # initial camera parameters
    # """
    first_cam_setup = False
    if first_cam_setup:
        # if -1: obtain intrinsics for all cameras
        # if c >= 0: copy&paste intrinsic from given camera index, c.
        configs.num_stereo_imgs = 10
        configs.center_cam_idx = 0
        configs.center_img_name = '0500'
        configs.num_single_calib_imgs = 80
        standard_intrinsics_cam = 0
        Calibrator.compute_initial_camera_parameters(root_path, work_path, standard_intrinsics_cam, configs)
    else:
        from_path = r'D:\CalibrationData\CameraCalibration\191205_16Cams\SingleCalibrations'
        to_path = work_path + '\\SingleCalibrations'
        if not os.path.exists(to_path):
            os.mkdir(to_path)
        Calibrator.copy_intial_camera_parameters(from_path, to_path)

    """
    Bundle adjustment input
    """
    Generator.generate_bund_adj_input(root_path, work_path, configs)
    Generator.generate_bund_adj_initial_values_16cams(root_path, work_path, configs)
    options = {'exclude_outliers': True, 'center_region': False}
    Generator.generate_chb_image_points_input(root_path, work_path, configs, options)


if __name__ == '__main__':
    # run = input('Function index to run: ')
    # main_calibration(int(run))
    main_one_stop()

    # run it after bundle adjustment, before triangulation
    # main_calibration(12)

    # export camera parameters
    # main_calibration(13)
    # main_calibration(14)

    # render 3d scene
    # main_calibration(8)

    """
    # run it once per capture
    user_inputs = load_user_inputs('user_inputs.txt')
    work_path = user_inputs['work_path']
    root_path = user_inputs['root_path']

    configs = Configs()
    configs.num_cams = user_inputs['num_cams']
    Generator.generate_detection_results(root_path, work_path, configs)
    """
