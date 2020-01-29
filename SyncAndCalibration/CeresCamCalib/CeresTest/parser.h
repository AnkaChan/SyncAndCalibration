#pragma once
#define  _CRT_SECURE_NO_WARNINGS
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>

#include "../include/rapidjson/rapidjson.h"
#include "../include/rapidjson/document.h"
#include "../include/rapidjson/filereadstream.h"

using std::cout;
using std::endl;
struct Camera {
	int index;
	double rvec[3];
	double tvec[3];
	double fx, fy, cx, cy;
	double k[6];
	double p[6];
	Camera(int idx) {
		index = idx;
	}
};
struct Configs {
	enum RadialModel {
		polynomial, rational
	};
	enum Loss {
		None, Huber
	};
	int num_cams;
	int num_frames;
	int num_cam_params;
	int chb_num_rows;
	int chb_num_cols;
	int chb_num_corners;
	int chb_sqr_size;

	// c++ defined configs
	RadialModel radial_model = polynomial;
	bool dist_regularization = true;
	int max_k = 2;
	int max_p = 2;
	Loss loss = None;
	std::string input_image_pts_path;
	std::string input_initial_params_path;
	int ExportConfigs(const char* path) {
		std::ofstream output;
		output.open(path);
		output << "input_image_points=" << input_image_pts_path << "\n";
		output << "input_initial_params=" << input_initial_params_path << "\n";
		output << "k=" << max_k << "\n";
		output << "p=" << max_p << "\n";
		output << "radial_model=" << radial_model << "\n";
		output << "loss=" << loss << "\n";
		output << "distortion_reg=" << dist_regularization;
		output.close();
		return 0;
	}
};

struct BundleAdjParameters {
		std::vector<Camera> cameras_;

		// initial values
		double *cam_params_ = NULL;
		double *world_points_; // for non-6dof
		double *chb_rvecs_; // for 6dof
		double *chb_tvecs_; // for 6dof
		double *chb_points_; // z==0, flat on ground

		BundleAdjParameters() {
			chb_points_ = new double[88 * 3];
			double s = 60;
			for (int r = 0; r < 8; r++) {
				for (int c = 0; c < 11; c++) {
					int i = (r * 11 + c) * 3;
					double x = -c * s;
					double y = r * s;
					double z = 0;
					chb_points_[i + 0] = x;
					chb_points_[i + 1] = y;
					chb_points_[i + 2] = z;
				}
			}
		}
		~BundleAdjParameters() {
			delete[] world_points_;
			delete[] chb_rvecs_;
			delete[] chb_tvecs_;
			delete[] cam_params_;
			delete[] chb_points_;
		}

		int SetCamParams(const int num_cams, const int num_cam_params, const rapidjson::Value& cam_params) {
			if (cameras_.size() > 0) {
				cout << "Deleting previously loaded " << cameras_.size() << " cameras" << endl;
				cameras_.clear();
			}
			if (cam_params_ == NULL) {
				cam_params_ = new double[num_cams * num_cam_params];
			}
			for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
				Camera cam(cam_idx);

				const rapidjson::Value& cam_param = cam_params[std::to_string(cam_idx).c_str()];
				const rapidjson::Value& rvec = cam_param["rvec"];
				for (rapidjson::SizeType i = 0; i < rvec.Size(); i++) {
					cam.rvec[i] = rvec[i].GetDouble();
				}
				const rapidjson::Value& tvec = cam_param["tvec"];
				for (rapidjson::SizeType i = 0; i < tvec.Size(); i++) {
					cam.tvec[i] = tvec[i].GetDouble();
				}
				cam.fx = cam_param["fx"].GetDouble();
				cam.fy = cam_param["fy"].GetDouble();
				cam.cx = cam_param["cx"].GetDouble();
				cam.cy = cam_param["cy"].GetDouble();

				cam.k[0] = cam_param["k1"].GetDouble();
				cam.k[1] = cam_param["k2"].GetDouble();
				cam.k[2] = cam_param["k3"].GetDouble();
				cam.k[3] = cam_param["k4"].GetDouble();
				cam.k[4] = cam_param["k5"].GetDouble();
				cam.k[5] = cam_param["k6"].GetDouble();

				cam.p[0] = cam_param["p1"].GetDouble();
				cam.p[1] = cam_param["p2"].GetDouble();
				cam.p[2] = cam_param["p3"].GetDouble();
				cam.p[3] = cam_param["p4"].GetDouble();
				cam.p[4] = cam_param["p5"].GetDouble();
				cam.p[5] = cam_param["p6"].GetDouble();

				int i = cam_idx * num_cam_params;
				cam_params_[i + 0] = cam.rvec[0];
				cam_params_[i + 1] = cam.rvec[1];
				cam_params_[i + 2] = cam.rvec[2];
				cam_params_[i + 3] = cam.tvec[0];
				cam_params_[i + 4] = cam.tvec[1];
				cam_params_[i + 5] = cam.tvec[2];
				cam_params_[i + 6] = cam.fx;
				cam_params_[i + 7] = cam.fy;
				cam_params_[i + 8] = cam.cx;
				cam_params_[i + 9] = cam.cy;
				cam_params_[i + 10] = cam.k[0];
				cam_params_[i + 11] = cam.k[1];
				cam_params_[i + 12] = cam.k[2];
				cam_params_[i + 13] = cam.k[3];
				cam_params_[i + 14] = cam.k[4];
				cam_params_[i + 15] = cam.k[5];
				cam_params_[i + 16] = cam.p[0];
				cam_params_[i + 17] = cam.p[1];
				cam_params_[i + 18] = cam.p[2];
				cam_params_[i + 19] = cam.p[3];
				cam_params_[i + 20] = cam.p[4];
				cam_params_[i + 21] = cam.p[5];
				cout << "Camera[" << cam_idx << "] | ";
				for (int n = 0; n < 22; n++) {
					cout << cam_params_[i + n] << " ";
				}
				cout << endl;
				cameras_.push_back(cam);
			}
			printf("  %zd cameras loaded.\n", cameras_.size());
			return 0;
		}
		int SetChbParams(const std::vector<std::string> img_names_vec_in, const int num_frames, const int num_corners, const rapidjson::Value& chb_params) {
			if (world_points_ != NULL) {
				cout << "Deleting previously loaded initial chb parameters." << endl;
				delete[] world_points_;
				delete[] chb_rvecs_;
				delete[] chb_tvecs_;
			}
			world_points_ = new double[num_frames * num_corners * 3];
			chb_rvecs_ = new double[num_frames * 3];
			chb_tvecs_ = new double[num_frames * 3];
			for (int frame_idx = 0; frame_idx < img_names_vec_in.size(); frame_idx++) {
				std::string target_img_name = img_names_vec_in[frame_idx];
				const rapidjson::Value& chb_vals = chb_params[target_img_name.c_str()];
				//int frame_idx_curr = chb_vals["frame_idx"].GetInt();

				int i = frame_idx * num_corners * 3;
				const rapidjson::Value& wps_arr = chb_vals["world_pts"];

				for (int p_idx = 0; p_idx < num_corners; p_idx++) {
					const rapidjson::Value& wp = wps_arr[p_idx];
					world_points_[i + p_idx * 3 + 0] = wp[0].GetDouble();
					world_points_[i + p_idx * 3 + 1] = wp[1].GetDouble();
					world_points_[i + p_idx * 3 + 2] = wp[2].GetDouble();
				}

				const rapidjson::Value& rvec = chb_vals["rvec"];
				i = frame_idx * 3;
				chb_rvecs_[i + 0] = rvec[0].GetDouble();
				chb_rvecs_[i + 1] = rvec[1].GetDouble();
				chb_rvecs_[i + 2] = rvec[2].GetDouble();

				const rapidjson::Value& tvec = chb_vals["tvec"];
				chb_tvecs_[i + 0] = tvec[0].GetDouble();
				chb_tvecs_[i + 1] = tvec[1].GetDouble();
				chb_tvecs_[i + 2] = tvec[2].GetDouble();
			}

			printf("  %d checkerboards set.\n", num_frames);
			return 0;
		}
	};
class Parser {
public:
	Parser();

	int LoadImagePoints(const char* path, Configs &configs_out, bool *&detected_out, double *&img_pts_out, std::vector<std::string> &img_names_vec_out, const int sanity_num_corners);
	int LoadInitialParameters(const char* path, const std::vector<std::string> &img_names_vec_in, Configs &configs_out, BundleAdjParameters& params);

	static int LoadInitialWorldPoints(const char* path, const int num_cams, double* &wps_out);
};