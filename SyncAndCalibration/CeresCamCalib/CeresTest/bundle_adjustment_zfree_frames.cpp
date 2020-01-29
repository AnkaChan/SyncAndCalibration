#define _USE_MATH_DEFINES
#include <math.h>
#include "bundle_adjustment_zfree_frames.h"
#include "parser.h"
#define NUM_CAMPARAMS 22 // 3 + 3 + 2 + 2 + 6 + 6
#define NUM_CORNERS 88

class LossFunction {
public:
	virtual void Evaluate(double s, double out[3]) const = 0;
};

class BundleAdjustmentProblemZFreeFrames {
public:
	struct ReprojectionError {
		static double chb_points[NUM_CORNERS * 3];
		double *image_points_mea;
		Configs *configs;
		~ReprojectionError() {
			delete[] image_points_mea;
		}

		ReprojectionError(Configs *cf, const int num_pts, double *observed_points) {
			configs = cf;
			image_points_mea = new double[num_pts * 2];

			// observed_points -> 2*88 size
			for (int i = 0; i < num_pts; i++) {
				image_points_mea[i * 2 + 0] = observed_points[i * 2 + 0];
				image_points_mea[i * 2 + 1] = observed_points[i * 2 + 1];
			}
		}

		static ceres::CostFunction* Create(Configs *cf, double *observed_points) {
			static const int kNumCamParam = NUM_CAMPARAMS;
			static const int kChbRvecDim = 3;
			static const int kChbTvecDim = 3;

			ReprojectionError *re = new BundleAdjustmentProblemZFreeFrames::ReprojectionError(cf, 88, observed_points);
			return (new ceres::AutoDiffCostFunction<ReprojectionError, (88 * 2) + (6 + 2), kNumCamParam, kChbRvecDim, kChbTvecDim, 88>(re));
		}

		template<typename T>
		bool operator()(const T* const camera, const T* const chbR, const T* const chbT, const T* const chb_points_zs, T* residuals) const {
			// extract camera parameters
			const T ang_axis[3] = { camera[0], camera[1], camera[2] };
			const T trans[3] = { camera[3], camera[4], camera[5] };
			const T f[2] = { camera[6], camera[7] };
			const T c[2] = { camera[8], camera[9] };

			T k[6] = { T(0) };
			T p[6] = { T(0) };
			if (configs->radial_model == Configs::RadialModel::polynomial) {
				switch (configs->max_k) {
				case 6:
					k[5] = camera[15];
				case 5:
					k[4] = camera[14];
				case 4:
					k[3] = camera[13];
				case 3:
					k[2] = camera[12];
				case 2:
					k[1] = camera[11];
				case 1:
					k[0] = camera[10];
				}
			}

			else if (configs->radial_model == Configs::RadialModel::rational) {
				k[0] = camera[10];
				k[1] = camera[11];
				k[2] = camera[12];
				k[3] = camera[13];
				k[4] = camera[14];
				k[5] = camera[15];
			}

			switch (configs->max_p) {
			case 6:
				p[5] = camera[21];
			case 5:
				p[4] = camera[20];
			case 4:
				p[3] = camera[19];
			case 3:
				p[2] = camera[18];
			case 2:
				p[1] = camera[17];
			case 1:
				p[0] = camera[16];
			}

			// chb parameters
			const T chb_ang_axis[3] = { chbR[0], chbR[1], chbR[2] };
			const T chb_trans[3] = { chbT[0], chbT[1], chbT[2] };

			// world_points
			for (int i = 0; i < configs->chb_num_corners; i++) {
				// update checkerboard points
				const T chb_point[3] = { T(chb_points[i * 3]), T(chb_points[i * 3 + 1]), T(chb_points_zs[i]) };
				T world_point[3];
				ceres::AngleAxisRotatePoint(chb_ang_axis, chb_point, world_point);

				// translation
				world_point[0] += chb_trans[0];
				world_point[1] += chb_trans[1];
				world_point[2] += chb_trans[2];

				T cam_point[3];
				// angle-axis rotation
				ceres::AngleAxisRotatePoint(ang_axis, world_point, cam_point);

				// translation
				cam_point[0] += trans[0];
				cam_point[1] += trans[1];
				cam_point[2] += trans[2];

				// center of distortion
				T xp = cam_point[0] / cam_point[2];
				T yp = cam_point[1] / cam_point[2];

				T radial_dist = T(0);
				T r2 = T(0);
				if (configs->radial_model == Configs::RadialModel::polynomial) {
					// radial distortions
					r2 = xp * xp + yp * yp;
					T r2_radial = T(1.0);
					radial_dist = T(1.0);
					for (int kn = 0; kn < configs->max_k; kn++) {
						r2_radial *= r2;
						radial_dist += k[kn] * r2_radial;
					}
				}
				else if (configs->radial_model == Configs::RadialModel::rational) {
					// radial distortions
					r2 = xp * xp + yp * yp;
					T r4 = r2 * r2;
					T r6 = r2 * r2 * r2;
					radial_dist = (1.0 + k[0] * r2 + k[1] * r4 + k[2] * r6) / (1.0 + k[3] * r2 + k[4] * r4 + k[5] * r6);
				}

				// tangential distortions
				T tan_post = T(1.0);
				T r2_tangential = T(1.0);
				for (int pn = 2; pn < configs->max_p; pn++) {
					r2_tangential *= r2;
					tan_post += p[pn] * r2_tangential;
				}

				T tangential_dist_x = (p[1] * (r2 + 2.0 * xp*xp) + 2.0 * p[0] * xp*yp) * tan_post;
				T tangential_dist_y = (p[0] * (r2 + 2.0 * yp*yp) + 2.0 * p[1] * xp*yp) * tan_post;


				T u = xp * radial_dist + tangential_dist_x;
				T v = yp * radial_dist + tangential_dist_y;

				// projected point position
				T predicted_x = f[0] * u + c[0];
				T predicted_y = f[1] * v + c[1];

				// error
				T dx = predicted_x - image_points_mea[i * 2 + 0];
				T dy = predicted_y - image_points_mea[i * 2 + 1];

				// output
				residuals[2 * i + 0] = dx;
				residuals[2 * i + 1] = dy;
			}

			int idx = 2 * configs->chb_num_corners;
			static const double lambda_k[6] = { 1, 1, 1, 1, 1, 1 };
			static const double lambda_p[2] = { 1, 1 };
			if (configs->dist_regularization) {
				residuals[idx + 0] = lambda_k[0] * k[0];
				residuals[idx + 1] = lambda_k[1] * k[1];
				residuals[idx + 2] = lambda_k[2] * k[2];
				residuals[idx + 3] = lambda_k[3] * k[3];
				residuals[idx + 4] = lambda_k[4] * k[4];
				residuals[idx + 5] = lambda_k[5] * k[5];
				residuals[idx + 6] = lambda_p[0] * p[0];
				residuals[idx + 7] = lambda_p[1] * p[1];
			}
			else {
				residuals[idx + 0] = T(0);
				residuals[idx + 1] = T(0);
				residuals[idx + 2] = T(0);
				residuals[idx + 3] = T(0);
				residuals[idx + 4] = T(0);
				residuals[idx + 5] = T(0);
				residuals[idx + 6] = T(0);
				residuals[idx + 7] = T(0);
			}

			return true;
		}
	};

private:
	BundleAdjParameters params;
	Configs configs_;
	bool *detected_;
	double *image_points_;
	std::vector<std::string> img_names_vec;
public:
	Configs* GetConfigs() {
		return &configs_;
	}
	bool detected(const int frame_idx, const int cam_idx) {
		return detected_[frame_idx * configs_.num_cams + cam_idx];
	}
	double* image_points(const int frame_idx, const int cam_idx) {
		return image_points_ + frame_idx * configs_.num_cams*configs_.chb_num_corners * 2 + cam_idx * configs_.chb_num_corners * 2;
	}
	~BundleAdjustmentProblemZFreeFrames() {
		delete[] detected_;
		delete[] image_points_;
	}
	double* chb_rvec(const int frame_idx) {
		return params.chb_rvecs_ + frame_idx * 3;
	}
	double* chb_tvec(const int frame_idx) {
		return params.chb_tvecs_ + frame_idx * 3;
	}

	double *world_points(const int frame_idx, const int p_idx) {
		return params.world_points_ + frame_idx * configs_.chb_num_corners * 3 + p_idx * 3;
	}
	double* cam_params(const int cam_idx) {
		return params.cam_params_ + cam_idx * configs_.num_cam_params;
	}

	std::string image_name(const int frame_idx) {
		return img_names_vec[frame_idx];
	}
	int Load(const char* img_pts_json, const char* initial_params_json) {
		Parser parser;
		parser.LoadImagePoints(img_pts_json, configs_, detected_, image_points_, img_names_vec, NUM_CORNERS);
		parser.LoadInitialParameters(initial_params_json, img_names_vec, configs_, params);
		return 0;
	}
};


double BundleAdjustmentProblemZFreeFrames::ReprojectionError::chb_points[NUM_CORNERS * 3] = { -0, 0, 0,
-60, 0, 0,
-120, 0, 0,
-180, 0, 0,
-240, 0, 0,
-300, 0, 0,
-360, 0, 0,
-420, 0, 0,
-480, 0, 0,
-540, 0, 0,
-600, 0, 0,
-0, 60, 0,
-60, 60, 0,
-120, 60, 0,
-180, 60, 0,
-240, 60, 0,
-300, 60, 0,
-360, 60, 0,
-420, 60, 0,
-480, 60, 0,
-540, 60, 0,
-600, 60, 0,
-0, 120, 0,
-60, 120, 0,
-120, 120, 0,
-180, 120, 0,
-240, 120, 0,
-300, 120, 0,
-360, 120, 0,
-420, 120, 0,
-480, 120, 0,
-540, 120, 0,
-600, 120, 0,
-0, 180, 0,
-60, 180, 0,
-120, 180, 0,
-180, 180, 0,
-240, 180, 0,
-300, 180, 0,
-360, 180, 0,
-420, 180, 0,
-480, 180, 0,
-540, 180, 0,
-600, 180, 0,
-0, 240, 0,
-60, 240, 0,
-120, 240, 0,
-180, 240, 0,
-240, 240, 0,
-300, 240, 0,
-360, 240, 0,
-420, 240, 0,
-480, 240, 0,
-540, 240, 0,
-600, 240, 0,
-0, 300, 0,
-60, 300, 0,
-120, 300, 0,
-180, 300, 0,
-240, 300, 0,
-300, 300, 0,
-360, 300, 0,
-420, 300, 0,
-480, 300, 0,
-540, 300, 0,
-600, 300, 0,
-0, 360, 0,
-60, 360, 0,
-120, 360, 0,
-180, 360, 0,
-240, 360, 0,
-300, 360, 0,
-360, 360, 0,
-420, 360, 0,
-480, 360, 0,
-540, 360, 0,
-600, 360, 0,
-0, 420, 0,
-60, 420, 0,
-120, 420, 0,
-180, 420, 0,
-240, 420, 0,
-300, 420, 0,
-360, 420, 0,
-420, 420, 0,
-480, 420, 0,
-540, 420, 0,
-600, 420, 0, };

BundleAdjustmentZFreeFrames::BundleAdjustmentZFreeFrames() {
}
int BundleAdjustmentZFreeFrames::DoWork(std::string work_path) {
	google::InitGoogleLogging("HJP");

	// load input
	BundleAdjustmentProblemZFreeFrames prob;
	Configs *configs = prob.GetConfigs();
	configs->input_image_pts_path = "C:/Users/joont/Documents/PycharmProjects/CameraCalibration/data/ImagePoints/chb/ExcludedOutliers_CenterRegion/image_points.json";
	configs->input_initial_params_path = "C:/Users/joont/Documents/PycharmProjects/CameraCalibration/data/BundleAdjustment/input/bund_adj_inital_params.json";
	prob.Load(configs->input_image_pts_path.c_str(), configs->input_initial_params_path.c_str());
	configs->max_k = 6;
	configs->max_p = 2;
	configs->loss = Configs::Loss::None;
	configs->radial_model = Configs::RadialModel::rational;
	configs->dist_regularization = true;
	int num_corners = configs->chb_num_corners;
	int num_frames = configs->num_frames;
	int num_cams = configs->num_cams;
	const char* output_configs_path = "C:/Users/joont/Documents/PycharmProjects/CameraCalibration/data/BundleAdjustment/output/bundle_adjustment_zfree_frames/configs.txt";
	configs->ExportConfigs(output_configs_path);

	// output
	std::ofstream output_txt;
	output_txt.open("C:/Users/joont/Documents/PycharmProjects/CameraCalibration/data/BundleAdjustment/output/bundle_adjustment_zfree_frames/bundleadjustment_output.txt");

	// configure
	ceres::LossFunction* loss = NULL;
	if (configs->loss == Configs::Loss::Huber) {
		// delta: https://en.wikipedia.org/wiki/Huber_loss
		const double delta = 0.5;
		ceres::LossFunction* loss = new ceres::HuberLoss(delta);
	}

	ceres::Problem ceres_prob;
	double *chb_points_z = new double[configs->num_frames * configs->chb_num_corners]{ 0 };
	for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
		int num_detected = 0;
		for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
			if (prob.detected(frame_idx, cam_idx)) {
				num_detected += 1;
			}
		}

		if (num_detected >= 2) {
			for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
				if (prob.detected(frame_idx, cam_idx)) {
					ceres::CostFunction* cost_func = BundleAdjustmentProblemZFreeFrames::ReprojectionError::Create(prob.GetConfigs(), prob.image_points(frame_idx, cam_idx));
					double *chb_rvec = prob.chb_rvec(frame_idx);
					double *chb_tvec = prob.chb_tvec(frame_idx);
					double *cam_params = prob.cam_params(cam_idx);
					double *chb_z = chb_points_z + frame_idx * configs->chb_num_corners;
					ceres_prob.AddResidualBlock(cost_func, loss, cam_params, chb_rvec, chb_tvec, chb_z);
				}
			}
		}
	}


	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = 1;
	options.max_num_iterations = 1000;
	options.num_linear_solver_threads = 2;
	options.num_threads = 2;
	//options.function_tolerance = 0.0000000000000001;
	//options.parameter_tolerance = 0.0000000000000001;
	//options.gradient_tolerance = 0.0000000000000001;
	//options.inner_iteration_tolerance = 0.0000000000000001;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &ceres_prob, &summary);
	cout << summary.FullReport() << endl;
	cout << "Cost: " << summary.initial_cost << " -> " << summary.final_cost << ", iterations: " << summary.iterations.back().iteration << endl;










	// export initial camera parameters
	output_txt << num_frames << "\n";
	double* cp;
	for (int cam_idx = 0; cam_idx < configs->num_cams; cam_idx++) {
		cp = prob.cam_params(cam_idx);
		for (int i = 0; i < configs->num_cam_params; i++) {
			output_txt << cp[i] << " ";
		}
	}
	output_txt << "\n";
	for (int frame_idx = 0; frame_idx < configs->num_frames; frame_idx++) {
		output_txt << frame_idx << " " << prob.image_name(frame_idx) << " ";
		for (int cam_idx = 0; cam_idx < configs->num_cams; cam_idx++) {
			output_txt << prob.detected(frame_idx, cam_idx) << " ";
		}
		for (int world_point_idx = 0; world_point_idx < configs->chb_num_corners; world_point_idx++) {
			output_txt << -7 << " " << -7 << " " << -7 << " ";
		}
		output_txt << "\n";
	}

	double *chb_points = BundleAdjustmentProblemZFreeFrames::ReprojectionError::chb_points;
	// export camera parameters
	for (int cam_idx = 0; cam_idx < configs->num_cams; cam_idx++) {
		cp = prob.cam_params(cam_idx);
		for (int i = 0; i < configs->num_cam_params; i++) {
			output_txt << cp[i] << " ";
		}
	}
	output_txt << "\n";
	for (int frame_idx = 0; frame_idx < configs->num_frames; frame_idx++) {
		int num_detected = 0;
		for (int cam_idx = 0; cam_idx < configs->num_cams; cam_idx++) {
			num_detected += int(prob.detected(frame_idx, cam_idx));
		}

		output_txt << frame_idx << " " << prob.image_name(frame_idx) << " ";
		for (int cam_idx = 0; cam_idx < configs->num_cams; cam_idx++) {
			output_txt << prob.detected(frame_idx, cam_idx) << " ";
		}

		double *chb_rvec = prob.chb_rvec(frame_idx);
		double *chb_tvec = prob.chb_tvec(frame_idx);
		for (int p_idx = 0; p_idx < configs->chb_num_corners; p_idx++) {
			double chb_point[3] = { chb_points[p_idx * 3], chb_points[p_idx * 3 + 1], chb_points_z[frame_idx*configs->chb_num_corners + p_idx] };
			double world_point[3];
			ceres::AngleAxisRotatePoint(chb_rvec, chb_point, world_point);

			// translation
			world_point[0] += chb_tvec[0];
			world_point[1] += chb_tvec[1];
			world_point[2] += chb_tvec[2];
			output_txt << world_point[0] << " " << world_point[1] << " " << world_point[2] << " ";
		}
		output_txt << "\n";
	}
	for (int frame_idx = 0; frame_idx < configs->num_frames; frame_idx++) {
		int num_detected = 0;
		for (int cam_idx = 0; cam_idx < configs->num_cams; cam_idx++) {
			num_detected += int(prob.detected(frame_idx, cam_idx));
		}

		output_txt << frame_idx << " " << prob.image_name(frame_idx) << " ";
		for (int cam_idx = 0; cam_idx < configs->num_cams; cam_idx++) {
			output_txt << prob.detected(frame_idx, cam_idx) << " ";
		}

		output_txt << prob.chb_rvec(frame_idx)[0] << " " << prob.chb_rvec(frame_idx)[1] << " " << prob.chb_rvec(frame_idx)[2] << " " << prob.chb_tvec(frame_idx)[0] << " " << prob.chb_tvec(frame_idx)[1] << " " << prob.chb_tvec(frame_idx)[2] << " ";
		output_txt << "\n";
	}
	for (int frame_idx = 0; frame_idx < configs->num_frames; frame_idx++) {
		for (int p_idx = 0; p_idx < configs->chb_num_corners; p_idx++) {
			output_txt << chb_points[p_idx * 3] << " " << chb_points[p_idx * 3 + 1] << " " << chb_points_z[frame_idx * configs->chb_num_corners + p_idx] << " ";
		}
	}
	output_txt << "\n";

	output_txt << summary.initial_cost << " ";
	output_txt << summary.final_cost << " ";
	output_txt.close();


	printf("\n>> Complete\n");
	std::getchar();
	return 0;
}
