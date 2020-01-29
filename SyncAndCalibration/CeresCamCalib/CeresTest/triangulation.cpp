#include "triangulation.h"
#include "parser.h"
#include <iostream>
#include <fstream>

#define NUM_CORNERS 88


class TriangulationProblem {
public:

	struct ReprojectionError {
		double u_mea, v_mea;
		Camera *cam;
		Configs *configs;

		~ReprojectionError() {

		}
		ReprojectionError(const double* uv, Camera *c, Configs *conf) {
			u_mea = uv[0];
			v_mea = uv[1];
			cam = c;
			configs = conf;
		}

		template<typename T>
		bool operator()(const T* const world_point, T* residuals) const {
			// extract camera parameters
			const T ang_axis[3] = { T(cam->rvec[0]), T(cam->rvec[1]), T(cam->rvec[2]) };
			const T trans[3] = { T(cam->tvec[0]), T(cam->tvec[1]), T(cam->tvec[2]) };
			const T f[2] = { T(cam->fx), T(cam->fy) };
			const T c[2] = { T(cam->cx), T(cam->cy) };

			T k[6] = { T(0) };
			T p[6] = { T(0) };
			for (int i = 0; i < 6; i++) {
				k[i] = T(cam->k[i]);
				p[i] = T(cam->p[i]);
			}

			// world_points
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
			T u_pred = f[0] * u + c[0];
			T v_pred = f[1] * v + c[1];

			// error
			T du = u_pred - u_mea;
			T dv = v_pred - v_mea;

			// output
			residuals[0] = du;
			residuals[1] = dv;
			return true;
		}

		static ceres::CostFunction* Create(const double* uv, Camera *cp_in, Configs *conf) {
			ReprojectionError *re = new TriangulationProblem::ReprojectionError(uv, cp_in, conf);
			return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3>(re));
		}
	};

	std::map<std::string, int> imgname_2_frameidx;
	std::map<int, std::string> frameidx_2_imgname;

	~TriangulationProblem() {
	}
	Configs* configs() {
		return &configs_;
	}
	bool detected(const int frame_idx, const int cam_idx) {
		return detected_[frame_idx * configs_.num_cams + cam_idx];
	}
	double* image_points(const int frame_idx, const int cam_idx, const int p_idx) {
		return image_points_ + frame_idx * configs_.num_cams * configs_.chb_num_corners * 2 + cam_idx * configs_.chb_num_corners * 2 + p_idx * 2;
	}
	Camera* camera(const int cam_idx) {
		return &cams_[cam_idx];
	}
	int LoadImagePoints(const char* path) {
		printf("Load json: %s\n", path);
		FILE* fp = fopen(path, "rb"); // non-Windows use "r"

		char readBuffer[65536]; // max buffer size
		rapidjson::FileReadStream fs(fp, readBuffer, sizeof(readBuffer));

		// parse
		rapidjson::Document doc;
		doc.ParseStream(fs);
		assert(doc.HasMember("configs"));
		assert(doc.HasMember("frames"));


		// configs
		const rapidjson::Value& configs = doc["configs"];
		const rapidjson::Value& chb = configs["chb"];
		configs_.num_cams = configs["num_cams"].GetInt();
		configs_.num_frames = configs["num_frames"].GetInt();

		configs_.chb_num_corners = chb["num_corners"].GetInt();
		configs_.chb_num_rows = chb["num_rows"].GetInt();
		configs_.chb_num_cols = chb["num_cols"].GetInt();
		configs_.chb_sqr_size = chb["chb_sqr_size"].GetInt();

		// sanity check
		if (configs_.chb_num_corners != NUM_CORNERS) {
			std::cerr << "[ERROR] some mismatch!" << endl;
			std::getchar();
			return 1;
		}

		detected_ = new bool[configs_.num_frames * configs_.num_cams];
		image_points_ = new double[configs_.num_frames * configs_.num_cams * configs_.chb_num_corners * 2]{ 0 };
		const rapidjson::Value& frames = doc["frames"];
		int num_img_pts = 0;
		for (rapidjson::SizeType frame_idx = 0; frame_idx < frames.Size(); frame_idx++) {
			const rapidjson::Value& frame = frames[frame_idx];
			const rapidjson::Value& img_pts = frame["img_pts"];
			std::string img_name = frame["img_name"].GetString();
			imgname_2_frameidx[img_name] = frame_idx;
			frameidx_2_imgname[frame_idx] = img_name;
			// image points
			for (int cam_idx = 0; cam_idx < configs_.num_cams; cam_idx++) {
				std::string cam_idx_str = std::to_string(cam_idx);
				if (img_pts.HasMember(cam_idx_str.c_str())) {
					detected_[frame_idx * configs_.num_cams + cam_idx] = true;
					const rapidjson::Value &pts = img_pts[cam_idx_str.c_str()];
					for (int p_idx = 0; p_idx < configs_.chb_num_corners; p_idx++) {
						const rapidjson::Value &pt = pts[p_idx];
						double u = pt[0].GetDouble();
						double v = pt[1].GetDouble();
						int idx = frame_idx * configs_.num_cams * configs_.chb_num_corners * 2 + cam_idx * configs_.chb_num_corners * 2 + p_idx * 2;
						image_points_[idx + 0] = u;
						image_points_[idx + 1] = v;
						num_img_pts += 1;
					}
				}
				else
				{
					detected_[frame_idx * configs_.num_cams + cam_idx] = false;
				}
			}
		}
		printf("  %d frames, %d image points loaded.\n", frames.Size(), num_img_pts);
		fclose(fp);
		return 0;
	}

	int LoadCameraParams(const char* path) {
		printf("Load json: %s\n", path);
		FILE* fp = fopen(path, "rb"); // non-Windows use "r"

		char readBuffer[65536]; // max buffer size
		rapidjson::FileReadStream fs(fp, readBuffer, sizeof(readBuffer));

		// parse
		rapidjson::Document doc;
		doc.ParseStream(fs);
		assert(doc.HasMember("configs"));
		assert(doc.HasMember("cam_params"));

		const rapidjson::Value& configs = doc["configs"];
		configs_.num_cams = configs["num_cams"].GetInt();

		
		const rapidjson::Value& cams = doc["cam_params"];
		for (int cam_idx = 0; cam_idx < configs_.num_cams; cam_idx++) {
			const rapidjson::Value& params = cams[std::to_string(cam_idx).c_str()];
			Camera cam(cam_idx);

			const rapidjson::Value& rvec = params["rvec"];
			cam.rvec[0] = rvec[0].GetDouble();
			cam.rvec[1] = rvec[1].GetDouble();
			cam.rvec[2] = rvec[2].GetDouble();
			const rapidjson::Value& tvec = params["tvec"];
			cam.tvec[0] = tvec[0].GetDouble();
			cam.tvec[1] = tvec[1].GetDouble();
			cam.tvec[2] = tvec[2].GetDouble();
			cam.fx = params["fx"].GetDouble();
			cam.fy = params["fy"].GetDouble();
			cam.cx = params["cx"].GetDouble();
			cam.cy = params["cy"].GetDouble();
			cam.k[0] = params["k1"].GetDouble();
			cam.k[1] = params["k2"].GetDouble();
			cam.k[2] = params["k3"].GetDouble();
			cam.k[3] = params["k4"].GetDouble();
			cam.k[4] = params["k5"].GetDouble();
			cam.k[5] = params["k6"].GetDouble();

			cam.p[0] = params["p1"].GetDouble();
			cam.p[1] = params["p2"].GetDouble();
			cam.p[2] = params["p3"].GetDouble();
			cam.p[3] = params["p4"].GetDouble();
			cam.p[4] = params["p5"].GetDouble();
			cam.p[5] = params["p6"].GetDouble();

			printf("Camera[%d]\n", cam_idx);
			printf("  (%.4f, %.4f, %.4f), (%.4f, %.4f, %.4f), (%.4f, %.4f), (%.4f, %.4f)\n", cam.rvec[0], cam.rvec[1], cam.rvec[2], cam.tvec[0], cam.tvec[1], cam.tvec[2], cam.fx, cam.fy, cam.cx, cam.cy);
			printf("  [k] %.4f, %.4f, %.4f, %.4f, %.4f, %.4f | p[p] %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n", cam.k[0], cam.k[1], cam.k[2], cam.k[3], cam.k[4], cam.k[5], cam.p[0], cam.p[1], cam.p[2], cam.p[3], cam.p[4], cam.p[5]);
			cams_.push_back(cam);
		}
		printf("  %zd cameras loaded.\n", cams_.size());

		return 0;

	}
private:
	std::vector<Camera> cams_;
	Configs configs_;
	bool *detected_;
	double *image_points_;
};
Triangulation::Triangulation() {
}
int Triangulation::DoWork(std::string work_path) {


	google::InitGoogleLogging("HJP");

	TriangulationProblem prob;
	std::string input_img_pts = work_path + std::string("/BundleAdjustment/input/image_points.json");
	prob.LoadImagePoints(input_img_pts.c_str());
	std::string input_cam_params = work_path + std::string("/Triangulation/input/cam_params.json");
	prob.LoadCameraParams(input_cam_params.c_str());
	Configs *configs = prob.configs();
	configs->max_k = 6;
	configs->max_p = 2;
	configs->loss = Configs::Loss::None;
	configs->radial_model = Configs::RadialModel::rational;
	std::string output_path = work_path + std::string("/Triangulation/output/triangulation_output.txt");
	std::ofstream output_txt;
	output_txt.open(output_path.c_str());

	output_txt << configs->num_frames << " " << configs->num_cams << " " << input_img_pts << "\n";
	for (int cam_idx = 0; cam_idx < configs->num_cams; cam_idx++) {
		Camera *c = prob.camera(cam_idx);
		output_txt << c->rvec[0] << " " << c->rvec[1] << " " << c->rvec[2] << " " << c->tvec[0] << " " << c->tvec[1] << " " << c->tvec[2] << " ";
		output_txt << c->fx << " " << c->fy << " " << c->cx << " " << c->cy << " ";
		output_txt << c->k[0] << " " << c->k[1] << " " << c->k[2] << " " << c->k[3] << " " << c->k[4] << " " << c->k[5] << " ";
		output_txt << c->p[0] << " " << c->p[1] << " " << c->p[2] << " " << c->p[3] << " " << c->p[4] << " " << c->p[5] << "\n";
	}
	
	int num_frames = prob.configs()->num_frames;
	int num_corners = prob.configs()->chb_num_corners;
	int num_cams = prob.configs()->num_cams;


	// initial world points
	std::string input_inital_points = work_path + std::string("/BundleAdjustment/output/bundle_adjustment_6dof/bundleadjustment_output.txt");
	double* wps;
	int res = Parser::LoadInitialWorldPoints(input_inital_points.c_str(), num_cams, wps);

	double cost_sum_final = 0;
	double cost_sum_initial = 0;
	double *world_point_out = new double[configs->chb_num_corners * 3]{ 0 };
	double *final_costs = new double[configs->chb_num_corners]{ 0 };
	for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
		int remainder = frame_idx % 100;
		if (remainder == 0) {
			printf("[%d]\n", frame_idx);
		}

		for (int i = 0; i < configs->chb_num_corners * 3; i++) {
			world_point_out[i] = 0;
		}
		for (int i = 0; i < configs->chb_num_corners; i++) {
			final_costs[i] = 0;
		}

		for (int p_idx = 0; p_idx < num_corners; p_idx++) {
			int cam_used = 0;
			for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
				if (prob.detected(frame_idx, cam_idx)) {
					cam_used += 1;
				}
			}

			if (cam_used >= 2) {
				ceres::Problem ceres_prob;
				ceres::LossFunction* loss = NULL;

				/*
				int rand_int = rand() % 100 + 1;
				double rand_x = double(rand_int);
				double rand_y = double(rand_int);
				double rand_z = double(rand_int);
				double world_point[3] = { rand_x, rand_y, rand_z };
				*/
				double world_point[3] = { wps[frame_idx*num_corners * 3 + p_idx * 3], wps[frame_idx*num_corners * 3 + p_idx * 3 + 1] ,wps[frame_idx*num_corners * 3 + p_idx * 3 + 2] };
				for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
					if (prob.detected(frame_idx, cam_idx)) {
						double *uv = prob.image_points(frame_idx, cam_idx, p_idx);
						Camera *cam = prob.camera(cam_idx);
						ceres::CostFunction* cost_function = TriangulationProblem::ReprojectionError::Create(uv, cam, configs);
						ceres_prob.AddResidualBlock(cost_function, loss, world_point);
					}
				}

				ceres::Solver::Options options;
				options.linear_solver_type = ceres::DENSE_SCHUR;
				options.minimizer_progress_to_stdout = 0;

				options.num_linear_solver_threads = 1;
				options.num_threads = 1;
				options.function_tolerance = 0.00000000001;
				options.gradient_tolerance = 0.00000000001;
				ceres::Solver::Summary summary;
				ceres::Solve(options, &ceres_prob, &summary);
				if (summary.termination_type != ceres::TerminationType::CONVERGENCE) {
					cout << summary.FullReport() << endl;
				}
				cost_sum_final += summary.final_cost;
				final_costs[p_idx] = summary.final_cost;
				cost_sum_initial += summary.initial_cost;
				world_point_out[p_idx * 3 + 0] = world_point[0];
				world_point_out[p_idx * 3 + 1] = world_point[1];
				world_point_out[p_idx * 3 + 2] = world_point[2];
			}
		} // p_idx ends

		int num_detected = 0;
		for (int cam_idx = 0; cam_idx < configs->num_cams; cam_idx++) {
			num_detected += int(prob.detected(frame_idx, cam_idx));
		}
		if (num_detected >= 2) {
			output_txt << prob.frameidx_2_imgname[frame_idx] << " ";

			for (int cam_idx = 0; cam_idx < configs->num_cams; cam_idx++) {
				output_txt << int(prob.detected(frame_idx, cam_idx)) << " ";
			}
			output_txt << configs->chb_num_corners << " ";
			for (int pi = 0; pi < configs->chb_num_corners; pi++) {
				output_txt << world_point_out[pi * 3] << " " << world_point_out[pi * 3 + 1] << " " << world_point_out[pi * 3 + 2] << " ";
			}
			for (int pi = 0; pi < configs->chb_num_corners; pi++) {
				output_txt << final_costs[pi] << " ";
			}
			output_txt << "\n";
		}
	}

	cout << "Complete! " << cost_sum_initial << " -> " << cost_sum_final << endl;
	cout << "Saved to: " << output_path << endl;
	delete[] world_point_out, final_costs;
	return 0;
}