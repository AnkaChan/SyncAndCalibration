#include "triangulation_pairwise.h"
#include "parser.h"
#include <iostream>
#include <fstream>
#include <map>

#include <chrono>
#include <ctime>  
#include <time.h>
#include <errno.h>

#define NUM_CORNERS 88

const std::string currentDateTime() {
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);
	// Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
	// for more information about date/time format
	strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

	return buf;
}

std::vector<std::string> LoadImageNames(const char* path) {
	printf("Load txt: %s\n", path);
	std::vector<std::string> img_names;

	std::ifstream file(path);
	std::string str;
	while (std::getline(file, str))
	{
		img_names.push_back(str);
	}

	return img_names;
}

class TriangulationPairwiseProblem {
public:
	struct ImagePoint {
		double u, v;
		ImagePoint(const double u_, const double v_) {
			u = u_;
			v = v_;
		}
	};

	struct ReprojectionError {
		double u_mea, v_mea;
		Camera* cam;
		Configs* configs;

		~ReprojectionError() {

		}
		ReprojectionError(const double* uv, Camera* c, Configs* conf) {
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

			T tangential_dist_x = (p[1] * (r2 + 2.0 * xp * xp) + 2.0 * p[0] * xp * yp) * tan_post;
			T tangential_dist_y = (p[0] * (r2 + 2.0 * yp * yp) + 2.0 * p[1] * xp * yp) * tan_post;

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

		static ceres::CostFunction* Create(const double* uv, Camera* cp_in, Configs* conf) {
			ReprojectionError* re = new TriangulationPairwiseProblem::ReprojectionError(uv, cp_in, conf);
			return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3>(re));
		}
	};

	std::map<std::string, int> imgname_2_frameidx;
	std::map<int, std::string> frameidx_2_imgname;

	~TriangulationPairwiseProblem() {
		delete[] initial_world_points_;
		delete[] image_points_;
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
		 printf("  load: %s\n", path);
		FILE* fp = fopen(path, "rb"); // non-Windows use "r"
		if (fp != NULL) {

			char readBuffer[65536]; // max buffer size
			rapidjson::FileReadStream fs(fp, readBuffer, sizeof(readBuffer));

			// parse
			rapidjson::Document doc;
			doc.ParseStream(fs);
			assert(doc.HasMember("img_pts"));

			const rapidjson::Value& image_points = doc["img_pts"];
			for (int cam_idx = 0; cam_idx < 16; cam_idx++) {
				img_pts.push_back({});
				std::string key = std::to_string(cam_idx);
				const rapidjson::Value& cam_pts = image_points[key.c_str()];

				for (rapidjson::Value::ConstMemberIterator iter = cam_pts.MemberBegin(); iter != cam_pts.MemberEnd(); ++iter) {
					std::string p_idx = iter->name.GetString();
					const rapidjson::Value& pts = cam_pts[p_idx.c_str()];
					double u = pts[0].GetDouble();
					double v = pts[1].GetDouble();

					img_pts[cam_idx].insert({ p_idx, ImagePoint(u, v) });
				}
			}

			const rapidjson::Value& pairs = doc["shared_pts_indices"];
			for (rapidjson::Value::ConstMemberIterator iter = pairs.MemberBegin(); iter != pairs.MemberEnd(); ++iter) {
				std::string pair_str = iter->name.GetString();
				const rapidjson::Value& indices = pairs[pair_str.c_str()];

				std::vector<std::string> is;
				for (rapidjson::SizeType i = 0; i < indices.Size(); i++) {
					is.push_back(std::to_string(indices[i].GetInt()));
				}

				pair_point_indices.insert({ pair_str, is });
			}
			fclose(fp);
			return 0;
		} 
		printf("  NULL: %s\n", path);
		cout << errno << endl;
		std::getchar();
		return 1;
	}

	int LoadCameraParams(const char* path) {
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

			cams_.push_back(cam);
		}
		fclose(fp);
		return 0;

	}

	int LoadInitialWorldPoints(const char* path) {
		 printf("  load: %s\n", path);
		FILE* fp = fopen(path, "rb"); // non-Windows use "r"
		if (fp != NULL) {

			char readBuffer[65536]; // max buffer size
			rapidjson::FileReadStream fs(fp, readBuffer, sizeof(readBuffer));

			// parse
			rapidjson::Document doc;
			doc.ParseStream(fs);
			assert(doc.HasMember("3d_points"));

			const rapidjson::Value& pts = doc["3d_points"];
			int num_pts = int(pts.Size());
			initial_world_points_ = new double[num_pts * 3];

			for (rapidjson::SizeType i = 0; i < pts.Size(); i++) {
				const rapidjson::Value& pt = pts[i];
				double x = pt[0].GetDouble();
				double y = pt[1].GetDouble();
				double z = pt[2].GetDouble();
				initial_world_points_[i * 3 + 0] = x;
				initial_world_points_[i * 3 + 1] = y;
				initial_world_points_[i * 3 + 2] = z;
			}
			fclose(fp);
			return int(pts.Size());
		}
		printf("  NULL: %s\n", path);
		cout << errno << endl;
		std::getchar();
		return NULL;
	}
	double* GetInitialWorldPoint(const int p_idx) {
		return initial_world_points_ + p_idx * 3;
	}
	ImagePoint GetImagePoint(const int cam_idx, const std::string p_idx) {
		// printf("  GetImagePoint: %d, %s\n", cam_idx, p_idx.c_str());
		return img_pts[cam_idx].at(p_idx);
	}
	std::vector<std::string> GetValidPointIndices(const int cam_idx) {
		std::vector<std::string> p_indices;
		for (std::map<std::string, ImagePoint>::iterator it = img_pts[cam_idx].begin(); it != img_pts[cam_idx].end(); ++it) {
			p_indices.push_back(it->first);
		}
		return p_indices;
	}
	std::vector<std::string> GetPairIndices(const std::string& key) {
		// cout << "GetPairIndices: " << key << endl;
		return pair_point_indices.at(key);
	}
private:
	std::vector<Camera> cams_;
	Configs configs_;
	bool* detected_;
	double* image_points_;
	double* initial_world_points_;

	std::vector<std::map<std::string, ImagePoint>> img_pts;
	std::map<std::string, std::vector<std::string>> pair_point_indices;
};
TriangulationPairwise::TriangulationPairwise() {
}
int TriangulationPairwise::DoWork(std::string work_path) {
	google::InitGoogleLogging("HJP");
	work_path = "D:/Pictures/2019_12_13_LadaCapture/TriangulationPairwise";


	// load image names
	std::string img_names_path = work_path + std::string("/inputs/_image_names.txt");
	std::vector<std::string> img_names = LoadImageNames(img_names_path.c_str());

	for (int img_idx = 0; img_idx < img_names.size(); img_idx++) {
		auto start = std::chrono::system_clock::now();

		// log txt
		std::string log_path = work_path + std::string("/output/_log.txt");
		std::ofstream log_txt;
		log_txt.open(log_path.c_str(), std::ios_base::app);  // append instead of overwrite


		TriangulationPairwiseProblem *prob = new TriangulationPairwiseProblem();

		std::string input_cam_params = work_path + std::string("/inputs/CameraParameters/cam_params.json");
		prob->LoadCameraParams(input_cam_params.c_str());

		std::string img_name = img_names[img_idx];

		// load initial world points
		std::string initial_wp_path = work_path + std::string("/inputs/InitialWorldPts/") + img_name + std::string(".json");
		int num_wps = prob->LoadInitialWorldPoints(initial_wp_path.c_str());

		// load all iamge points for this image
		std::string img_pts_path = work_path + std::string("/inputs/PairwiseImgPts/") + img_name + std::string(".json");
		prob->LoadImagePoints(img_pts_path.c_str());

		// configs
		Configs* configs = prob->configs();
		configs->max_k = 6;
		configs->max_p = 2;
		configs->loss = Configs::Loss::None;
		configs->radial_model = Configs::RadialModel::rational;

		// output
		std::string output_path = work_path + std::string("/output/") + img_name + std::string(".txt");
		std::ofstream output_txt;
		output_txt.open(output_path.c_str());
		output_txt << configs->num_frames << " " << num_wps << "\n";

		// triangulate pairwise
		int num_solved = 0;
		int not_converged = 0;
		for (int i = 0; i < 16; i++) {
			for (int j = i + 1; j < 16; j++) {
				std::string pair_key = std::to_string(i) + std::string("_") + std::to_string(j);
				std::vector<std::string> point_indices = prob->GetPairIndices(pair_key);

				double* world_point_out = new double[point_indices.size() * 3]{ 0 };
				for (int k = 0; k < point_indices.size(); k++) {
					ceres::Problem ceres_prob;
					ceres::LossFunction* loss = NULL;

					// get image points
					double* wp0 = prob->GetInitialWorldPoint(std::atoi(point_indices[k].c_str()));
					double world_point[3] = { wp0[0], wp0[1], wp0[2] };

					TriangulationPairwiseProblem::ImagePoint pt1 = prob->GetImagePoint(i, point_indices[k]);
					double uv1[2] = { pt1.u, pt1.v };
					Camera* cam1 = prob->camera(i);

					TriangulationPairwiseProblem::ImagePoint pt2 = prob->GetImagePoint(j, point_indices[k]);
					double uv2[2] = { pt2.u, pt2.v };
					Camera* cam2 = prob->camera(j);

					ceres::CostFunction* cost_function1 = TriangulationPairwiseProblem::ReprojectionError::Create(uv1, cam1, configs);
					ceres::CostFunction* cost_function2 = TriangulationPairwiseProblem::ReprojectionError::Create(uv2, cam2, configs);
					ceres_prob.AddResidualBlock(cost_function1, loss, world_point);
					ceres_prob.AddResidualBlock(cost_function2, loss, world_point);

					// Solve!
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::DENSE_SCHUR;
					options.minimizer_progress_to_stdout = 0;

					options.num_linear_solver_threads = 1;
					options.num_threads = 1;
					//options.function_tolerance = 0.00000000001;
					//options.gradient_tolerance = 0.00000000001;
					ceres::Solver::Summary summary;
					ceres::Solve(options, &ceres_prob, &summary);
					if (summary.termination_type != ceres::TerminationType::CONVERGENCE) {
						// cout << summary.FullReport() << endl;
						not_converged += 1;
						cout << "  !! Not converged | " << not_converged << endl;
					}
					world_point_out[k * 3 + 0] = world_point[0];
					world_point_out[k * 3 + 1] = world_point[1];
					world_point_out[k * 3 + 2] = world_point[2];
				}

				// save to txt
				for (int p = 0; p < point_indices.size(); p++) {
					double x = world_point_out[p * 3 + 0];
					double y = world_point_out[p * 3 + 1];
					double z = world_point_out[p * 3 + 2];
					output_txt << pair_key << " " << p << " " << point_indices[p] << " " << x << " " << y << " " << z << endl;
					num_solved += 1;
				}
				delete[] world_point_out;
			}
		}

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		log_txt << "[" << currentDateTime() << "] dt=" << elapsed_seconds.count() << "s | " << img_idx  << " | " << img_name << " | " << num_solved << " points triangulated.\n";
		cout << "[" << currentDateTime() << "] dt=" << elapsed_seconds.count() << "s | " << img_idx << " | " << img_name << " | " << num_solved << " points triangulated.\n";
		log_txt.close();

		output_txt.close();
		delete prob;
	}
	return 0;
}