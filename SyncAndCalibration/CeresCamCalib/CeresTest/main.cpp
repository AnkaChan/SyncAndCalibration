#define MODE 0
#include "triangulation.h"
#include "bundle_adjustment.h"
#include "bundle_adjustment_6dof.h"
#include "bundle_adjustment_6dof+.h"
#include "bundle_adjustment_zfree.h"
#include "bundle_adjustment_zfree_frames.h"
#include "triangulation_pairwise.h"

int main(int argc, char** argv) {
	/*
	example usage in cmd.exe:
		C:\Users\hjoon\Documents\Visual Studio 2019\Projects\CeresCamCalib\x64\Release>CeresTest.exe 2 D:\CalibrationData\CameraCalibration\2019_12_24_Marianne_Capture
	*/
	int mode;
	std::string path;
	if (argc < 3) {
		/*
		if (mode == 6) {
		}
		else {
			std::cerr << "[ERROR] type input parameters: (int)mode (string)path" << endl;
			std::cerr << "        e.g., 1 D:/CalibrationData/CameraCalibration/2019_12_09_capture" << endl;
			std::getchar();
			return 1;
		}
		*/
		mode = 6;
		path = "";
	}
	else {
		mode = atoi(argv[1]);
		path = argv[2];
	}
	cout << "Mode: " << mode << " | Root path: " << path.c_str() << endl;

	Worker *worker = NULL;

	if (mode == 0) {
		cout << "Running: Triangulation" << endl;
		worker = new Triangulation();
	}
	else if (mode == 1) {
		cout << "Running: Bundle Adjustment" << endl;
		worker = new BundleAdjustment();
	}
	else if (mode == 2) {
		cout << "Running: Bundle Adjustment w/ 6DoF" << endl;
		worker = new BundleAdjustment6Dof();
	}
	else if (mode == 3) {
		cout << "Running: Bundle Adjustment w/ 6DoF, Z-free" << endl;
		worker = new BundleAdjustment6DofUp();
	}
	else if (mode == 4) {
		cout << "Running: Bundle Adjustment w/ Z-free" << endl;
		worker = new BundleAdjustmentZFree();
	}
	else if (mode == 5) {
		cout << "Running: Bundle Adjustment w/ Z-free every frame" << endl;
		worker = new BundleAdjustmentZFreeFrames();
	}
	else if (mode == 6) {
		cout << "Running: Triangulation Pairwise" << endl;
		worker = new TriangulationPairwise();
	}

	if (worker != NULL) {
		int result = worker->DoWork(path);
	}
	cout << "Press Enter to continue..." << endl;
	std::getchar();

	delete worker;
	return 0;
}