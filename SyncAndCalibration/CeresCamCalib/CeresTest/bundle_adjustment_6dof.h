#pragma once
#include "worker.h"

class BundleAdjustment6Dof : public Worker {
public:
	BundleAdjustment6Dof();
	int DoWork(std::string work_path);
};
