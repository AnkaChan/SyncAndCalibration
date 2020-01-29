#pragma once
#pragma once
#include "worker.h"

class BundleAdjustment6DofUp : public Worker {
public:
	BundleAdjustment6DofUp();
	int DoWork(std::string work_path);
};
