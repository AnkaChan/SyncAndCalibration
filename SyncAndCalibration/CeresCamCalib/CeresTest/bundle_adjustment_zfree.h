#pragma once
#include "worker.h"

class BundleAdjustmentZFree: public Worker {
public:
	BundleAdjustmentZFree();
	int DoWork(std::string work_path);
};
