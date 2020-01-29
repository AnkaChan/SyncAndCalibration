#pragma once
#include "worker.h"

class BundleAdjustmentZFreeFrames : public Worker {
public:
	BundleAdjustmentZFreeFrames();
	int DoWork(std::string work_path);
};
