#pragma once
#include "worker.h"

class BundleAdjustment : public Worker {
public:
	BundleAdjustment();
	int DoWork(std::string work_path);
};
