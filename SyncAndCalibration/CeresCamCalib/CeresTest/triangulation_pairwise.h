#pragma once
#define  _CRT_SECURE_NO_WARNINGS
#include "worker.h"

class TriangulationPairwise : public Worker {
public:
	TriangulationPairwise();
	int DoWork(std::string work_path);
};
