#pragma once
#define  _CRT_SECURE_NO_WARNINGS
#include "worker.h"

class Triangulation: public Worker {
public:
	Triangulation();
	int DoWork(std::string work_path);
};
