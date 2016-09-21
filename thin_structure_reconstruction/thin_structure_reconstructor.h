#ifndef THIN_STRUCTURE_RECONSTRUCTOR_H
#define THIN_STRUCTURE_RECONSTRUCTOR_H

#include "basic_types.h"

class CylinderPrimitive {
	Point3d pa, pb;
	double r;
};

class ThinStructureReconstructor {
public:
	ThinStructureReconstructor() {}
	ThinStructureReconstructor(const Dataset& dataset) : dataset_(dataset) {}
	void SetDataset(const Dataset& dataset) {
		dataset_ = dataset;
	}
	Dataset GetDataset() {
		return dataset_;
	}
	void ParseDataset();
	void ExportRawPoints(const string& file_path);
	void ComputeCylinderHypotheses();
private:
	Dataset dataset_;
	Point3d reference_point_;
	pcl::PointCloud<pcl::PointXYZ> point_cloud_;
	vector<CylinderPrimitive> cylinder_hypotheses_;
};

#endif