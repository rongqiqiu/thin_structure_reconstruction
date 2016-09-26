#ifndef THIN_STRUCTURE_RECONSTRUCTOR_H
#define THIN_STRUCTURE_RECONSTRUCTOR_H

#include "basic_types.h"

class CylinderPrimitive {
	Vector3d pa, pb;
	double r;
};

class ThinStructureReconstructor {
public:
	ThinStructureReconstructor() {}
	ThinStructureReconstructor(const Dataset& dataset) : dataset_(dataset) {
		ParseDataset();
	}
	ThinStructureReconstructor(const Dataset& dataset, const string& export_directory) 
		: dataset_(dataset), export_directory_(export_directory) {
		ParseDataset();
	}
	void SetDataset(const Dataset& dataset) {
		dataset_ = dataset;
		ParseDataset();
	}
	Dataset GetDataset() {
		return dataset_;
	}
	void ParseDataset();
	void SetExportDirectory(const string& export_directory) {
		export_directory_ = export_directory;
	}
	void ExportRawPoints();
	void ComputePCAValues();
	void LoadPCAValues();
	void ComputeCylinderHypotheses();
private:
	string export_directory_;
	Dataset dataset_;
	Vector3d reference_point_;
	pcl::PointCloud<pcl::PointXYZ> point_cloud_;
	vector<Vector3d> pca_values_;
	vector<int> index_pca_filtered_;
	pcl::PointCloud<pcl::PointXYZ> point_cloud_pca_filtered_;
	vector<CylinderPrimitive> cylinder_hypotheses_;
	double ComputeMean(const vector<int>& pointIdx, const int& dimension);
	double ComputeStandardDeviation(const vector<int>& pointIdx, const int& dimension);
	Vector3d ComputePCAValue(const vector<int>& pointIdx);
};

#endif