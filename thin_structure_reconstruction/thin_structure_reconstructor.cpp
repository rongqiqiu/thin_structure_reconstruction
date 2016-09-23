#include "thin_structure_reconstructor.h"

#include <pcl/common/common_headers.h>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <iostream>
#include <fstream>
#include <iomanip>

void ThinStructureReconstructor::ParseDataset() {
	reference_point_ = dataset_.utm_reference_point;
	point_cloud_ = VectorVector3dToPointCloud(dataset_.points_utm, reference_point_);
}

void ThinStructureReconstructor::ExportRawPoints() {
	ofstream out_stream;
	out_stream.open(export_directory_ + "raw_pc.dat");
	for (int index = 0; index < point_cloud_.points.size(); ++index) {
		const pcl::PointXYZ& point = point_cloud_.points[index];
		out_stream << setprecision(8) << fixed << point.x << " " << point.y << " " << point.z << endl;
	}
	out_stream.close();
}

double ThinStructureReconstructor::ComputeStandardDeviation(const int& index, const vector<int>& pointIdx, const int& dimension) {
	return 0.0;
}

Vector3d ThinStructureReconstructor::ComputePCAValue(const int& index, const vector<int>& pointIdx) {
	Eigen::Vector3d pca_value;
	for (int dimension = 0; dimension < 3; ++dimension) {
		pca_value(dimension) = ComputeStandardDeviation(index, pointIdx, dimension);
	}
	pca_value /= pca_value.maxCoeff();
	return Vector3d(pca_value);
}

void ThinStructureReconstructor::ComputePCAValues() {
	pca_values_.clear();
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(point_cloud_.makeShared());
	for (int index = 0; index < point_cloud_.points.size(); ++index) {
		const pcl::PointXYZ& point = point_cloud_.points[index];
		vector<int> pointIdx;
		vector<float> pointSquaredDistance;
		kdtree.radiusSearch(point, 1.0, pointIdx, pointSquaredDistance);
		pca_values_.push_back(ComputePCAValue(index, pointIdx));
	}
	ofstream out_stream;
	out_stream.open(export_directory_ + "pca.dat");
	for (int index = 0; index < point_cloud_.points.size(); ++index) {
		const Vector3d& pca_value(pca_values_[index]);
		out_stream << setprecision(2) << fixed << pca_value.x << " " << pca_value.y << " " << pca_value.z << endl;
	}
	out_stream.close();
}

void ThinStructureReconstructor::LoadPCAValues() {
	pca_values_.clear();
	ifstream in_stream;
	in_stream.open(export_directory_ + "pca.dat");
	double variation_x, variation_y, variation_z;
	while (in_stream >> variation_x >> variation_y >> variation_z) {
		pca_values_.emplace_back(variation_x, variation_y, variation_z);
	}
	in_stream.close();
}

void ThinStructureReconstructor::ComputeCylinderHypotheses() {
}
