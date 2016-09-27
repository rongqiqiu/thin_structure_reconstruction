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

void ThinStructureReconstructor::ExportPointCloud(const pcl::PointCloud<pcl::PointXYZ>& point_cloud, const string& file_name) {
	ofstream out_stream;
	out_stream.open(export_directory_ + file_name);
	for (int index = 0; index < point_cloud.points.size(); ++index) {
		const pcl::PointXYZ& point = point_cloud.points[index];
		out_stream << setprecision(8) << fixed << point.x << " " << point.y << " " << point.z << endl;
	}
	out_stream.close();
}

pcl::PointCloud<pcl::PointXYZ> ThinStructureReconstructor::ImportPointCloud(const string& file_name) {
	pcl::PointCloud<pcl::PointXYZ> point_cloud;
	ifstream in_stream;
	in_stream.open(export_directory_ + file_name);
	double x, y, z;
	while (in_stream >> x >> y >> z) {
		point_cloud.points.emplace_back(x, y, z);
	}
	point_cloud.width = point_cloud.points.size();
	point_cloud.height = 1;
	in_stream.close();
}

void ThinStructureReconstructor::ExportRawPoints() {
	ExportPointCloud(point_cloud_, "raw_pc.xyz");
}

double ThinStructureReconstructor::ComputeMean(const vector<int>& pointIdx, const int& dimension) {
	double mean = 0.0;
	for (int idx = 0; idx < pointIdx.size(); ++idx) {
		mean += point_cloud_subsampled_.points[pointIdx[idx]].data[dimension];
	}
	mean /= pointIdx.size();
	return mean;
}

double ThinStructureReconstructor::ComputeStandardDeviation(const vector<int>& pointIdx, const int& dimension) {
	double standard_deviation = 0.0;
	double mean = ComputeMean(pointIdx, dimension);
	for (int idx = 0; idx < pointIdx.size(); ++idx) {
		const double diff = point_cloud_subsampled_.points[pointIdx[idx]].data[dimension] - mean;
		standard_deviation += diff * diff;
	}
	standard_deviation /= pointIdx.size();
	standard_deviation = sqrt(standard_deviation);
	return standard_deviation;
}

Vector3d ThinStructureReconstructor::ComputePCAValue(const vector<int>& pointIdx) {
	Eigen::Vector3d pca_value;
	for (int dimension = 0; dimension < 3; ++dimension) {
		pca_value(dimension) = ComputeStandardDeviation(pointIdx, dimension);
	}
	pca_value /= pca_value.maxCoeff();
	return Vector3d(pca_value);
}

void ThinStructureReconstructor::ApplyRandomSubsampling(const double& sampling_ratio) {
	for (int index = 0; index < point_cloud_.points.size(); ++index) {
		double random_number = ((double) rand()) / RAND_MAX;
		if (random_number < sampling_ratio) {
			const pcl::PointXYZ& point = point_cloud_.points[index];
			point_cloud_subsampled_.points.push_back(point);
		}
	}
	point_cloud_subsampled_.width = point_cloud_subsampled_.points.size();
	point_cloud_subsampled_.height = 1;
	ExportPointCloud(point_cloud_subsampled_, "subsampled.xyz");
}

void ThinStructureReconstructor::ComputePCAValues() {
	pca_values_.clear();
	cout << "Computing PCA values" << endl;
	ApplyRandomSubsampling(0.2);
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(point_cloud_subsampled_.makeShared());
	cout << "Kd-tree is set up" << endl;
	for (int index = 0; index < point_cloud_.points.size(); ++index) {
		if (index % 100 == 0) {
			cout << "Computing " << index << " out of " << point_cloud_.points.size() << endl;
		}
		const pcl::PointXYZ& point = point_cloud_.points[index];
		vector<int> pointIdx;
		vector<float> pointSquaredDistance;
		kdtree.radiusSearch(point, 3.0, pointIdx, pointSquaredDistance);
		pca_values_.push_back(ComputePCAValue(pointIdx));
	}
	{
		ofstream out_stream;
		out_stream.open(export_directory_ + "pca.dat");
		for (int index = 0; index < point_cloud_.points.size(); ++index) {
			const Vector3d& pca_value(pca_values_[index]);
			out_stream << setprecision(2) << fixed << pca_value.x << " " << pca_value.y << " " << pca_value.z << endl;
		}
		out_stream.close();
	}
	{
		ofstream out_stream;
		out_stream.open(export_directory_ + "points_pca.xyz");
		for (int index = 0; index < point_cloud_.points.size(); ++index) {
			const pcl::PointXYZ& point = point_cloud_.points[index];
			const Vector3d& pca_value(pca_values_[index]);
			out_stream << setprecision(8) << fixed << point.x << " " << point.y << " " << point.z << " ";
			out_stream << setprecision(2) << fixed << pca_value.x << " " << pca_value.y << " " << pca_value.z << endl;
		}
		out_stream.close();
	}
}

void ThinStructureReconstructor::LoadPCAValues() {
	pca_values_.clear();
	cout << "Loading PCA values" << endl;
	ifstream in_stream;
	in_stream.open(export_directory_ + "pca.dat");
	double variation_x, variation_y, variation_z;
	while (in_stream >> variation_x >> variation_y >> variation_z) {
		pca_values_.emplace_back(variation_x, variation_y, variation_z);
	}
	in_stream.close();
}

bool ThinStructureReconstructor::IsVerticalLinear(const Vector3d& pca_value, const double& threshold) {
	return pca_value.x < threshold && pca_value.y < threshold && pca_value.z >= 1.0 - 1e-3;
}

void ThinStructureReconstructor::ComputeFilteredPoints() {
	index_filtered_.clear();
	point_cloud_filtered_.clear();
	cout << "Computing filtered points" << endl;
	for (int index = 0; index < point_cloud_.points.size(); ++index) {
		if (IsVerticalLinear(pca_values_[index], 0.6)) {
			index_filtered_.push_back(index);
			point_cloud_filtered_.points.push_back(point_cloud_.points[index]);
		}
	}
	point_cloud_filtered_.width = point_cloud_filtered_.points.size();
	point_cloud_filtered_.height = 1;
	ExportPointCloud(point_cloud_filtered_, "filtered.xyz");
}

void ThinStructureReconstructor::LoadFilteredPoints() {
	point_cloud_filtered_ = ImportPointCloud("filtered.xyz");
}

void ThinStructureReconstructor::ComputeCylinderHypotheses() {
}
