#include "thin_structure_reconstructor.h"

#include <Eigen/Geometry>

#include <opencv2/imgproc/imgproc.hpp>

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

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <time.h>

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

void ThinStructureReconstructor::ExportCylinderPrimitives(const vector<CylinderPrimitive>& cylinders, const string& file_name) {
	ofstream out_stream;
	out_stream.open(export_directory_ + file_name);
	for (int index = 0; index < cylinders.size(); ++index) {
		const CylinderPrimitive& cylinder = cylinders[index];
		out_stream << setprecision(8) << fixed << cylinder.pa.x << " " << cylinder.pa.y << " " << cylinder.pa.z << " ";
		out_stream << setprecision(8) << fixed << cylinder.pb.x << " " << cylinder.pb.y << " " << cylinder.pb.z << " ";
		out_stream << setprecision(8) << fixed << cylinder.r << endl;
	}
	out_stream.close();
}

void ThinStructureReconstructor::ExportCylinderMeshes(const vector<CylinderPrimitive>& cylinders, const string& file_name) {
	ofstream out_stream;
	out_stream.open(export_directory_ + file_name);
	int vertex_count = 0;
	for (int index = 0; index < cylinders.size(); ++index) {
		const CylinderPrimitive& cylinder = cylinders[index];
		const Eigen::Vector3d dir_z = (cylinder.pb.ToEigenVector() - cylinder.pa.ToEigenVector()).normalized();
		const Eigen::Vector3d dir_x = Eigen::Vector3d(1.0, 0.0, 0.0).cross(dir_z).normalized();
		const Eigen::Vector3d dir_y = dir_z.cross(dir_x).normalized();
		for (int subdivision_index = 0; subdivision_index < 16; ++subdivision_index) {
			const double angle = subdivision_index * PI * 2.0 / 16;
			const Eigen::Vector3d vertex = cylinder.pa.ToEigenVector() + dir_x * cylinder.r * cos(angle) + dir_y * cylinder.r * sin(angle);
			out_stream << setprecision(8) << fixed << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << endl;
		}
		for (int subdivision_index = 0; subdivision_index < 16; ++subdivision_index) {
			const double angle = subdivision_index * PI * 2.0 / 16;
			const Eigen::Vector3d vertex = cylinder.pb.ToEigenVector() + dir_x * cylinder.r * cos(angle) + dir_y * cylinder.r * sin(angle);
			out_stream << setprecision(8) << fixed << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << endl;
		}
		for (int subdivision_index = 0; subdivision_index < 16; ++subdivision_index) {
			out_stream << "f " << vertex_count + subdivision_index + 1 << " " << vertex_count + (subdivision_index + 1) % 16 + 1 << " " << vertex_count + subdivision_index + 16 + 1 << endl;
			out_stream << "f " << vertex_count + (subdivision_index + 1) % 16 + 1 << " " << vertex_count + (subdivision_index + 1) % 16 + 16 + 1 << " " << vertex_count + subdivision_index + 16 + 1 << endl;
		}
		vertex_count += 16 * 2;
	}
	out_stream.close();
}

void ThinStructureReconstructor::ExportTruncatedConePrimitives(const vector<TruncatedConePrimitive>& truncated_cones, const string& file_name) {
	ofstream out_stream;
	out_stream.open(export_directory_ + file_name);
	for (int index = 0; index < truncated_cones.size(); ++index) {
		const TruncatedConePrimitive& truncated_cone = truncated_cones[index];
		out_stream << setprecision(8) << fixed << truncated_cone.pa.x << " " << truncated_cone.pa.y << " " << truncated_cone.pa.z << " ";
		out_stream << setprecision(8) << fixed << truncated_cone.pb.x << " " << truncated_cone.pb.y << " " << truncated_cone.pb.z << " ";
		out_stream << setprecision(8) << fixed << truncated_cone.ra << " " << truncated_cone.rb << endl;
	}
	out_stream.close();
}

void ThinStructureReconstructor::ExportTruncatedConeMeshes(const vector<TruncatedConePrimitive>& truncated_cones, const string& file_name) {
	ofstream out_stream;
	out_stream.open(export_directory_ + file_name);
	int vertex_count = 0;
	for (int index = 0; index < truncated_cones.size(); ++index) {
		const TruncatedConePrimitive& truncated_cone = truncated_cones[index];
		const Eigen::Vector3d dir_z = (truncated_cone.pb.ToEigenVector() - truncated_cone.pa.ToEigenVector()).normalized();
		const Eigen::Vector3d dir_x = Eigen::Vector3d(1.0, 0.0, 0.0).cross(dir_z).normalized();
		const Eigen::Vector3d dir_y = dir_z.cross(dir_x).normalized();
		for (int subdivision_index = 0; subdivision_index < 16; ++subdivision_index) {
			const double angle = subdivision_index * PI * 2.0 / 16;
			const Eigen::Vector3d vertex = truncated_cone.pa.ToEigenVector() + dir_x * truncated_cone.ra * cos(angle) + dir_y * truncated_cone.ra * sin(angle);
			out_stream << setprecision(8) << fixed << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << endl;
		}
		for (int subdivision_index = 0; subdivision_index < 16; ++subdivision_index) {
			const double angle = subdivision_index * PI * 2.0 / 16;
			const Eigen::Vector3d vertex = truncated_cone.pb.ToEigenVector() + dir_x * truncated_cone.rb * cos(angle) + dir_y * truncated_cone.rb * sin(angle);
			out_stream << setprecision(8) << fixed << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << endl;
		}
		for (int subdivision_index = 0; subdivision_index < 16; ++subdivision_index) {
			out_stream << "f " << vertex_count + subdivision_index + 1 << " " << vertex_count + (subdivision_index + 1) % 16 + 1 << " " << vertex_count + subdivision_index + 16 + 1 << endl;
			out_stream << "f " << vertex_count + (subdivision_index + 1) % 16 + 1 << " " << vertex_count + (subdivision_index + 1) % 16 + 16 + 1 << " " << vertex_count + subdivision_index + 16 + 1 << endl;
		}
		vertex_count += 16 * 2;
	}
	out_stream.close();
}

void ThinStructureReconstructor::ExportPoleWithLampPrimitives(const vector<PoleWithLamp>& pole_with_lamps, const string& file_name) {
	ofstream out_stream;
	out_stream.open(export_directory_ + file_name);
	for (int index = 0; index < pole_with_lamps.size(); ++index) {
		const PoleWithLamp& pole_with_lamp = pole_with_lamps[index];
		const TruncatedConePrimitive& truncated_cone = pole_with_lamp.truncated_cone;
		out_stream << setprecision(8) << fixed << truncated_cone.pa.x << " " << truncated_cone.pa.y << " " << truncated_cone.pa.z << " ";
		out_stream << setprecision(8) << fixed << truncated_cone.pb.x << " " << truncated_cone.pb.y << " " << truncated_cone.pb.z << " ";
		out_stream << setprecision(8) << fixed << truncated_cone.ra << " " << truncated_cone.rb << " ";
		if (pole_with_lamp.has_lamp) {
			out_stream << "1 ";
			out_stream << setprecision(8) << fixed << pole_with_lamp.end_point.x << " " << pole_with_lamp.end_point.y << " " << pole_with_lamp.end_point.z << " ";
			out_stream << setprecision(8) << fixed << pole_with_lamp.ellipse_radius_large << " " << pole_with_lamp.ellipse_radius_small << endl;
		} else {
			out_stream << "0" << endl;
		}
	}
	out_stream.close();
}

void ThinStructureReconstructor::ExportPoleWithLampMeshes(const vector<PoleWithLamp>& pole_with_lamps, const string& file_name) {
	ofstream out_stream;
	out_stream.open(export_directory_ + file_name);
	int vertex_count = 0;
	for (int index = 0; index < pole_with_lamps.size(); ++index) {
		for (int pass = 0; pass <= 1; ++pass) {
			TruncatedConePrimitive truncated_cone;
			if (pass == 0) {
				truncated_cone = pole_with_lamps[index].truncated_cone;
			} else {
				if (!pole_with_lamps[index].has_lamp) {
					continue;
				}
				truncated_cone = pole_with_lamps[index].GetArmTruncatedCone();
			}
			const Eigen::Vector3d dir_z = (truncated_cone.pb.ToEigenVector() - truncated_cone.pa.ToEigenVector()).normalized();
			const Eigen::Vector3d dir_x = Eigen::Vector3d(1.0, 0.0, 0.0).cross(dir_z).normalized();
			const Eigen::Vector3d dir_y = dir_z.cross(dir_x).normalized();
			for (int subdivision_index = 0; subdivision_index < 16; ++subdivision_index) {
				const double angle = subdivision_index * PI * 2.0 / 16;
				const Eigen::Vector3d vertex = truncated_cone.pa.ToEigenVector() + dir_x * truncated_cone.ra * cos(angle) + dir_y * truncated_cone.ra * sin(angle);
				out_stream << setprecision(8) << fixed << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << endl;
			}
			for (int subdivision_index = 0; subdivision_index < 16; ++subdivision_index) {
				const double angle = subdivision_index * PI * 2.0 / 16;
				const Eigen::Vector3d vertex = truncated_cone.pb.ToEigenVector() + dir_x * truncated_cone.rb * cos(angle) + dir_y * truncated_cone.rb * sin(angle);
				out_stream << setprecision(8) << fixed << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << endl;
			}
			for (int subdivision_index = 0; subdivision_index < 16; ++subdivision_index) {
				out_stream << "f " << vertex_count + subdivision_index + 1 << " " << vertex_count + (subdivision_index + 1) % 16 + 1 << " " << vertex_count + subdivision_index + 16 + 1 << endl;
				out_stream << "f " << vertex_count + (subdivision_index + 1) % 16 + 1 << " " << vertex_count + (subdivision_index + 1) % 16 + 16 + 1 << " " << vertex_count + subdivision_index + 16 + 1 << endl;
			}
			vertex_count += 16 * 2;
		}
		if (pole_with_lamps[index].has_lamp) {
			const Eigen::Vector3d axis_x = Eigen::Vector3d(1.0, 0.0, 0.0);
			const Eigen::Vector3d axis_y = Eigen::Vector3d(0.0, 1.0, 0.0);
			const Eigen::Vector3d axis_z = Eigen::Vector3d(0.0, 0.0, 1.0);
			const Eigen::Vector3d south_pole = pole_with_lamps[index].end_point.ToEigenVector() - pole_with_lamps[index].ellipse_radius_small * axis_z;
			const Eigen::Vector3d north_pole = pole_with_lamps[index].end_point.ToEigenVector() + pole_with_lamps[index].ellipse_radius_small * axis_z;
			out_stream << setprecision(8) << fixed << "v " << south_pole.x() << " " << south_pole.y() << " " << south_pole.z() << endl;
			for (int vertical_angle_index = -7; vertical_angle_index <= 7; ++vertical_angle_index) {
				for (int horizontal_angle_index = 0; horizontal_angle_index < 16; ++horizontal_angle_index) {
					double vertical_angle = PI / 2.0 * (double) vertical_angle_index / 8;
					double horizontal_angle = PI * 2.0 * (double) horizontal_angle_index / 16;
					const Eigen::Vector3d vertex = pole_with_lamps[index].end_point.ToEigenVector()
						+ pole_with_lamps[index].ellipse_radius_small * axis_z * sin(vertical_angle)
						+ pole_with_lamps[index].ellipse_radius_large * axis_x * cos(vertical_angle) * cos(horizontal_angle)
						+ pole_with_lamps[index].ellipse_radius_large * axis_y * cos(vertical_angle) * sin(horizontal_angle);
					out_stream << setprecision(8) << fixed << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << endl;
				}
			}
			out_stream << setprecision(8) << fixed << "v " << north_pole.x() << " " << north_pole.y() << " " << north_pole.z() << endl;
			for (int horizontal_angle_index = 0; horizontal_angle_index < 16; ++horizontal_angle_index) {
				out_stream << "f " << vertex_count + 1 << " " << vertex_count + 1 + horizontal_angle_index % 16 + 1 << " " << vertex_count + 1 + (horizontal_angle_index + 1) % 16 + 1 << endl;
			}
			++ vertex_count;
			for (int vertical_angle_index = -7; vertical_angle_index < 7; ++vertical_angle_index) {
				for (int horizontal_angle_index = 0; horizontal_angle_index < 16; ++horizontal_angle_index) {
					out_stream << "f " << vertex_count + horizontal_angle_index % 16 + 1 << " " << vertex_count + (horizontal_angle_index + 1) % 16 + 1 << " " << vertex_count + 16 + horizontal_angle_index % 16 + 1 << endl;
					out_stream << "f " << vertex_count + (horizontal_angle_index + 1) % 16 + 1 << " " << vertex_count + 16 + horizontal_angle_index % 16 + 1 << " " << vertex_count + 16 + (horizontal_angle_index + 1) % 16 + 1 << endl;
				}
				vertex_count += 16;
			}
			for (int horizontal_angle_index = 0; horizontal_angle_index < 16; ++horizontal_angle_index) {
				out_stream << "f " << vertex_count + horizontal_angle_index % 16 + 1 << " " << vertex_count + (horizontal_angle_index + 1) % 16 + 1 << " " << vertex_count + 16 + 1 << endl;
			}
			vertex_count += 16 + 1;
			//vertex_count += 16 * 15 + 2;
		}
	}
	out_stream.close();
}

vector<CylinderPrimitive> ThinStructureReconstructor::ImportCylinderPrimitives(const string& file_name) {
	vector<CylinderPrimitive> cylinders;
	ifstream in_stream;
	in_stream.open(export_directory_ + file_name);
	CylinderPrimitive cylinder;
	string line;
	while (getline(in_stream, line)) {
		istringstream iss(line);
		iss >> cylinder.pa.x >> cylinder.pa.y >> cylinder.pa.z >> cylinder.pb.x >> cylinder.pb.y >> cylinder.pb.z >> cylinder.r;
		cylinders.push_back(cylinder);
	}
	in_stream.close();
	return cylinders;
}

vector<TruncatedConePrimitive> ThinStructureReconstructor::ImportTruncatedConePrimitives(const string& file_name) {
	vector<TruncatedConePrimitive> truncated_cones;
	ifstream in_stream;
	in_stream.open(export_directory_ + file_name);
	TruncatedConePrimitive truncated_cone;
	string line;
	while (getline(in_stream, line)) {
		istringstream iss(line);
		iss >> truncated_cone.pa.x >> truncated_cone.pa.y >> truncated_cone.pa.z;
		iss >> truncated_cone.pb.x >> truncated_cone.pb.y >> truncated_cone.pb.z;
		iss >> truncated_cone.ra >> truncated_cone.rb;
		truncated_cones.push_back(truncated_cone);
	}
	in_stream.close();
	return truncated_cones;
}

vector<PoleWithLamp> ThinStructureReconstructor::ImportPoleWithLampPrimitives(const string& file_name) {
	vector<PoleWithLamp> pole_with_lamps;
	ifstream in_stream;
	in_stream.open(export_directory_ + file_name);
	string line;
	while (getline(in_stream, line)) {
		PoleWithLamp pole_with_lamp;
		TruncatedConePrimitive& truncated_cone = pole_with_lamp.truncated_cone;
		istringstream iss(line);
		iss >> truncated_cone.pa.x >> truncated_cone.pa.y >> truncated_cone.pa.z;
		iss >> truncated_cone.pb.x >> truncated_cone.pb.y >> truncated_cone.pb.z;
		iss >> truncated_cone.ra >> truncated_cone.rb;
		int has_lamp;
		iss >> has_lamp;
		if (has_lamp) {
			pole_with_lamp.has_lamp = true;
			iss >> pole_with_lamp.end_point.x >> pole_with_lamp.end_point.y >> pole_with_lamp.end_point.z;
			iss >> pole_with_lamp.ellipse_radius_large >> pole_with_lamp.ellipse_radius_small;
		} else {
			pole_with_lamp.has_lamp = false;
		}
		pole_with_lamps.push_back(pole_with_lamp);
	}
	in_stream.close();
	return pole_with_lamps;
}

pcl::PointCloud<pcl::PointXYZ> ThinStructureReconstructor::ImportPointCloud(const string& file_name) {
	cout << "Loading point cloud " << file_name << endl;
	pcl::PointCloud<pcl::PointXYZ> point_cloud;
	ifstream in_stream;
	in_stream.open(export_directory_ + file_name);
	double x, y, z;
	string line;
	while (getline(in_stream, line)) {
		istringstream iss(line);
		iss >> x >> y >> z;
		point_cloud.points.push_back(pcl::PointXYZ(x, y, z));
	}
	point_cloud.width = point_cloud.points.size();
	point_cloud.height = 1;
	in_stream.close();
	return point_cloud;
}

void ThinStructureReconstructor::ExportReferencePoint() {
	ofstream out_stream;
	out_stream.open(export_directory_ + "reference_point.xyz");
	out_stream << setprecision(8) << fixed << reference_point_.x << " " << reference_point_.y << " " << reference_point_.z << endl;
	out_stream.close();
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
	if (pointIdx.empty()) {
		return Vector3d(0.0, 0.0, 0.0);
	}
	Eigen::Vector3d pca_value;
	for (int dimension = 0; dimension < 3; ++dimension) {
		pca_value(dimension) = ComputeStandardDeviation(pointIdx, dimension);
	}
	const double max_coeff = pca_value.maxCoeff();
	if (fabs(max_coeff) < 1e-6) {
		return Vector3d(0.0, 0.0, 0.0); 
	}
	pca_value /= pca_value.maxCoeff();
	return Vector3d(pca_value);
}

void ThinStructureReconstructor::ApplyRandomSubsampling(const double& sampling_ratio) {
	point_cloud_subsampled_.clear();
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
	//ApplyRandomSubsampling(1.0);
	ApplyRandomSubsampling(0.2);
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(point_cloud_subsampled_.makeShared());
	cout << "Kd-tree is set up" << endl;
	for (int index = 0; index < point_cloud_.points.size(); ++index) {
		if (index % 100 == 0) {
			cout << "Computing " << index << " out of " << point_cloud_.points.size() << endl;
		}
		const pcl::PointXYZ& point = point_cloud_.points[index];
		//cout << dataset_.utm_box.bbox.min_bounds.x << " " << dataset_.utm_box.bbox.min_bounds.y << " - ";
		//cout << dataset_.utm_box.bbox.max_bounds.x << " " << dataset_.utm_box.bbox.max_bounds.y << " : ";
		//cout << point.x << " " << point.y << endl;
		if (!dataset_.utm_box.Contains(point.x, point.y, dataset_.utm_box.utm_zone)) {
			pca_values_.push_back(Vector3d(0.0, 0.0, 0.0));
		} else {
			vector<int> pointIdx;
			vector<float> pointSquaredDistance;
			kdtree.radiusSearch(point, 1.0, pointIdx, pointSquaredDistance);
			pca_values_.push_back(ComputePCAValue(pointIdx));
		}
	}
	if (pca_values_.size() != point_cloud_.points.size()) {
		cout << "PCA values size (" << pca_values_.size() << ") does not match with point cloud size (" << point_cloud_.points.size() << ")" << endl;
		throw std::exception();
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
	string line;
	while (getline(in_stream, line)) {
		istringstream iss(line);
		iss >> variation_x >> variation_y >> variation_z;
		pca_values_.push_back(Vector3d(variation_x, variation_y, variation_z));
	}
	in_stream.close();
	if (pca_values_.size() != point_cloud_.points.size()) {
		cout << "PCA values size (" << pca_values_.size() << ") does not match with point cloud size (" << point_cloud_.points.size() << ")" << endl;
		throw std::exception();
	}
}

bool ThinStructureReconstructor::IsVerticalLinear(const Vector3d& pca_value, const double& threshold) {
	return pca_value.x < threshold && pca_value.y < threshold && pca_value.z >= 1.0 - 1e-3;
}

void ThinStructureReconstructor::ComputePCAFilteredPoints() {
	point_cloud_pca_filtered_.clear();
	cout << "Computing PCA filtered points" << endl;
	for (int index = 0; index < point_cloud_.points.size(); ++index) {
		if (IsVerticalLinear(pca_values_[index], 0.7)) {
			point_cloud_pca_filtered_.points.push_back(point_cloud_.points[index]);
		}
	}
	point_cloud_pca_filtered_.width = point_cloud_pca_filtered_.points.size();
	point_cloud_pca_filtered_.height = 1;
	ExportPointCloud(point_cloud_pca_filtered_, "pca_filtered.xyz");
	point_cloud_filtered_ = point_cloud_pca_filtered_;
}

void ThinStructureReconstructor::LoadPCAFilteredPoints() {
	cout << "Loading PCA filtered points" << endl;
	point_cloud_pca_filtered_ = ImportPointCloud("pca_filtered.xyz");
	point_cloud_filtered_ = point_cloud_pca_filtered_;
}

void ThinStructureReconstructor::ComputeEmptinessValues() {
	emptiness_values_.clear();
	cout << "Computing emptiness values" << endl;
	ApplyRandomSubsampling(1.0);
	//ApplyRandomSubsampling(0.2);
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(point_cloud_subsampled_.makeShared());
	cout << "Kd-tree is set up" << endl;
	for (int index = 0; index < point_cloud_pca_filtered_.points.size(); ++index) {
		if (index % 100 == 0) {
			cout << "Computing " << index << " out of " << point_cloud_pca_filtered_.points.size() << endl;
		}
		const pcl::PointXYZ& point = point_cloud_pca_filtered_.points[index];
		vector<int> pointIdxSmall;
		vector<float> pointSquaredDistanceSmall;
		kdtree.radiusSearch(point, 1.0, pointIdxSmall, pointSquaredDistanceSmall);
		int pointNumSmall = 0;
		for (int idx = 0; idx < pointIdxSmall.size(); ++idx) {
			if (fabs(point_cloud_subsampled_.points[pointIdxSmall[idx]].z - point.z) <= 0.5) {
				++pointNumSmall;
			}
		}
		vector<int> pointIdxLarge;
		vector<float> pointSquaredDistanceLarge;
		kdtree.radiusSearch(point, 2.0, pointIdxLarge, pointSquaredDistanceLarge);
		int pointNumLarge = 0;
		for (int idx = 0; idx < pointIdxLarge.size(); ++idx) {
			if (fabs(point_cloud_subsampled_.points[pointIdxLarge[idx]].z - point.z) <= 0.5) {
				++pointNumLarge;
			}
		}
		emptiness_values_.push_back(min((double) (pointNumLarge - pointNumSmall) / pointNumSmall, 10.0));
	}
	if (emptiness_values_.size() != point_cloud_pca_filtered_.points.size()) {
		cout << "Emptiness values size (" << emptiness_values_.size() << ") does not match with PCA filtered point cloud size (" << point_cloud_pca_filtered_.points.size() << ")" << endl;
		throw std::exception();
	}
	{
		ofstream out_stream;
		out_stream.open(export_directory_ + "emptiness.dat");
		for (int index = 0; index < point_cloud_pca_filtered_.points.size(); ++index) {
			const double& emptiness_value(emptiness_values_[index]);
			out_stream << setprecision(8) << fixed << emptiness_value << endl;
		}
		out_stream.close();
	}
	{
		ofstream out_stream;
		out_stream.open(export_directory_ + "points_emptiness.xyz");
		for (int index = 0; index < point_cloud_pca_filtered_.points.size(); ++index) {
			const pcl::PointXYZ& point = point_cloud_pca_filtered_.points[index];
			const double& emptiness_value(emptiness_values_[index]);
			out_stream << setprecision(8) << fixed << point.x << " " << point.y << " " << point.z << " ";
			out_stream << setprecision(8) << fixed << emptiness_value << endl;
		}
		out_stream.close();
	}
}

void ThinStructureReconstructor::LoadEmptinessValues() {
	emptiness_values_.clear();
	cout << "Loading emptiness values" << endl;
	ifstream in_stream;
	in_stream.open(export_directory_ + "emptiness.dat");
	double emptiness;
	string line;
	while (getline(in_stream, line)) {
		istringstream iss(line);
		iss >> emptiness;
		emptiness_values_.push_back(emptiness);
	}
	in_stream.close();
	if (emptiness_values_.size() != point_cloud_pca_filtered_.points.size()) {
		cout << "Emptiness values size (" << emptiness_values_.size() << ") does not match with PCA filtered point cloud size (" << point_cloud_pca_filtered_.points.size() << ")" << endl;
		throw std::exception();
	}
}

void ThinStructureReconstructor::ComputeEmptinessFilteredPoints() {
	point_cloud_emptiness_filtered_.clear();
	cout << "Computing emptiness filtered points" << endl;
	for (int index = 0; index < point_cloud_pca_filtered_.points.size(); ++index) {
		if (emptiness_values_[index] < 0.5) {
			point_cloud_emptiness_filtered_.points.push_back(point_cloud_pca_filtered_.points[index]);
		}
	}
	point_cloud_emptiness_filtered_.width = point_cloud_emptiness_filtered_.points.size();
	point_cloud_emptiness_filtered_.height = 1;
	ExportPointCloud(point_cloud_emptiness_filtered_, "emptiness_filtered.xyz");
	point_cloud_filtered_ = point_cloud_emptiness_filtered_;
}

void ThinStructureReconstructor::LoadEmptinessFilteredPoints() {
	cout << "Loading emptiness filtered points" << endl;
	point_cloud_emptiness_filtered_ = ImportPointCloud("emptiness_filtered.xyz");
	point_cloud_filtered_ = point_cloud_emptiness_filtered_;
}

Vector3d ThinStructureReconstructor::ComputeXYCentroid(const pcl::PointCloud<pcl::PointXYZ>& point_cloud, const vector<int>& pointIdx) {
	vector<double> x_values(pointIdx.size());
	vector<double> y_values(pointIdx.size());
	for (int index = 0; index < pointIdx.size(); ++index) {
		x_values[index] = point_cloud.points[pointIdx[index]].x;
		y_values[index] = point_cloud.points[pointIdx[index]].y;
	}
	sort(x_values.begin(), x_values.end());
	sort(y_values.begin(), y_values.end());
	Vector3d centroid;
	if (pointIdx.size() % 2 == 0) {
		centroid.x = (x_values[pointIdx.size() / 2 - 1] + x_values[pointIdx.size() / 2]) / 2.0;
		centroid.y = (y_values[pointIdx.size() / 2 - 1] + y_values[pointIdx.size() / 2]) / 2.0;
	} else {
		centroid.x = x_values[pointIdx.size() / 2];
		centroid.y = y_values[pointIdx.size() / 2];
	}
	centroid.z = 0.0;
	return centroid;
}

void ThinStructureReconstructor::ComputeExtentsFromPointCloud(const pcl::PointCloud<pcl::PointXYZ>& point_cloud, const vector<int>& pointIdx, const Vector3d& axis, const Vector3d& point, double* min_dot, double* max_dot) {
	if (pointIdx.empty() || min_dot == NULL || max_dot == NULL) {
		return;
	}

	for (int index = 0; index < pointIdx.size(); ++index) {
		double dot = (Vector3d(point_cloud.points[pointIdx[index]]).ToEigenVector() - point.ToEigenVector()).dot(axis.ToEigenVector());
		if (index == 0 || dot < (*min_dot)) {
			*min_dot = dot;
		}
		if (index == 0 || dot > (*max_dot)) {
			*max_dot = dot;
		}
	}
}

CylinderPrimitive ThinStructureReconstructor::ComputeVerticalLine(const pcl::PointCloud<pcl::PointXYZ>& point_cloud, const vector<int>& pointIdx) {
	const Vector3d centroid = ComputeXYCentroid(point_cloud, pointIdx);
	const Vector3d axis(0.0, 0.0, 1.0);
	double min_height, max_height;
	ComputeExtentsFromPointCloud(point_cloud, pointIdx, axis, centroid, &min_height, &max_height);
	CylinderPrimitive cylinder;
	cylinder.pa = Vector3d(centroid.ToEigenVector() + axis.ToEigenVector() * min_height);
	cylinder.pb = Vector3d(centroid.ToEigenVector() + axis.ToEigenVector() * max_height);
	cylinder.r = 1.0;
	return cylinder;
}

pcl::PointCloud<pcl::PointXYZ> ThinStructureReconstructor::ProjectXY(const pcl::PointCloud<pcl::PointXYZ>& input_cloud) {
	pcl::PointCloud<pcl::PointXYZ> projected_cloud = input_cloud;
	for (int index = 0; index < projected_cloud.points.size(); ++index) {
		projected_cloud.points[index].z = 0.0;
	}
	return projected_cloud;
}

void ThinStructureReconstructor::ComputeRANSAC() {
	cout << "Computing RANSAC" << endl;
	cout << "point_cloud_filtered_.size() == " << point_cloud_filtered_.size() << endl;
	cylinder_hypotheses_.clear();
	pcl::PointCloud<pcl::PointXYZ> point_cloud_projected = ProjectXY(point_cloud_filtered_);
	pcl::PointCloud<pcl::PointXYZ> point_cloud_remaining = point_cloud_filtered_;
	pcl::PointCloud<pcl::PointXYZ> point_cloud_projected_remaining = point_cloud_projected;
	while (true) {
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud(point_cloud_projected_remaining.makeShared());
		vector<int> bestPointIdx;
		for (int index = 0; index < point_cloud_projected_remaining.points.size(); ++index) {
			vector<int> pointIdx;
			vector<float> pointSquaredDistance;
			kdtree.radiusSearch(point_cloud_projected_remaining.points[index], 1.0, pointIdx, pointSquaredDistance);
			if (pointIdx.size() > bestPointIdx.size()) {
				bestPointIdx = pointIdx;
			}
		}
		cout << "Detection! Size: " << bestPointIdx.size() << endl;
		if (bestPointIdx.size() < 400) {
			break;
		}
		sort(bestPointIdx.begin(), bestPointIdx.end());
		cylinder_hypotheses_.push_back(ComputeVerticalLine(point_cloud_remaining, bestPointIdx));
		pcl::PointCloud<pcl::PointXYZ> point_cloud_temp;
		pcl::PointCloud<pcl::PointXYZ> point_cloud_projected_temp;
		int current_index = 0;
		for (int index = 0; index < point_cloud_remaining.points.size(); ++index) {
			if (current_index < bestPointIdx.size() && index == bestPointIdx[current_index]) {
				current_index ++;
				continue;
			}
			point_cloud_temp.points.push_back(point_cloud_remaining.points[index]);
			point_cloud_projected_temp.points.push_back(point_cloud_projected_remaining.points[index]);
		}
		point_cloud_temp.width = point_cloud_temp.points.size();
		point_cloud_temp.height = 1;
		point_cloud_projected_temp.width = point_cloud_projected_temp.points.size();
		point_cloud_projected_temp.height = 1;

		point_cloud_remaining = point_cloud_temp;
		point_cloud_projected_remaining = point_cloud_projected_temp;
	}
	ExportCylinderPrimitives(cylinder_hypotheses_, "cylinder_hypotheses.dat");
	ExportCylinderMeshes(cylinder_hypotheses_, "cylinder_hypotheses.obj");
}

void ThinStructureReconstructor::LoadRANSAC() {
	cout << "Loading RANSAC" << endl;
	cylinder_hypotheses_ = ImportCylinderPrimitives("cylinder_hypotheses.dat");
	cout << "Number of cylinder hypotheses: " << cylinder_hypotheses_.size() << endl;
	for (int index = 0; index < cylinder_hypotheses_.size(); ++index) {
		const CylinderPrimitive& cylinder = cylinder_hypotheses_[index];
		cout << "Cylinder #" << index << ": (" << cylinder.pa.x << " " << cylinder.pa.y << " " << cylinder.pa.z << "),"
									   << " (" << cylinder.pb.x << " " << cylinder.pb.y << " " << cylinder.pb.z << "),"
									   << " " << cylinder.r << endl;
	}
}

void ThinStructureReconstructor::ComputeCylinderHypotheses() {
}

CylinderPrimitive ThinStructureReconstructor::ExtendVerticalCylinder(const CylinderPrimitive& cylinder) {
	CylinderPrimitive cylinder_new = cylinder;
	double min_height = cylinder_new.pa.z;
	double max_height = cylinder_new.pb.z;
	for (int index = 0; index < point_cloud_.points.size(); ++index) {
		if (Eigen::Vector2d(cylinder_new.pa.x - point_cloud_.points[index].x, 
			                cylinder_new.pa.y - point_cloud_.points[index].y).norm() < 1.0) {
			if (point_cloud_.points[index].z < min_height) {
				min_height = point_cloud_.points[index].z;
			}
			if (point_cloud_.points[index].z > max_height) {
				max_height = point_cloud_.points[index].z;
			}
		}
	}
	cylinder_new.pa.z = min_height;
	cylinder_new.pb.z = max_height;
	return cylinder_new;
}

void ThinStructureReconstructor::ComputeExtendedVerticalCylinders() {
	cout << "Computing extended vertical cylinders" << endl;
	extended_cylinder_hypotheses_.clear();
	for (int cylinder_index = 0; cylinder_index < cylinder_hypotheses_.size(); ++cylinder_index) {
		const CylinderPrimitive& cylinder = cylinder_hypotheses_[cylinder_index];
		const CylinderPrimitive cylinder_new = ExtendVerticalCylinder(cylinder);
		extended_cylinder_hypotheses_.push_back(cylinder_new);
	}
	ExportCylinderPrimitives(extended_cylinder_hypotheses_, "extended_cylinder_hypotheses.dat");
	ExportCylinderMeshes(extended_cylinder_hypotheses_, "extended_cylinder_hypotheses.obj");
}

void ThinStructureReconstructor::LoadExtendedVerticalCylinders() {
	cout << "Loading extended vertical cylinders" << endl;
	extended_cylinder_hypotheses_ = ImportCylinderPrimitives("extended_cylinder_hypotheses.dat");
	cout << "Number of extended cylinder hypotheses: " << extended_cylinder_hypotheses_.size() << endl;
	for (int index = 0; index < extended_cylinder_hypotheses_.size(); ++index) {
		const CylinderPrimitive& cylinder = extended_cylinder_hypotheses_[index];
		cout << "Cylinder #" << index << ": (" << cylinder.pa.x << " " << cylinder.pa.y << " " << cylinder.pa.z << "),"
									   << " (" << cylinder.pb.x << " " << cylinder.pb.y << " " << cylinder.pb.z << "),"
									   << " " << cylinder.r << endl;
	}
}

void ThinStructureReconstructor::LoadAndCropSubimages() {
	cropped_image_cameras_.clear();
	cropped_subimages_.clear();
	int output_index = 0;
	for (int index = 0; index < dataset_.image_cameras.size(); ++index) {
		cout << "Loading and cropping image #" << index << endl;
		const ImageCamera& image_camera = dataset_.image_cameras[index];
		ImageCamera cropped_image_camera;
		cropped_image_camera.camera_model = image_camera.camera_model;
		const cv::Mat raw_subimage = cv::imread(image_camera.subimage.file_path, CV_LOAD_IMAGE_COLOR);
		cropped_image_camera.subimage.original_image_size = image_camera.subimage.original_image_size;
		bool is_vertical = true;
		for (int cylinder_index = 0; cylinder_index < extended_cylinder_hypotheses_.size(); ++cylinder_index) {
			const CylinderPrimitive& cylinder = extended_cylinder_hypotheses_[cylinder_index];
			const Vector2d projected_pixel_a = ProjectShiftedUtmPoint(cropped_image_camera.camera_model, cylinder.pa);
			const Vector2d projected_pixel_b = ProjectShiftedUtmPoint(cropped_image_camera.camera_model, cylinder.pb);
			cropped_image_camera.subimage.bounds.ExtendsTo(projected_pixel_a);
			cropped_image_camera.subimage.bounds.ExtendsTo(projected_pixel_b);

			const Eigen::Vector2d axis_vector = (projected_pixel_b.ToEigenVector() - projected_pixel_a.ToEigenVector()).normalized();
			const Eigen::Vector2d vertical_vector(0.0, 1.0);
			const double dot_product = axis_vector.dot(vertical_vector);
			if (fabs(dot_product) < 0.9) {
				is_vertical = false;
			}
		}
		cropped_image_camera.subimage.bounds.Intersect(image_camera.subimage.bounds);
		//cout << "Is vertical: " << is_vertical << endl;
		//cout << "Cropped bounds: (" << cropped_image_camera.subimage.bounds.min_bounds.x << ", " << cropped_image_camera.subimage.bounds.min_bounds.y << ")"
		//	 << " - (" << cropped_image_camera.subimage.bounds.max_bounds.x << ", " << cropped_image_camera.subimage.bounds.max_bounds.y << ")" << endl;
		//cout << "Is non-empty: " << !cropped_image_camera.subimage.bounds.IsEmpty() << endl;

		if (!cropped_image_camera.subimage.bounds.IsEmpty() && is_vertical) {
			cropped_image_camera.subimage.bounds.Expands(100);
			cropped_image_camera.subimage.bounds.Intersect(image_camera.subimage.bounds);
			cv::Mat cropped_subimage = ExtractCroppedSubimage(raw_subimage, image_camera.subimage.bounds, cropped_image_camera.subimage.bounds);
			cv::imwrite(export_directory_ + NumberToString(output_index) + "_cropped.png", cropped_subimage);
			++output_index;
			cropped_image_cameras_.push_back(cropped_image_camera);
			cropped_subimages_.push_back(cropped_subimage);
		}
	}
}

cv::Mat ThinStructureReconstructor::ExtractCroppedSubimage(const cv::Mat& raw_subimage, const HalfOpenBox2i& raw_bounds, const HalfOpenBox2i& cropped_bounds) {
	const cv::Rect rect(cropped_bounds.min_bounds.x - raw_bounds.min_bounds.x,
				  cropped_bounds.min_bounds.y - raw_bounds.min_bounds.y,
				  cropped_bounds.max_bounds.x - cropped_bounds.min_bounds.x,
				  cropped_bounds.max_bounds.y - cropped_bounds.min_bounds.y);
	return raw_subimage(rect).clone();
}

Vector2d ThinStructureReconstructor::ProjectShiftedUtmPoint(const ExportCameraModel& camera_model, const Vector3d& shifted_utm_point) {
	return camera_model.ProjectUtm(Vector3d(shifted_utm_point.ToEigenVector() + reference_point_.ToEigenVector()), dataset_.utm_box.utm_zone);
}

void ThinStructureReconstructor::ExportRawSubimages() {
	for (int index = 0; index < dataset_.image_cameras.size(); ++index) {
		const ImageCamera& image_camera = dataset_.image_cameras[index];
		const cv::Mat raw_subimage = cv::imread(image_camera.subimage.file_path, CV_LOAD_IMAGE_COLOR);
		cv::imwrite(export_directory_ + NumberToString(index) + ".png", raw_subimage);
	}
}

void ThinStructureReconstructor::ExportRawSubimagesWithMarkedEcefPoint(const Vector3d& ecef_point) {
	for (int index = 0; index < dataset_.image_cameras.size(); ++index) {
		const ImageCamera& image_camera = dataset_.image_cameras[index];
		cv::Mat subimage = cv::imread(image_camera.subimage.file_path, CV_LOAD_IMAGE_COLOR);
		MarkSubimageWithEcefPoint(image_camera.subimage, image_camera.camera_model, ecef_point, cv::Scalar(0, 0, 255), 5.0, &subimage);
		cv::imwrite(export_directory_ + NumberToString(index) + "_marked_point.png", subimage);
	}
}

void ThinStructureReconstructor::ExportCroppedSubimagesWithMarkedEcefPoint(const Vector3d& ecef_point) {
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		const ImageCamera& image_camera = cropped_image_cameras_[index];
		cv::Mat subimage = cropped_subimages_[index].clone();
		MarkSubimageWithEcefPoint(image_camera.subimage, image_camera.camera_model, ecef_point, cv::Scalar(0, 0, 255), 5.0, &subimage);
		cv::imwrite(export_directory_ + NumberToString(index) + "_marked_point.png", subimage);
	}
}

void ThinStructureReconstructor::ExportRawSubimagesWithMarkedHypotheses() {
	cout << "About to export " << dataset_.image_cameras.size() << " images." << endl;
	for (int index = 0; index < dataset_.image_cameras.size(); ++index) {
		cout << "Exporting image #" << index << endl;
		const ImageCamera& image_camera = dataset_.image_cameras[index];
		cv::Mat subimage = cv::imread(image_camera.subimage.file_path, CV_LOAD_IMAGE_COLOR);
		for (int cylinder_index = 0; cylinder_index < extended_cylinder_hypotheses_.size(); ++cylinder_index) {
			const CylinderPrimitive& cylinder = extended_cylinder_hypotheses_[cylinder_index];
			MarkSubimageWithCylinderSurfaceAxisOutline(image_camera.subimage, image_camera.camera_model, cylinder, &subimage);
		}
		cv::imwrite(export_directory_ + NumberToString(index) + "_marked_cylinder.png", subimage);
	}
}

void ThinStructureReconstructor::ExportCroppedSubimagesWithMarkedHypotheses() {
	cout << "About to export " << cropped_image_cameras_.size() << " images." << endl;
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		cout << "Exporting image #" << index << endl;
		const ImageCamera& image_camera = cropped_image_cameras_[index];
		cv::Mat subimage = cropped_subimages_[index].clone();
		for (int cylinder_index = 0; cylinder_index < extended_cylinder_hypotheses_.size(); ++cylinder_index) {
			const CylinderPrimitive& cylinder = extended_cylinder_hypotheses_[cylinder_index];
			MarkSubimageWithCylinderSurfaceAxisOutline(image_camera.subimage, image_camera.camera_model, cylinder, &subimage);
		}
		cv::imwrite(export_directory_ + NumberToString(index) + "_marked_cylinder.png", subimage);
	}
}

void ThinStructureReconstructor::ExportCroppedSubimagesWithMarkedTruncatedCone(const TruncatedConePrimitive& truncated_cone, const string& file_name) {
	cout << "About to export " << cropped_image_cameras_.size() << " images." << endl;
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		cout << "Exporting image #" << index << endl;
		const ImageCamera& image_camera = cropped_image_cameras_[index];
		cv::Mat subimage = cropped_subimages_[index].clone();
		MarkSubimageWithTruncatedConeSurfaceAxisOutline(image_camera.subimage, image_camera.camera_model, truncated_cone, &subimage);
		cv::imwrite(export_directory_ + NumberToString(index) + file_name, subimage);
	}
}

void ThinStructureReconstructor::MarkSubimageWithEcefPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& ecef_point, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage) {
	const Vector2d projected_pixel = camera_model.ProjectEcef(ecef_point);
	MarkSubimagePixel(rasterized_subimage, projected_pixel, color, radius_in_pixel, subimage);
}

void ThinStructureReconstructor::MarkSubimageWithUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& utm_point, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage) {
	const Vector2d projected_pixel = camera_model.ProjectUtm(utm_point, dataset_.utm_box.utm_zone);
	MarkSubimagePixel(rasterized_subimage, projected_pixel, color, radius_in_pixel, subimage);
}

void ThinStructureReconstructor::MarkSubimageWithShiftedUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& shifted_utm_point, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage) {
	MarkSubimageWithUtmPoint(rasterized_subimage, camera_model, Vector3d(shifted_utm_point.ToEigenVector() + reference_point_.ToEigenVector()), color, radius_in_pixel, subimage);
}

void ThinStructureReconstructor::MarkSubimageWithCylinderSurfaceAxisOutline(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, cv::Mat* subimage) {
	MarkSubimageWithCylinderSurface(rasterized_subimage, camera_model, cylinder, cv::Scalar(0, 255, 0), subimage);
	MarkSubimageWithCylinderAxis(rasterized_subimage, camera_model, cylinder, cv::Scalar(0, 0, 255), 1.0, subimage);
	MarkSubimageWithCylinderOutline(rasterized_subimage, camera_model, cylinder, cv::Scalar(255, 0, 0), 1.0, subimage);
}

void ThinStructureReconstructor::MarkSubimageWithTruncatedConeSurfaceAxisOutline(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone, cv::Mat* subimage) {
	MarkSubimageWithTruncatedConeSurface(rasterized_subimage, camera_model, truncated_cone, cv::Scalar(0, 255, 0), subimage);
	MarkSubimageWithTruncatedConeAxis(rasterized_subimage, camera_model, truncated_cone, cv::Scalar(0, 0, 255), 1.0, subimage);
	MarkSubimageWithTruncatedConeOutline(rasterized_subimage, camera_model, truncated_cone, cv::Scalar(255, 0, 0), 1.0, subimage);
}

void ThinStructureReconstructor::MarkSubimageWithLampSurface(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const EllipsePrimitive& lamp, const cv::Scalar& color, cv::Mat* subimage) {
	const Eigen::Vector3d axis_x = Eigen::Vector3d(1.0, 0.0, 0.0);
	const Eigen::Vector3d axis_y = Eigen::Vector3d(0.0, 1.0, 0.0);
	const Eigen::Vector3d axis_z = Eigen::Vector3d(0.0, 0.0, 1.0);
	const Eigen::Vector3d south_pole = lamp.center.ToEigenVector() - lamp.radius_small * axis_z;
	MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(south_pole), color, 1.0, subimage);
	const Eigen::Vector3d north_pole = lamp.center.ToEigenVector() + lamp.radius_small * axis_z;
	MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(north_pole), color, 1.0, subimage);
	for (int vertical_angle_index = -7; vertical_angle_index <= 7; ++vertical_angle_index) {
		for (int horizontal_angle_index = 0; horizontal_angle_index < 16; ++horizontal_angle_index) {
			double vertical_angle = PI / 2.0 * (double) vertical_angle_index / 8;
			double horizontal_angle = PI * 2.0 * (double) horizontal_angle_index / 16;
			const Eigen::Vector3d vertex = lamp.center.ToEigenVector()
				+ lamp.radius_small * axis_z * sin(vertical_angle)
				+ lamp.radius_large * axis_x * cos(vertical_angle) * cos(horizontal_angle)
				+ lamp.radius_large * axis_y * cos(vertical_angle) * sin(horizontal_angle);
			MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), color, 1.0, subimage);
		}
	}
}

void ThinStructureReconstructor::MarkSubimageWithLampSurfaceOutline(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const EllipsePrimitive& lamp, cv::Mat* subimage) {
	MarkSubimageWithLampSurface(rasterized_subimage, camera_model, lamp, cv::Scalar(0, 255, 0), subimage);
}

void ThinStructureReconstructor::MarkSubimageWithPoleWithLampSurfaceAxisOutline(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const PoleWithLamp& pole_with_lamp, cv::Mat* subimage) {
	MarkSubimageWithTruncatedConeSurfaceAxisOutline(rasterized_subimage, camera_model, pole_with_lamp.truncated_cone, subimage);
	if (pole_with_lamp.has_lamp) {
		MarkSubimageWithTruncatedConeSurfaceAxisOutline(rasterized_subimage, camera_model, pole_with_lamp.GetArmTruncatedCone(), subimage);
		MarkSubimageWithLampSurfaceOutline(rasterized_subimage, camera_model, pole_with_lamp.GetLampEllipse(), subimage);
	}
}

void ThinStructureReconstructor::MarkSubimageWithCylinderSurface(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const cv::Scalar& color, cv::Mat* subimage) {
	for (int sample_index = 0; sample_index <= 500; ++sample_index) {
		const Eigen::Vector3d sample_axis = (500 - sample_index) * 1.0 / 500 * cylinder.pa.ToEigenVector()
			+ sample_index * 1.0 / 500 * cylinder.pb.ToEigenVector();
		const Eigen::Vector3d dir_z = (cylinder.pb.ToEigenVector() - cylinder.pa.ToEigenVector()).normalized();
		const Eigen::Vector3d dir_x = Eigen::Vector3d(1.0, 0.0, 0.0).cross(dir_z).normalized();
		const Eigen::Vector3d dir_y = dir_z.cross(dir_x).normalized();
		for (int subdivision_index = 0; subdivision_index < 64; ++subdivision_index) {
			const double angle = subdivision_index * PI * 2.0 / 64;
			const Eigen::Vector3d vertex = sample_axis + dir_x * cylinder.r * cos(angle) + dir_y * cylinder.r * sin(angle);
			MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), color, 1.0, subimage);
		}
	}
}

void ThinStructureReconstructor::MarkSubimageWithTruncatedConeSurface(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone, const cv::Scalar& color, cv::Mat* subimage) {
	for (int sample_index = 0; sample_index <= 500; ++sample_index) {
		const Eigen::Vector3d sample_axis = (500 - sample_index) * 1.0 / 500 * truncated_cone.pa.ToEigenVector()
			+ sample_index * 1.0 / 500 * truncated_cone.pb.ToEigenVector();
		const double sample_radius = (500 - sample_index) * 1.0 / 500 * truncated_cone.ra
			+ sample_index * 1.0 / 500 * truncated_cone.rb;
		const Eigen::Vector3d dir_z = (truncated_cone.pb.ToEigenVector() - truncated_cone.pa.ToEigenVector()).normalized();
		const Eigen::Vector3d dir_x = Eigen::Vector3d(1.0, 0.0, 0.0).cross(dir_z).normalized();
		const Eigen::Vector3d dir_y = dir_z.cross(dir_x).normalized();
		for (int subdivision_index = 0; subdivision_index < 64; ++subdivision_index) {
			const double angle = subdivision_index * PI * 2.0 / 64;
			const Eigen::Vector3d vertex = sample_axis + dir_x * sample_radius * cos(angle) + dir_y * sample_radius * sin(angle);
			MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), color, 1.0, subimage);
		}
	}
}

void ThinStructureReconstructor::MarkSubimageWithCylinderAxis(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage) {
	for (int sample_index = 0; sample_index <= 500; ++sample_index) {
		const Vector3d sample_axis = (500 - sample_index) * 1.0 / 500 * cylinder.pa.ToEigenVector()
			+ sample_index * 1.0 / 500 * cylinder.pb.ToEigenVector();
		MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, sample_axis, color, radius_in_pixel, subimage);
	}
}

void ThinStructureReconstructor::MarkSubimageWithTruncatedConeAxis(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage) {
	CylinderPrimitive cylinder;
	cylinder.pa = truncated_cone.pa;
	cylinder.pb = truncated_cone.pb;
	cylinder.r = 1.0;
	MarkSubimageWithCylinderAxis(rasterized_subimage, camera_model, cylinder, color, radius_in_pixel, subimage);
}

double ThinStructureReconstructor::ComputeOutlineAngle(const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder) {
	double outline_angle = 0.0;
	double min_dot = 1e100;

	const Eigen::Vector3d axis_point = cylinder.pa.ToEigenVector();
	const Eigen::Vector2d projected_axis_point = camera_model.ProjectUtm(Vector3d(axis_point + reference_point_.ToEigenVector()), dataset_.utm_box.utm_zone).ToEigenVector();
	
	const Eigen::Vector3d other_axis_point = cylinder.pb.ToEigenVector();
	const Eigen::Vector2d projected_other_axis_point = camera_model.ProjectUtm(Vector3d(other_axis_point + reference_point_.ToEigenVector()), dataset_.utm_box.utm_zone).ToEigenVector();
	
	const Eigen::Vector2d vector_axis = (projected_other_axis_point - projected_axis_point).normalized();

	const Eigen::Vector3d dir_z = (cylinder.pb.ToEigenVector() - cylinder.pa.ToEigenVector()).normalized();
	const Eigen::Vector3d dir_x = Eigen::Vector3d(1.0, 0.0, 0.0).cross(dir_z).normalized();
	const Eigen::Vector3d dir_y = dir_z.cross(dir_x).normalized();
	for (int subdivision_index = 0; subdivision_index < 64; ++subdivision_index) {
		const double angle = subdivision_index * PI * 2.0 / 64;
		const Eigen::Vector3d surface_point = axis_point + dir_x * cos(angle) + dir_y * sin(angle);
		const Eigen::Vector2d projected_surface_point = camera_model.ProjectUtm(Vector3d(surface_point + reference_point_.ToEigenVector()), dataset_.utm_box.utm_zone).ToEigenVector();

		const Eigen::Vector2d vector_surface = (projected_surface_point - projected_axis_point).normalized();

		double dot = fabs(vector_axis.dot(vector_surface));
		if (dot < min_dot) {
			min_dot = dot;
			outline_angle = angle;
		}
	}
	return outline_angle;
}

double ThinStructureReconstructor::ComputeOutlineAngle(const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone) {
	CylinderPrimitive cylinder;
	cylinder.pa = truncated_cone.pa;
	cylinder.pb = truncated_cone.pb;
	cylinder.r = 1.0;
	return ComputeOutlineAngle(camera_model, cylinder);
}

void ThinStructureReconstructor::MarkSubimageWithCylinderOutline(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage) {
	const double outline_angle = ComputeOutlineAngle(camera_model, cylinder);
	for (int sample_index = 0; sample_index <= 500; ++sample_index) {
		const Eigen::Vector3d sample_axis = (500 - sample_index) * 1.0 / 500 * cylinder.pa.ToEigenVector()
			+ sample_index * 1.0 / 500 * cylinder.pb.ToEigenVector();
		const Eigen::Vector3d dir_z = (cylinder.pb.ToEigenVector() - cylinder.pa.ToEigenVector()).normalized();
		const Eigen::Vector3d dir_x = Eigen::Vector3d(1.0, 0.0, 0.0).cross(dir_z).normalized();
		const Eigen::Vector3d dir_y = dir_z.cross(dir_x).normalized();
		{
			const double angle = outline_angle;
			const Eigen::Vector3d vertex = sample_axis + dir_x * cylinder.r * cos(angle) + dir_y * cylinder.r * sin(angle);
			MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), color, 1.0, subimage);
		}
		{
			const double angle = outline_angle > PI ? outline_angle - PI : outline_angle + PI;
			const Eigen::Vector3d vertex = sample_axis + dir_x * cylinder.r * cos(angle) + dir_y * cylinder.r * sin(angle);
			MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), color, 1.0, subimage);
		}
	}
}

void ThinStructureReconstructor::MarkSubimageWithTruncatedConeOutline(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage) {
	const double outline_angle = ComputeOutlineAngle(camera_model, truncated_cone);
	for (int sample_index = 0; sample_index <= 500; ++sample_index) {
		const Eigen::Vector3d sample_axis = (500 - sample_index) * 1.0 / 500 * truncated_cone.pa.ToEigenVector()
			+ sample_index * 1.0 / 500 * truncated_cone.pb.ToEigenVector();
		const double sample_radius = (500 - sample_index) * 1.0 / 500 * truncated_cone.ra
			+ sample_index * 1.0 / 500 * truncated_cone.rb;
		const Eigen::Vector3d dir_z = (truncated_cone.pb.ToEigenVector() - truncated_cone.pa.ToEigenVector()).normalized();
		const Eigen::Vector3d dir_x = Eigen::Vector3d(1.0, 0.0, 0.0).cross(dir_z).normalized();
		const Eigen::Vector3d dir_y = dir_z.cross(dir_x).normalized();
		{
			const double angle = outline_angle;
			const Eigen::Vector3d vertex = sample_axis + dir_x * sample_radius * cos(angle) + dir_y * sample_radius * sin(angle);
			MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), color, 1.0, subimage);
		}
		{
			const double angle = outline_angle > PI ? outline_angle - PI : outline_angle + PI;
			const Eigen::Vector3d vertex = sample_axis + dir_x * sample_radius * cos(angle) + dir_y * sample_radius * sin(angle);
			MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), color, 1.0, subimage);
		}
	}
}

bool ThinStructureReconstructor::MarkSubimagePixel(const RasterizedSubimage& rasterized_subimage, const Vector2d& pixel, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage) {
	const Vector2i integer_pixel(round(pixel.x), round(pixel.y));
	if (rasterized_subimage.bounds.Contains(integer_pixel)) {
		const Vector2i shifted_pixel = integer_pixel.ToEigenVector() - rasterized_subimage.bounds.min_bounds.ToEigenVector();
		cv::circle(*subimage, shifted_pixel.ToCvPoint(), radius_in_pixel, color, -1);
		return true;
	} else {
		return false;
	}
}

cv::Mat ThinStructureReconstructor::ComputeVerticalEdgeMap(const cv::Mat& subimage, const int& index) {
	cv::Mat gray;
	cv::cvtColor(subimage, gray, CV_BGR2GRAY);

	cv::Mat grad_x, grad_y;
	cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

	cv::Mat abs_grad_x, abs_grad_y;
	abs_grad_x = cv::abs(grad_x);
	abs_grad_y = cv::abs(grad_y);

	cv::Mat normalized_grad_x, normalized_grad_y;
	cv::normalize(abs_grad_x, normalized_grad_x, 0.0, 1.0, cv::NORM_MINMAX);
	cv::normalize(abs_grad_y, normalized_grad_y, 0.0, 1.0, cv::NORM_MINMAX);
	//cv::imwrite(export_directory_ + NumberToString(index) + "_normalized_grad_x.png", normalized_grad_x * 255.0);
	//cv::imwrite(export_directory_ + NumberToString(index) + "_normalized_grad_y.png", normalized_grad_y * 255.0);

	cv::Mat raw_vertical_edge_response = cv::max(abs_grad_x - abs_grad_y, 0.0);
	cv::Mat vertical_edge_response;
	cv::normalize(raw_vertical_edge_response, vertical_edge_response, 0.0, 1.0, cv::NORM_MINMAX);

	return vertical_edge_response;
}

cv::Mat ThinStructureReconstructor::ComputeLampMap(const cv::Mat& subimage, const int& index) {
	cv::Mat gray;
	cv::cvtColor(subimage, gray, CV_BGR2GRAY);

	cv::Mat float_gray;
	gray.convertTo(float_gray, CV_32F);

	cv::Mat normalized_gray;
	cv::normalize(float_gray, normalized_gray, 0.0, 1.0, cv::NORM_MINMAX);

	return normalized_gray;
}

double ThinStructureReconstructor::RetrieveSubimagePixelRounding(const RasterizedSubimage& rasterized_subimage, const Vector2d& pixel, const cv::Mat& subimage) {
	const Vector2i integer_pixel(round(pixel.x), round(pixel.y));
	if (rasterized_subimage.bounds.Contains(integer_pixel)) {
		const Vector2i shifted_pixel = integer_pixel.ToEigenVector() - rasterized_subimage.bounds.min_bounds.ToEigenVector();
		return subimage.at<float>(shifted_pixel.y, shifted_pixel.x);
	} else {
		return -1.0;
	}
}

double ThinStructureReconstructor::RetrieveSubimagePixel(const RasterizedSubimage& rasterized_subimage, const Vector2d& pixel, const cv::Mat& subimage) {
	const Vector2i integer_pixel_00(floor(pixel.x), floor(pixel.y));
	const Vector2i integer_pixel_01(floor(pixel.x), floor(pixel.y) + 1);
	const Vector2i integer_pixel_10(floor(pixel.x) + 1, floor(pixel.y));
	const Vector2i integer_pixel_11(floor(pixel.x) + 1, floor(pixel.y) + 1);
	if (rasterized_subimage.bounds.Contains(integer_pixel_00)
		&& rasterized_subimage.bounds.Contains(integer_pixel_11)) {
		const Vector2i shifted_pixel_00 = integer_pixel_00.ToEigenVector() - rasterized_subimage.bounds.min_bounds.ToEigenVector();
		const Vector2i shifted_pixel_01 = integer_pixel_01.ToEigenVector() - rasterized_subimage.bounds.min_bounds.ToEigenVector();
		const Vector2i shifted_pixel_10 = integer_pixel_10.ToEigenVector() - rasterized_subimage.bounds.min_bounds.ToEigenVector();
		const Vector2i shifted_pixel_11 = integer_pixel_11.ToEigenVector() - rasterized_subimage.bounds.min_bounds.ToEigenVector();

		const float intensity_00 = subimage.at<float>(shifted_pixel_00.y, shifted_pixel_00.x);
		const float intensity_01 = subimage.at<float>(shifted_pixel_01.y, shifted_pixel_01.x);
		const float intensity_10 = subimage.at<float>(shifted_pixel_10.y, shifted_pixel_10.x);
		const float intensity_11 = subimage.at<float>(shifted_pixel_11.y, shifted_pixel_11.x);

		const double coeff_x = pixel.x - floor(pixel.x);
		const double coeff_y = pixel.y - floor(pixel.y);

		return intensity_00 * (1.0 - coeff_x) * (1.0 - coeff_y)
			 + intensity_01 * (1.0 - coeff_x) * coeff_y
			 + intensity_10 * coeff_x * (1.0 - coeff_y)
			 + intensity_11 * coeff_x * coeff_y;
	} else {
		return -1.0;
	}
}

double ThinStructureReconstructor::RetrieveSubimageWithUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& utm_point, const cv::Mat& subimage) {
	const Vector2d projected_pixel = camera_model.ProjectUtm(utm_point, dataset_.utm_box.utm_zone);
	return RetrieveSubimagePixel(rasterized_subimage, projected_pixel, subimage);
}

double ThinStructureReconstructor::RetrieveSubimageWithShiftedUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& shifted_utm_point, const cv::Mat& subimage) {
	return RetrieveSubimageWithUtmPoint(rasterized_subimage, camera_model, Vector3d(shifted_utm_point.ToEigenVector() + reference_point_.ToEigenVector()), subimage);
}

double ThinStructureReconstructor::ComputeCylinderEdgeResponse(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const cv::Mat& subimage) {
	const double outline_angle = ComputeOutlineAngle(camera_model, cylinder);
	double edge_response = 0.0;
	int valid_point_num = 0;
	for (int sample_index = 0; sample_index <= 500; ++sample_index) {
		const Eigen::Vector3d sample_axis = (500 - sample_index) * 1.0 / 500 * cylinder.pa.ToEigenVector()
			+ sample_index * 1.0 / 500 * cylinder.pb.ToEigenVector();
		const Eigen::Vector3d dir_z = (cylinder.pb.ToEigenVector() - cylinder.pa.ToEigenVector()).normalized();
		const Eigen::Vector3d dir_x = Eigen::Vector3d(1.0, 0.0, 0.0).cross(dir_z).normalized();
		const Eigen::Vector3d dir_y = dir_z.cross(dir_x).normalized();
		{
			const double angle = outline_angle;
			const Eigen::Vector3d vertex = sample_axis + dir_x * cylinder.r * cos(angle) + dir_y * cylinder.r * sin(angle);
			const double edge_value = RetrieveSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), subimage);
			if (edge_value >= 0.0) {
				edge_response += edge_value;
				++valid_point_num;
			}
		}
		{
			const double angle = outline_angle > PI ? outline_angle - PI : outline_angle + PI;
			const Eigen::Vector3d vertex = sample_axis + dir_x * cylinder.r * cos(angle) + dir_y * cylinder.r * sin(angle);
			const double edge_value = RetrieveSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), subimage);
			if (edge_value >= 0.0) {
				edge_response += edge_value;
				++valid_point_num;
			}
		}
	}
	if (valid_point_num != 0) {
		edge_response /= valid_point_num;
	}
	return edge_response;
}

double ThinStructureReconstructor::ComputeTruncatedConeEdgeResponse(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone, const cv::Mat& subimage) {
	const double outline_angle = ComputeOutlineAngle(camera_model, truncated_cone);
	double edge_response = 0.0;
	int valid_point_num = 0;
	for (int sample_index = 0; sample_index <= 500; ++sample_index) {
		const Eigen::Vector3d sample_axis = (500 - sample_index) * 1.0 / 500 * truncated_cone.pa.ToEigenVector()
			+ sample_index * 1.0 / 500 * truncated_cone.pb.ToEigenVector();
		const double sample_radius = (500 - sample_index) * 1.0 / 500 * truncated_cone.ra
			+ sample_index * 1.0 / 500 * truncated_cone.rb;
		const Eigen::Vector3d dir_z = (truncated_cone.pb.ToEigenVector() - truncated_cone.pa.ToEigenVector()).normalized();
		const Eigen::Vector3d dir_x = Eigen::Vector3d(1.0, 0.0, 0.0).cross(dir_z).normalized();
		const Eigen::Vector3d dir_y = dir_z.cross(dir_x).normalized();
		{
			const double angle = outline_angle;
			const Eigen::Vector3d vertex = sample_axis + dir_x * sample_radius * cos(angle) + dir_y * sample_radius * sin(angle);
			const double edge_value = RetrieveSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), subimage);
			if (edge_value >= 0.0) {
				edge_response += edge_value;
				++valid_point_num;
			}
		}
		{
			const double angle = outline_angle > PI ? outline_angle - PI : outline_angle + PI;
			const Eigen::Vector3d vertex = sample_axis + dir_x * sample_radius * cos(angle) + dir_y * sample_radius * sin(angle);
			const double edge_value = RetrieveSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), subimage);
			if (edge_value >= 0.0) {
				edge_response += edge_value;
				++valid_point_num;
			}
		}
	}
	if (valid_point_num != 0) {
		edge_response /= valid_point_num;
	}
	return edge_response;
}

double ThinStructureReconstructor::ComputeEllipseEdgeResponse(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const EllipsePrimitive& ellipse, const cv::Mat& subimage) {
	double edge_response = 0.0;
	int valid_point_num = 0;
	const Eigen::Vector3d axis_x = Eigen::Vector3d(1.0, 0.0, 0.0);
	const Eigen::Vector3d axis_y = Eigen::Vector3d(0.0, 1.0, 0.0);
	const Eigen::Vector3d axis_z = Eigen::Vector3d(0.0, 0.0, 1.0);
	const Eigen::Vector3d south_pole = ellipse.center.ToEigenVector() - ellipse.radius_small * axis_z;
	{
		const double edge_value = RetrieveSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(south_pole), subimage);
		if (edge_value >= 0.0) {
			edge_response += edge_value;
			++valid_point_num;
		}
	}
	const Eigen::Vector3d north_pole = ellipse.center.ToEigenVector() + ellipse.radius_small * axis_z;
	{
		const double edge_value = RetrieveSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(north_pole), subimage);
		if (edge_value >= 0.0) {
			edge_response += edge_value;
			++valid_point_num;
		}
	}
	for (int vertical_angle_index = -7; vertical_angle_index <= 7; ++vertical_angle_index) {
		for (int horizontal_angle_index = 0; horizontal_angle_index < 16; ++horizontal_angle_index) {
			double vertical_angle = PI / 2.0 * (double) vertical_angle_index / 8;
			double horizontal_angle = PI * 2.0 * (double) horizontal_angle_index / 16;
			const Eigen::Vector3d vertex = ellipse.center.ToEigenVector()
				+ ellipse.radius_small * axis_z * sin(vertical_angle)
				+ ellipse.radius_large * axis_x * cos(vertical_angle) * cos(horizontal_angle)
				+ ellipse.radius_large * axis_y * cos(vertical_angle) * sin(horizontal_angle);
			const double edge_value = RetrieveSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), subimage);
			if (edge_value >= 0.0) {
				edge_response += edge_value;
				++valid_point_num;
			}
		}
	}
	if (valid_point_num != 0) {
		edge_response /= valid_point_num;
	}
	return edge_response;
}

double ThinStructureReconstructor::ComputeTruncatedConeSumEdgeResponse(const TruncatedConePrimitive& truncated_cone) {
	const double ratio = 0.5;
	vector<double> edge_responses;
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		const ImageCamera& image_camera = cropped_image_cameras_[index];
		const cv::Mat& vertical_edge_response = cropped_edge_maps_[index];

		const double edge_response = ComputeTruncatedConeEdgeResponse(image_camera.subimage, image_camera.camera_model, truncated_cone, vertical_edge_response);
		edge_responses.push_back(edge_response);
	}
	sort(edge_responses.begin(), edge_responses.end());
	reverse(edge_responses.begin(), edge_responses.end());
	double sum_edge_response = 0.0;
	int sum_images = 0;
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		if (((double)index / cropped_image_cameras_.size() < ratio) || (edge_responses[index] >= 0.01)) {
			const double& edge_response = edge_responses[index];
			sum_edge_response += edge_response;
			++sum_images;
		}
	}
	if (sum_images == 0) {
		sum_edge_response = 0.0;
	} else {
		sum_edge_response /= sum_images;
	}
	return sum_edge_response;
}

double ThinStructureReconstructor::ComputeTruncatedConeSumEdgeResponse(const vector<TruncatedConePrimitive>& truncated_cone) {
	double sum = 0.0;
	for (int index = 0; index < truncated_cone.size(); ++index) {
		sum += ComputeTruncatedConeSumEdgeResponse(truncated_cone[index]);
	}
	return sum;
}

double ThinStructureReconstructor::ComputeTruncatedConeSumEdgeResponseDebug(const TruncatedConePrimitive& truncated_cone) {
	const double ratio = 0.5;
	vector<double> edge_responses;
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		const ImageCamera& image_camera = cropped_image_cameras_[index];
		const cv::Mat& vertical_edge_response = cropped_edge_maps_[index];

		const double edge_response = ComputeTruncatedConeEdgeResponse(image_camera.subimage, image_camera.camera_model, truncated_cone, vertical_edge_response);
		edge_responses.push_back(edge_response);
	}
	sort(edge_responses.begin(), edge_responses.end());
	reverse(edge_responses.begin(), edge_responses.end());
	for (int index = 0; index < edge_responses.size(); ++index) {
		cout << "index: " << index << " edge response: " << edge_responses[index] << endl;
	}
	double sum_edge_response = 0.0;
	int sum_images = 0;
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		if (((double)index / cropped_image_cameras_.size() < ratio) || (edge_responses[index] >= 0.01)) {
			const double& edge_response = edge_responses[index];
			sum_edge_response += edge_response;
			++sum_images;
		}
	}
	if (sum_images == 0) {
		sum_edge_response = 0.0;
	} else {
		sum_edge_response /= sum_images;
	}
	return sum_edge_response;
}

double ThinStructureReconstructor::ComputeEllipseSumEdgeResponse(const EllipsePrimitive& ellipse) {
	const double ratio = 0.5;
	vector<double> edge_responses;
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		const ImageCamera& image_camera = cropped_image_cameras_[index];
		const cv::Mat& lamp_map = cropped_lamp_maps_[index];

		const double edge_response = ComputeEllipseEdgeResponse(image_camera.subimage, image_camera.camera_model, ellipse, lamp_map);
		edge_responses.push_back(edge_response);
	}
	sort(edge_responses.begin(), edge_responses.end());
	reverse(edge_responses.begin(), edge_responses.end());
	double sum_edge_response = 0.0;
	int sum_images = 0;
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		if (((double)index / cropped_image_cameras_.size() < ratio) || (edge_responses[index] >= 0.01)) {
			const double& edge_response = edge_responses[index];
			sum_edge_response += edge_response;
			++sum_images;
		}
	}
	if (sum_images == 0) {
		sum_edge_response = 0.0;
	} else {
		sum_edge_response /= sum_images;
	}
	return sum_edge_response;
}

void ThinStructureReconstructor::ComputeRawSubimagesRadiusByVoting() {
	cylinder_hypotheses_with_radii_.clear();
	ofstream out_stream(export_directory_ + "radius_by_voting.txt");
	for (int cylinder_index = 0; cylinder_index < extended_cylinder_hypotheses_.size(); ++cylinder_index) {
		const CylinderPrimitive& cylinder = extended_cylinder_hypotheses_[cylinder_index];
		map<int, int> radius_histogram;
		for (int radius_division = 1; radius_division <= 100; ++radius_division) {
			radius_histogram[radius_division] = 0;
		}
		double best_radius_histogram = 0.0;
		double best_radius = 0.0;
		for (int index = 0; index < dataset_.image_cameras.size(); ++index) {
			cout << "Computing radius for Cylinder #" << cylinder_index << ", Image #" << index << endl;
			const ImageCamera& image_camera = dataset_.image_cameras[index];
			const cv::Mat subimage = cv::imread(image_camera.subimage.file_path, CV_LOAD_IMAGE_COLOR);
			const cv::Mat vertical_edge_response = ComputeVerticalEdgeMap(subimage, index);
			cv::imwrite(export_directory_ + NumberToString(index) + "_vertical_edge_response.png", vertical_edge_response * 255.0);

			double max_response = 0.0;
			int max_radius_division = 0;

			for (int radius_division = 1; radius_division <= 100; ++radius_division) {
				const double radius = radius_division * 1.0 / 100;
				CylinderPrimitive cylinder_new = cylinder;
				cylinder_new.r = radius;

				const double edge_response = ComputeCylinderEdgeResponse(image_camera.subimage, image_camera.camera_model, cylinder_new, vertical_edge_response);
				cout << "Radius: " << radius << " edge response: " << edge_response << endl;
				if (edge_response > max_response) {
					max_response = edge_response;
					max_radius_division = radius_division;
				}
			}

			cout << "Max radius division: " << max_radius_division << endl;
			if (max_radius_division == 0) continue;

			out_stream << index << " " << max_radius_division * 1.0 / 100 << endl;

			CylinderPrimitive cylinder_new = cylinder;
			cylinder_new.r = max_radius_division * 1.0 / 100;
			cv::Mat subimage_new = subimage.clone();
			MarkSubimageWithCylinderOutline(image_camera.subimage, image_camera.camera_model, cylinder_new, cv::Scalar(255, 0, 0), 1, &subimage_new);

			cv::imwrite(export_directory_ + NumberToString(index) + "_cylinder_" + NumberToString(cylinder_index) + ".png", subimage_new);

			++radius_histogram[max_radius_division];
			if (radius_histogram[max_radius_division] > best_radius_histogram) {
				best_radius_histogram = radius_histogram[max_radius_division];
				best_radius = max_radius_division * 1.0 / 100;
			}
		}
		CylinderPrimitive cylinder_new = cylinder;
		cylinder_new.r = best_radius;
		cylinder_hypotheses_with_radii_.push_back(cylinder_new);

		cout << "Best radius: " << best_radius << endl;
	}
	out_stream.close();
	ExportCylinderPrimitives(cylinder_hypotheses_with_radii_, "cylinder_hypotheses_with_radii.dat");
	ExportCylinderMeshes(cylinder_hypotheses_with_radii_, "cylinder_hypotheses_with_radii.obj");
}

void ThinStructureReconstructor::ComputeRawSubimagesRadiusBySearching() {
	clock_t start = clock();
	cout << "Computing radius by searching" << endl;
	cylinder_hypotheses_with_radii_.clear();
	ofstream out_stream(export_directory_ + "radius_by_searching.txt");
	for (int cylinder_index = 0; cylinder_index < extended_cylinder_hypotheses_.size(); ++cylinder_index) {
		const CylinderPrimitive& cylinder = extended_cylinder_hypotheses_[cylinder_index];
		double best_sum_edge_response = 0.0;
		double best_radius;
		for (int radius_division = 1; radius_division <= 100; ++radius_division) {
			const double radius = radius_division * 1.0 / 100;
			CylinderPrimitive cylinder_new = cylinder;
			cylinder_new.r = radius;
			double sum_edge_response = 0.0;
			for (int index = 0; index < dataset_.image_cameras.size(); ++index) {
				const ImageCamera& image_camera = dataset_.image_cameras[index];
				const cv::Mat subimage = cv::imread(image_camera.subimage.file_path, CV_LOAD_IMAGE_COLOR);
				const cv::Mat vertical_edge_response = ComputeVerticalEdgeMap(subimage, index);
				cv::imwrite(export_directory_ + NumberToString(index) + "_vertical_edge_response.png", vertical_edge_response * 255.0);

				const double edge_response = ComputeCylinderEdgeResponse(image_camera.subimage, image_camera.camera_model, cylinder_new, vertical_edge_response);
				sum_edge_response += edge_response;
			}
			cout << "Radius: " << radius << " sum edge response: " << sum_edge_response << endl;
			out_stream << setprecision(2) << fixed << radius << " " << sum_edge_response << endl;
			if (sum_edge_response > best_sum_edge_response) {
				best_sum_edge_response = sum_edge_response;
				best_radius = radius;
			}
		}			
		CylinderPrimitive cylinder_new = cylinder;
		cylinder_new.r = best_radius;
		cylinder_hypotheses_with_radii_.push_back(cylinder_new);

		cout << "Best radius: " << best_radius << endl;
	}
	double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Time spent in computing radius by searching: " << duration << "s" << endl;
	out_stream.close();
	ExportCylinderPrimitives(cylinder_hypotheses_with_radii_, "cylinder_hypotheses_with_radii.dat");
	ExportCylinderMeshes(cylinder_hypotheses_with_radii_, "cylinder_hypotheses_with_radii.obj");
}

void ThinStructureReconstructor::TestBilinearPixelRetrieval() {
	cout << "Test bilinear pixel retrieval" << endl;
	cout << RetrieveSubimagePixel(cropped_image_cameras_[0].subimage, Vector2d(98.3 + cropped_image_cameras_[0].subimage.bounds.min_bounds.x, 275.3 + cropped_image_cameras_[0].subimage.bounds.min_bounds.y), cropped_edge_maps_[0]) << endl;
	cout << RetrieveSubimagePixelRounding(cropped_image_cameras_[0].subimage, Vector2d(98.3 + cropped_image_cameras_[0].subimage.bounds.min_bounds.x, 275.3 + cropped_image_cameras_[0].subimage.bounds.min_bounds.y), cropped_edge_maps_[0]) << endl;
	cout << endl;
	cout << RetrieveSubimagePixel(cropped_image_cameras_[0].subimage, Vector2d(98.3 + cropped_image_cameras_[0].subimage.bounds.min_bounds.x, 275.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.y), cropped_edge_maps_[0]) << endl;
	cout << RetrieveSubimagePixelRounding(cropped_image_cameras_[0].subimage, Vector2d(98.3 + cropped_image_cameras_[0].subimage.bounds.min_bounds.x, 275.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.y), cropped_edge_maps_[0]) << endl;
	cout << endl;
	cout << RetrieveSubimagePixel(cropped_image_cameras_[0].subimage, Vector2d(98.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.x, 275.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.y), cropped_edge_maps_[0]) << endl;
	cout << RetrieveSubimagePixelRounding(cropped_image_cameras_[0].subimage, Vector2d(98.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.x, 275.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.y), cropped_edge_maps_[0]) << endl;
	cout << endl;
	cout << RetrieveSubimagePixel(cropped_image_cameras_[0].subimage, Vector2d(99.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.x, 275.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.y), cropped_edge_maps_[0]) << endl;
	cout << RetrieveSubimagePixelRounding(cropped_image_cameras_[0].subimage, Vector2d(99.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.x, 275.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.y), cropped_edge_maps_[0]) << endl;
	cout << endl;
	cout << RetrieveSubimagePixel(cropped_image_cameras_[0].subimage, Vector2d(98.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.x, 276.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.y), cropped_edge_maps_[0]) << endl;
	cout << RetrieveSubimagePixelRounding(cropped_image_cameras_[0].subimage, Vector2d(98.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.x, 276.0 + cropped_image_cameras_[0].subimage.bounds.min_bounds.y), cropped_edge_maps_[0]) << endl;
	cout << endl;
}

void ThinStructureReconstructor::ComputeCroppedSubimageVerticalEdgeMaps() {
	cout << "Computing vertical edge maps on cropped subimages" << endl;
	cropped_edge_maps_.clear();
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		const cv::Mat& subimage = cropped_subimages_[index];
		const cv::Mat vertical_edge_response = ComputeVerticalEdgeMap(subimage, index);
		//cv::imwrite(export_directory_ + NumberToString(index) + "_vertical_edge_response.png", vertical_edge_response * 255.0);
		cropped_edge_maps_.push_back(vertical_edge_response);
	}
}

void ThinStructureReconstructor::ComputeCroppedSubimageLampMaps() {
	cout << "Computing lamp maps on cropped subimages" << endl;
	cropped_lamp_maps_.clear();
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		const cv::Mat& subimage = cropped_subimages_[index];
		const cv::Mat lamp_map = ComputeLampMap(subimage, index);
		//cv::imwrite(export_directory_ + NumberToString(index) + "_lamp_map.png", lamp_map * 255.0);
		cropped_lamp_maps_.push_back(lamp_map);
	}
}

void ThinStructureReconstructor::ComputeCroppedSubimagesRadiusBySearching() {
	clock_t start = clock();
	cout << "Computing radius by searching" << endl;
	cylinder_hypotheses_with_radii_.clear();
	//TestBilinearPixelRetrieval();
	ofstream out_stream(export_directory_ + "radius_by_searching_cropped_subimages.txt");
	ofstream out_stream_vector(export_directory_ + "radius_vector_by_searching_cropped_subimages.txt");
	for (int cylinder_index = 0; cylinder_index < extended_cylinder_hypotheses_.size(); ++cylinder_index) {
		const CylinderPrimitive& cylinder = extended_cylinder_hypotheses_[cylinder_index];
		double best_sum_edge_response = 0.0;
		double best_radius;
		for (int radius_division = 1; radius_division <= 100; ++radius_division) {
			const double radius = radius_division * 1.0 / 100;
			CylinderPrimitive cylinder_new = cylinder;
			cylinder_new.r = radius;
			double sum_edge_response = 0.0;
			for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
				const ImageCamera& image_camera = cropped_image_cameras_[index];
				const cv::Mat& vertical_edge_response = cropped_edge_maps_[index];

				const double edge_response = ComputeCylinderEdgeResponse(image_camera.subimage, image_camera.camera_model, cylinder_new, vertical_edge_response);
				sum_edge_response += edge_response;
			}
			cout << "Radius: " << radius << " sum edge response: " << sum_edge_response << endl;
			out_stream << setprecision(2) << fixed << radius << " " << sum_edge_response << endl;
			out_stream_vector << setprecision(2) << fixed << sum_edge_response << " ";
			if (sum_edge_response > best_sum_edge_response) {
				best_sum_edge_response = sum_edge_response;
				best_radius = radius;
			}
		}			
		out_stream_vector << endl;

		CylinderPrimitive cylinder_new = cylinder;
		cylinder_new.r = best_radius;
		cylinder_hypotheses_with_radii_.push_back(cylinder_new);

		cout << "Best radius: " << best_radius << endl;
	}
	double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Time spent in computing radius by searching: " << duration << "s" << endl;
	out_stream.close();
	out_stream_vector.close();
	ExportCylinderPrimitives(cylinder_hypotheses_with_radii_, "cylinder_hypotheses_with_radii.dat");
	ExportCylinderMeshes(cylinder_hypotheses_with_radii_, "cylinder_hypotheses_with_radii.obj");
}

void ThinStructureReconstructor::ComputeCroppedSubimageTruncatedCones() {
	clock_t start = clock();
	cout << "Computing radii of truncated cones" << endl;
	truncated_cone_hypotheses_with_radii_.clear();
	ofstream out_stream(export_directory_ + "radii_truncated_cones_by_searching_cropped_subimages.txt");
	ofstream out_stream_matrix(export_directory_ + "radii_matrix_truncated_cones_by_searching_cropped_subimages.txt");
	for (int cylinder_index = 0; cylinder_index < extended_cylinder_hypotheses_.size(); ++cylinder_index) {
		out_stream << "<" << cylinder_index << ">" << endl;
		out_stream_matrix << "<" << cylinder_index << ">" << endl;
		const CylinderPrimitive& cylinder = extended_cylinder_hypotheses_[cylinder_index];
		double best_sum_edge_response = 0.0;
		TruncatedConePrimitive best_truncated_cone;
		for (int radius_a_division = 1; radius_a_division <= 100; ++radius_a_division) {
			const double radius_a = radius_a_division * 1.0 / 100;
			for (int radius_b_division = 1; radius_b_division <= 100; ++radius_b_division) {
				const double radius_b = radius_b_division * 1.0 / 100;
				TruncatedConePrimitive truncated_cone(cylinder);
				truncated_cone.ra = radius_a;
				truncated_cone.rb = radius_b;
				double sum_edge_response = ComputeTruncatedConeSumEdgeResponse(truncated_cone);
				cout << "Radius (buttom): " << radius_a << " radius (top): " << radius_b << " sum edge response: " << sum_edge_response << endl;
				out_stream << setprecision(2) << fixed << radius_a << " " << radius_b << " " << sum_edge_response << endl;
				out_stream_matrix << setprecision(2) << fixed << sum_edge_response << " ";
				if (sum_edge_response > best_sum_edge_response) {
					best_sum_edge_response = sum_edge_response;
					best_truncated_cone = truncated_cone;
				}
			}
			out_stream_matrix << endl;
		}			
		truncated_cone_hypotheses_with_radii_.push_back(best_truncated_cone);

		cout << "Best radius (buttom): " << best_truncated_cone.ra << " best radius (top): " << best_truncated_cone.rb << endl;
		out_stream << endl;
		out_stream_matrix << endl;
	}
	double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Time spent in computing radii of truncated cones: " << duration << "s" << endl;
	out_stream.close();
	out_stream_matrix.close();
	ExportTruncatedConePrimitives(truncated_cone_hypotheses_with_radii_, "truncated_cone_hypotheses_with_radii.dat");
	ExportTruncatedConeMeshes(truncated_cone_hypotheses_with_radii_, "truncated_cone_hypotheses_with_radii.obj");
	ExportCroppedSubimagesWithMarkedTruncatedCones(truncated_cone_hypotheses_with_radii_, "_marked_truncated_cones_with_radii");
}

void ThinStructureReconstructor::LoadTruncatedConesWithRadii() {
	cout << "Loading truncated cones with radii" << endl;
	truncated_cone_hypotheses_with_radii_ = ImportTruncatedConePrimitives("truncated_cone_hypotheses_with_radii.dat");
}

void ThinStructureReconstructor::ExportCroppedSubimagesWithMarkedTruncatedCones(const vector<TruncatedConePrimitive>& truncated_cones, const string& file_name) {
	cout << "About to export " << cropped_image_cameras_.size() << " images." << endl;
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		cout << "Exporting image #" << index << endl;
		const ImageCamera& image_camera = cropped_image_cameras_[index];
		cv::Mat subimage = cropped_subimages_[index].clone();
		for (int truncated_cone_index = 0; truncated_cone_index < truncated_cones.size(); ++truncated_cone_index) {
			const TruncatedConePrimitive& truncated_cone = truncated_cones[truncated_cone_index];
			MarkSubimageWithTruncatedConeSurfaceAxisOutline(image_camera.subimage, image_camera.camera_model, truncated_cone, &subimage);
		}
		cv::imwrite(export_directory_ + NumberToString(index) + file_name + ".png", subimage);
	}
}

void ThinStructureReconstructor::ExportCroppedSubimagesWithMarkedPoleWithLamps(const vector<PoleWithLamp>& pole_with_lamps, const string& file_name) {
	cout << "About to export " << cropped_image_cameras_.size() << " images." << endl;
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		cout << "Exporting image #" << index << endl;
		const ImageCamera& image_camera = cropped_image_cameras_[index];
		cv::Mat subimage = cropped_subimages_[index].clone();
		for (int pole_with_lamp_index = 0; pole_with_lamp_index < pole_with_lamps.size(); ++pole_with_lamp_index) {
			const PoleWithLamp& pole_with_lamp = pole_with_lamps[pole_with_lamp_index];
			MarkSubimageWithPoleWithLampSurfaceAxisOutline(image_camera.subimage, image_camera.camera_model, pole_with_lamp, &subimage);
		}
		cv::imwrite(export_directory_ + NumberToString(index) + file_name + ".png", subimage);
	}
}

bool ThinStructureReconstructor::FindBestNeighborRadiiOffsets(const TruncatedConePrimitive& truncated_cone, const TruncatedConePrimitive& original_truncated_cone, TruncatedConePrimitive* best_neighbor_truncated_cone) {
	clock_t start = clock();
	double best_sum_edge_response = 0.0;
	double is_self_best = false;
	for (int delta_ra_division = -1; delta_ra_division <= 1; ++delta_ra_division) {
		for (int delta_rb_division = -1; delta_rb_division <= 1; ++delta_rb_division) {
			for (int delta_offset_pa_x_division = -1; delta_offset_pa_x_division <= 1; ++delta_offset_pa_x_division) {
				for (int delta_offset_pa_y_division = -1; delta_offset_pa_y_division <= 1; ++delta_offset_pa_y_division) {
					for (int delta_offset_pb_x_division = -1; delta_offset_pb_x_division <= 1; ++delta_offset_pb_x_division) {
						for (int delta_offset_pb_y_division = -1; delta_offset_pb_y_division <= 1; ++delta_offset_pb_y_division) {
							double delta_ra = delta_ra_division * 1.0 / 100;
							double delta_rb = delta_rb_division * 1.0 / 100;
							double delta_offset_pa_x = delta_offset_pa_x_division * 1.0 / 100;
							double delta_offset_pa_y = delta_offset_pa_y_division * 1.0 / 100;
							double delta_offset_pb_x = delta_offset_pb_x_division * 1.0 / 100;
							double delta_offset_pb_y = delta_offset_pb_y_division * 1.0 / 100;
							TruncatedConePrimitive neighbor_truncated_cone = truncated_cone;
							neighbor_truncated_cone.ra += delta_ra;
							neighbor_truncated_cone.rb += delta_rb;
							neighbor_truncated_cone.pa.x += delta_offset_pa_x;
							neighbor_truncated_cone.pa.y += delta_offset_pa_y;
							neighbor_truncated_cone.pb.x += delta_offset_pb_x;
							neighbor_truncated_cone.pb.y += delta_offset_pb_y;
							double sum_edge_response = ComputeTruncatedConeSumEdgeResponse(neighbor_truncated_cone);
							if (sum_edge_response > best_sum_edge_response) {
								best_sum_edge_response = sum_edge_response;
								*best_neighbor_truncated_cone = neighbor_truncated_cone;

								if (delta_ra_division == 0
									&& delta_rb_division == 0
									&& delta_offset_pa_x_division == 0
									&& delta_offset_pa_y_division == 0
									&& delta_offset_pb_x_division == 0
									&& delta_offset_pb_y_division == 0) {
									is_self_best = true;
								} else {
									is_self_best = false;
								}
							}
						}
					}
				}
			}
		}
	}
	double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Duration of this iteration: " << duration << "s" << endl;
	cout << setprecision(3) << fixed;
	cout << best_neighbor_truncated_cone->pa.x << " " << best_neighbor_truncated_cone->pa.y << " ";
	cout << best_neighbor_truncated_cone->pb.x << " " << best_neighbor_truncated_cone->pb.y << " ";
	cout << best_neighbor_truncated_cone->ra << " " << best_neighbor_truncated_cone->rb << endl;
	return !is_self_best;
}

void ThreadHelperFunc(ThinStructureReconstructor* thin_structure_reconstructor, TruncatedConePrimitive* truncated_cone, double* result) {
	if (truncated_cone->ra <= 0.05 + 1e-3 || truncated_cone->rb <= 0.05 + 1e-3) {
		*result = 0.0;
	} else {
		*result = thin_structure_reconstructor->ComputeTruncatedConeSumEdgeResponse(*truncated_cone);
	}
}

bool ThinStructureReconstructor::FindBestNeighborRadiiOffsetsMultiThreading(const TruncatedConePrimitive& truncated_cone, const TruncatedConePrimitive& original_truncated_cone, TruncatedConePrimitive* best_neighbor_truncated_cone, double* best_sum_edge_response) {
	clock_t start = clock();
	const int num_neighbors = 3*3*3*3*3*3;
	vector<boost::thread*> threads(num_neighbors, NULL);
	vector<double> results(num_neighbors, 0.0);
	vector<TruncatedConePrimitive> neighbors(num_neighbors);
	int index = 0;
	for (int delta_ra_division = -1; delta_ra_division <= 1; ++delta_ra_division) {
		for (int delta_rb_division = -1; delta_rb_division <= 1; ++delta_rb_division) {
			for (int delta_offset_pa_x_division = -1; delta_offset_pa_x_division <= 1; ++delta_offset_pa_x_division) {
				for (int delta_offset_pa_y_division = -1; delta_offset_pa_y_division <= 1; ++delta_offset_pa_y_division) {
					for (int delta_offset_pb_x_division = -1; delta_offset_pb_x_division <= 1; ++delta_offset_pb_x_division) {
						for (int delta_offset_pb_y_division = -1; delta_offset_pb_y_division <= 1; ++delta_offset_pb_y_division) {
							const double delta_ra = delta_ra_division * 2.0 / 100;
							const double delta_rb = delta_rb_division * 2.0 / 100;
							const double delta_offset_pa_x = delta_offset_pa_x_division * 2.0 / 100;
							const double delta_offset_pa_y = delta_offset_pa_y_division * 2.0 / 100;
							const double delta_offset_pb_x = delta_offset_pb_x_division * 2.0 / 100;
							const double delta_offset_pb_y = delta_offset_pb_y_division * 2.0 / 100;
							TruncatedConePrimitive neighbor_truncated_cone = truncated_cone;
							neighbor_truncated_cone.ra += delta_ra;
							neighbor_truncated_cone.rb += delta_rb;
							neighbor_truncated_cone.pa.x += delta_offset_pa_x;
							neighbor_truncated_cone.pa.y += delta_offset_pa_y;
							neighbor_truncated_cone.pb.x += delta_offset_pb_x;
							neighbor_truncated_cone.pb.y += delta_offset_pb_y;
							neighbors[index] = neighbor_truncated_cone;
							++index;
						}
					}
				}
			}
		}
	}
	for (int index = 0; index < num_neighbors; ++index) {
		threads[index] = new boost::thread(ThreadHelperFunc, this, &(neighbors[index]), &(results[index]));
	}
	*best_sum_edge_response = 0.0;
	double is_local_maxima = false;
	for (int index = 0; index < num_neighbors; ++index) {
		threads[index]->join();
		delete threads[index];
		const double sum_edge_response = results[index];
		if (sum_edge_response > *best_sum_edge_response) {
			*best_sum_edge_response = sum_edge_response;
			*best_neighbor_truncated_cone = neighbors[index];
			if (index == num_neighbors / 2) {
				is_local_maxima = true;
			} else {
				is_local_maxima = false;
			}
		}
	}
	double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Duration of this iteration: " << duration << "s" << endl;
	cout << setprecision(3) << fixed;
	cout << best_neighbor_truncated_cone->pa.x << " " << best_neighbor_truncated_cone->pa.y << " ";
	cout << best_neighbor_truncated_cone->pb.x << " " << best_neighbor_truncated_cone->pb.y << " ";
	cout << best_neighbor_truncated_cone->ra << " " << best_neighbor_truncated_cone->rb << endl;
	return !is_local_maxima;
}

void ThinStructureReconstructor::ComputeCroppedSubimageTruncatedConesWithOffsets() {
	vector<pair<double, double> > radii_initials;
	radii_initials.push_back(make_pair(0.1, 0.1));
	radii_initials.push_back(make_pair(0.2, 0.2));

	clock_t start = clock();
	cout << "Computing radii and offsets of truncated cones" << endl;
	truncated_cone_hypotheses_with_radii_offsets_.clear();
	for (int cylinder_index = 0; cylinder_index < extended_cylinder_hypotheses_.size(); ++cylinder_index) {
		double best_sum_edge_response = 0.0;
		TruncatedConePrimitive best_truncated_cone;
		for (int initial_index = 0; initial_index < radii_initials.size(); ++initial_index) {
			const CylinderPrimitive& cylinder = extended_cylinder_hypotheses_[cylinder_index];
			TruncatedConePrimitive truncated_cone(cylinder);
			truncated_cone.ra = radii_initials[initial_index].first;
			truncated_cone.rb = radii_initials[initial_index].second;
			const TruncatedConePrimitive original_truncated_cone = truncated_cone;
			TruncatedConePrimitive current_neighbor_truncated_cone;
			double current_sum_edge_response;
			int iteration_times = 1;
			cout << "iteration #" << iteration_times << endl;
			while (FindBestNeighborRadiiOffsetsMultiThreading(truncated_cone, original_truncated_cone, &current_neighbor_truncated_cone, &current_sum_edge_response)) {
				truncated_cone = current_neighbor_truncated_cone;
				iteration_times ++;
				cout << "iteration #" << iteration_times << endl;
			}	
			cout << "Adjusted radius (buttom): " << truncated_cone.ra << " radius (top): " << truncated_cone.rb << endl;
			cout << "Adjusted sum edge response: " << current_sum_edge_response << endl;
			if (current_sum_edge_response > best_sum_edge_response) {
				best_sum_edge_response = current_sum_edge_response;
				best_truncated_cone = truncated_cone;
			}
		}
		truncated_cone_hypotheses_with_radii_offsets_.push_back(best_truncated_cone);
		cout << "Best radius (buttom): " << best_truncated_cone.ra << " radius (top): " << best_truncated_cone.rb << endl;
		cout << "Best sum edge response: " << best_sum_edge_response << endl;
	}
	double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Time spent in computing radii and offsets of truncated cones: " << duration << "s" << endl;
	ExportTruncatedConePrimitives(truncated_cone_hypotheses_with_radii_offsets_, "truncated_cone_hypotheses_with_radii_offsets.dat");
	ExportTruncatedConeMeshes(truncated_cone_hypotheses_with_radii_offsets_, "truncated_cone_hypotheses_with_radii_offsets.obj");
	ExportCroppedSubimagesWithMarkedTruncatedCones(truncated_cone_hypotheses_with_radii_offsets_, "_marked_truncated_cones_with_radii_offsets");
}

void ThinStructureReconstructor::LoadTruncatedConesWithRadiiOffsets() {
	cout << "Loading truncated cones with radii and offsets" << endl;
	truncated_cone_hypotheses_with_radii_offsets_ = ImportTruncatedConePrimitives("truncated_cone_hypotheses_with_radii_offsets.dat");
}

bool ThinStructureReconstructor::ComputeExtentsFromCroppedSubimages(const TruncatedConePrimitive& truncated_cone, TruncatedConePrimitive* truncated_cone_extents) {
	const int num_axial_sections = max(1, (int) ((truncated_cone.pa.ToEigenVector() - truncated_cone.pb.ToEigenVector()).norm() * 10));
	cout << "num axial sections: " << num_axial_sections << endl;
	vector<TruncatedConePrimitive> truncated_cone_axial_sections(num_axial_sections);
	vector<double> sum_edge_responses_truncated_cone_axial_sections(num_axial_sections);
	int min_index = -1;
	int max_index = -1;
	for (int index_axial_section = 0; index_axial_section < num_axial_sections; ++index_axial_section) {
		const double alpha_a = index_axial_section * 1.0 / num_axial_sections;
		const double alpha_b = (index_axial_section + 1) * 1.0 / num_axial_sections;
		truncated_cone_axial_sections[index_axial_section].pa = Vector3d((1.0 - alpha_a) * truncated_cone.pa.ToEigenVector() + alpha_a * truncated_cone.pb.ToEigenVector());
		truncated_cone_axial_sections[index_axial_section].pb = Vector3d((1.0 - alpha_b) * truncated_cone.pa.ToEigenVector() + alpha_b * truncated_cone.pb.ToEigenVector());
		truncated_cone_axial_sections[index_axial_section].ra = (1.0 - alpha_a) * truncated_cone.ra + alpha_a * truncated_cone.rb;
		truncated_cone_axial_sections[index_axial_section].rb = (1.0 - alpha_b) * truncated_cone.ra + alpha_b * truncated_cone.rb;
		sum_edge_responses_truncated_cone_axial_sections[index_axial_section] = ComputeTruncatedConeSumEdgeResponse(truncated_cone_axial_sections[index_axial_section]);
		cout << "index axial section: " << index_axial_section << " sum edge response: " << sum_edge_responses_truncated_cone_axial_sections[index_axial_section] << endl;
		if (sum_edge_responses_truncated_cone_axial_sections[index_axial_section] >= 0.1 && (min_index < 0 || index_axial_section < min_index)) {
			min_index = index_axial_section;
		}
		if (sum_edge_responses_truncated_cone_axial_sections[index_axial_section] >= 0.1 && (max_index < 0 || index_axial_section > max_index)) {
			max_index = index_axial_section;
		}
	}
	cout << "min index: " << min_index << " max index: " << max_index << endl;
	if (min_index < 0 || max_index < 0) return false;
	const double alpha_a = min_index * 1.0 / num_axial_sections;
	const double alpha_b = (max_index + 1) * 1.0 / num_axial_sections;
	truncated_cone_extents->pa = Vector3d((1.0 - alpha_a) * truncated_cone.pa.ToEigenVector() + alpha_a * truncated_cone.pb.ToEigenVector());
	truncated_cone_extents->pb = Vector3d((1.0 - alpha_b) * truncated_cone.pa.ToEigenVector() + alpha_b * truncated_cone.pb.ToEigenVector());
	truncated_cone_extents->ra = (1.0 - alpha_a) * truncated_cone.ra + alpha_a * truncated_cone.rb;
	truncated_cone_extents->rb = (1.0 - alpha_b) * truncated_cone.ra + alpha_b * truncated_cone.rb;
	return true;
}

bool ThinStructureReconstructor::ComputeExtentsFromCroppedSubimages(const TruncatedConePrimitive& truncated_cone, vector<TruncatedConePrimitive>* truncated_cone_extents) {
	const int num_axial_sections = max(1, (int) ((truncated_cone.pa.ToEigenVector() - truncated_cone.pb.ToEigenVector()).norm() * 10));
	cout << "num axial sections: " << num_axial_sections << endl;
	vector<TruncatedConePrimitive> truncated_cone_axial_sections(num_axial_sections);
	vector<double> sum_edge_responses_truncated_cone_axial_sections(num_axial_sections);
	vector<int> valid_truncated_cone_axial_sections(num_axial_sections);
	for (int index_axial_section = 0; index_axial_section < num_axial_sections; ++index_axial_section) {
		const double alpha_a = index_axial_section * 1.0 / num_axial_sections;
		const double alpha_b = (index_axial_section + 1) * 1.0 / num_axial_sections;
		truncated_cone_axial_sections[index_axial_section].pa = Vector3d((1.0 - alpha_a) * truncated_cone.pa.ToEigenVector() + alpha_a * truncated_cone.pb.ToEigenVector());
		truncated_cone_axial_sections[index_axial_section].pb = Vector3d((1.0 - alpha_b) * truncated_cone.pa.ToEigenVector() + alpha_b * truncated_cone.pb.ToEigenVector());
		truncated_cone_axial_sections[index_axial_section].ra = (1.0 - alpha_a) * truncated_cone.ra + alpha_a * truncated_cone.rb;
		truncated_cone_axial_sections[index_axial_section].rb = (1.0 - alpha_b) * truncated_cone.ra + alpha_b * truncated_cone.rb;
		sum_edge_responses_truncated_cone_axial_sections[index_axial_section] = ComputeTruncatedConeSumEdgeResponse(truncated_cone_axial_sections[index_axial_section]);
		cout << "index axial section: " << index_axial_section << " sum edge response: " << sum_edge_responses_truncated_cone_axial_sections[index_axial_section] << endl;
		if (sum_edge_responses_truncated_cone_axial_sections[index_axial_section] >= 0.1) {
			valid_truncated_cone_axial_sections[index_axial_section] = 1;
		} else {
			valid_truncated_cone_axial_sections[index_axial_section] = 0;
		}
	}
	// Collect intervals
	vector<pair<int, int> > intervals;
	int interval_start = -1;
	int interval_end = -1;
	for (int index_axial_section = 0; index_axial_section < num_axial_sections; ++index_axial_section) {
		if (valid_truncated_cone_axial_sections[index_axial_section] == 0) {
			if (interval_start >= 0) {
				intervals.push_back(make_pair(interval_start, interval_end));
			}
			interval_start = -1;
			interval_end = -1;
		} else {
			if (interval_start < 0) {
				interval_start = index_axial_section;
			}
			interval_end = index_axial_section;
		}
	}
	if (interval_start >= 0) {
		intervals.push_back(make_pair(interval_start, interval_end));
	}
	// Close gaps
	{
		int index = 0;
		while (index + 1 < intervals.size()) {
			if (intervals[index + 1].first - intervals[index].second < 80) {
				intervals[index].second = intervals[index + 1].second;
				intervals.erase(intervals.begin() + index + 1);
			} else {
				++index;
			}
		}
	}
	// Remove clutters
	{
		int index = 0;
		while (index < intervals.size()) {
			if (intervals[index].second - intervals[index].first < 20) {
				intervals.erase(intervals.begin() + index);
			} else {
				++index;
			}
		}
	}
	// Generate output
	for (int index = 0; index < intervals.size(); ++index) {
		TruncatedConePrimitive truncated_cone_extent;
		const double alpha_a = intervals[index].first * 1.0 / num_axial_sections;
		const double alpha_b = (intervals[index].second + 1) * 1.0 / num_axial_sections;
		truncated_cone_extent.pa = Vector3d((1.0 - alpha_a) * truncated_cone.pa.ToEigenVector() + alpha_a * truncated_cone.pb.ToEigenVector());
		truncated_cone_extent.pb = Vector3d((1.0 - alpha_b) * truncated_cone.pa.ToEigenVector() + alpha_b * truncated_cone.pb.ToEigenVector());
		truncated_cone_extent.ra = (1.0 - alpha_a) * truncated_cone.ra + alpha_a * truncated_cone.rb;
		truncated_cone_extent.rb = (1.0 - alpha_b) * truncated_cone.ra + alpha_b * truncated_cone.rb;
		truncated_cone_extents->push_back(truncated_cone_extent);
	}

	return !truncated_cone_extents->empty();
}

void ThinStructureReconstructor::ComputeCroppedSubimageTruncatedConesExtents() {
	cout << "Computing extents of truncated cones" << endl;
	truncated_cone_hypotheses_with_radii_offsets_extents_.clear();
	for (int truncated_cone_index = 0; truncated_cone_index < truncated_cone_hypotheses_with_radii_offsets_.size(); ++truncated_cone_index) {
		const TruncatedConePrimitive& truncated_cone = truncated_cone_hypotheses_with_radii_offsets_[truncated_cone_index];
		TruncatedConePrimitive truncated_cone_extents;
		if (ComputeExtentsFromCroppedSubimages(truncated_cone, &truncated_cone_extents)) {
			truncated_cone_hypotheses_with_radii_offsets_extents_.push_back(truncated_cone_extents);
		}
	}
	ExportTruncatedConePrimitives(truncated_cone_hypotheses_with_radii_offsets_extents_, "truncated_cone_hypotheses_with_radii_offsets_extents.dat");
	ExportTruncatedConeMeshes(truncated_cone_hypotheses_with_radii_offsets_extents_, "truncated_cone_hypotheses_with_radii_offsets_extents.obj");
	ExportCroppedSubimagesWithMarkedTruncatedCones(truncated_cone_hypotheses_with_radii_offsets_extents_, "_marked_truncated_cones_with_radii_offsets_extents");
}

void ThinStructureReconstructor::LoadTruncatedConesWithRadiiOffsetsExtents() {
	cout << "Loading truncated cones with radii, offsets and extents" << endl;
	truncated_cone_hypotheses_with_radii_offsets_extents_ = ImportTruncatedConePrimitives("truncated_cone_hypotheses_with_radii_offsets_extents.dat");
	cout << "Loaded " << truncated_cone_hypotheses_with_radii_offsets_extents_.size()
		 << " truncated cones with radii, offsets and extents" << endl;
}

void ThinStructureReconstructor::ComputeCroppedSubimageTruncatedConesWithOffsetsExtents() {
	vector<pair<double, double> > radii_initials;
	radii_initials.push_back(make_pair(0.1, 0.1));
	radii_initials.push_back(make_pair(0.2, 0.2));

	clock_t start = clock();
	cout << "Computing radii, offsets and extents of truncated cones" << endl;
	truncated_cone_hypotheses_with_radii_offsets_extents_.clear();
	for (int cylinder_index = 0; cylinder_index < extended_cylinder_hypotheses_.size(); ++cylinder_index) {
		vector<TruncatedConePrimitive> best_truncated_cone_extent;
		double best_truncated_cone_edge_response = 0.0;
		for (int initial_index = 0; initial_index < radii_initials.size(); ++initial_index) {
			const CylinderPrimitive& cylinder = extended_cylinder_hypotheses_[cylinder_index];
			TruncatedConePrimitive truncated_cone(cylinder);
			truncated_cone.ra = radii_initials[initial_index].first;
			truncated_cone.rb = radii_initials[initial_index].second;
			const TruncatedConePrimitive original_truncated_cone = truncated_cone;
			TruncatedConePrimitive current_neighbor_truncated_cone;
			double current_sum_edge_response;
			int iteration_times = 1;
			cout << "iteration #" << iteration_times << endl;
			while (FindBestNeighborRadiiOffsetsMultiThreading(truncated_cone, original_truncated_cone, &current_neighbor_truncated_cone, &current_sum_edge_response)) {
				truncated_cone = current_neighbor_truncated_cone;
				iteration_times ++;
				cout << "iteration #" << iteration_times << endl;
			}
			cout << "Adjusted end point A: " << truncated_cone.pa.x << " " << truncated_cone.pa.y << " " << truncated_cone.pa.z << endl;
			cout << "Adjusted end point B: " << truncated_cone.pb.x << " " << truncated_cone.pb.y << " " << truncated_cone.pb.z << endl;
			cout << "Adjusted radius (buttom): " << truncated_cone.ra << " radius (top): " << truncated_cone.rb << endl;
			cout << "Adjusted sum edge response: " << current_sum_edge_response << endl;
			ComputeTruncatedConeSumEdgeResponseDebug(truncated_cone);
			vector<TruncatedConePrimitive> truncated_cone_extent;
			if (ComputeExtentsFromCroppedSubimages(truncated_cone, &truncated_cone_extent)) {
				const double cropped_sum_edge_response = ComputeTruncatedConeSumEdgeResponse(truncated_cone_extent);
				cout << "Adjusted cropped sum edge response: " << cropped_sum_edge_response << endl;
				if (cropped_sum_edge_response > best_truncated_cone_edge_response) {
					best_truncated_cone_extent = truncated_cone_extent;
					best_truncated_cone_edge_response = cropped_sum_edge_response;
				}
			}
		}
		if (best_truncated_cone_edge_response > 0.0) {
			truncated_cone_hypotheses_with_radii_offsets_extents_.insert(
				truncated_cone_hypotheses_with_radii_offsets_extents_.end(), 
				best_truncated_cone_extent.begin(), best_truncated_cone_extent.end());
		}
		for (int index = 0; index < best_truncated_cone_extent.size(); ++index) {
			const TruncatedConePrimitive& truncated_cone = best_truncated_cone_extent[index];
			cout << "Best radius (buttom): " << truncated_cone.ra << " radius (top): " << truncated_cone.rb << endl;
		}
		cout << "Best edge response: " << best_truncated_cone_edge_response << endl;
	}
	double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Time spent in computing radii, offsets and extents of truncated cones: " << duration << "s" << endl;
	ExportTruncatedConePrimitives(truncated_cone_hypotheses_with_radii_offsets_extents_, "truncated_cone_hypotheses_with_radii_offsets_extents.dat");
	ExportTruncatedConeMeshes(truncated_cone_hypotheses_with_radii_offsets_extents_, "truncated_cone_hypotheses_with_radii_offsets_extents.obj");
	ExportCroppedSubimagesWithMarkedTruncatedCones(truncated_cone_hypotheses_with_radii_offsets_extents_, "_marked_truncated_cones_with_radii_offsets_extents");
}

void ThinStructureReconstructor::ExportTruncatdConesMeshesWithRadiiOffsetsExtents() {
	ExportTruncatedConeMeshes(truncated_cone_hypotheses_with_radii_offsets_extents_, "truncated_cone_hypotheses_with_radii_offsets_extents.obj");
}

void ThinStructureReconstructor::ExportRawSubimagesWithMarkedTruncatedConesWithRadiiOffsetsExtents() {
	ExportRawSubimagesWithMarkedTruncatedCones(truncated_cone_hypotheses_with_radii_offsets_extents_, "_marked_truncated_cones_with_radii_offsets_extents");
}

void ThinStructureReconstructor::ExportRawSubimagesWithMarkedTruncatedCones(const vector<TruncatedConePrimitive>& truncated_cones, const string& file_name) {
	for (int index = 0; index < dataset_.image_cameras.size(); ++index) {
		cout << "Loading raw subimage #" << index << endl;
		const ImageCamera& image_camera = dataset_.image_cameras[index];
		cv::Mat subimage = cv::imread(image_camera.subimage.file_path, CV_LOAD_IMAGE_COLOR);
		cout << "Exporting raw subimage #" << index << endl;
		for (int truncated_cone_index = 0; truncated_cone_index < truncated_cones.size(); ++truncated_cone_index) {
			const TruncatedConePrimitive& truncated_cone = truncated_cones[truncated_cone_index];
			MarkSubimageWithTruncatedConeSurfaceAxisOutline(image_camera.subimage, image_camera.camera_model, truncated_cone, &subimage);
		}
		cv::imwrite(export_directory_ + NumberToString(index) + file_name + ".png", subimage);
	}
}

pcl::PointCloud<pcl::PointXYZ> ThinStructureReconstructor::ComputeTruncatedConeNeighboringPoints(const TruncatedConePrimitive& truncated_cone) {
	pcl::PointCloud<pcl::PointXYZ> points;
	for (int index = 0; index < point_cloud_.points.size(); ++index) {
		if ((Eigen::Vector2d(truncated_cone.pa.x - point_cloud_.points[index].x, 
			                truncated_cone.pa.y - point_cloud_.points[index].y).norm() < 3.0
			|| Eigen::Vector2d(truncated_cone.pb.x - point_cloud_.points[index].x, 
			                truncated_cone.pb.y - point_cloud_.points[index].y).norm() < 3.0)
			&& point_cloud_.points[index].z < truncated_cone.pb.z + 1.0
			&& point_cloud_.points[index].z > truncated_cone.pa.z - 1.0) {
				points.points.push_back(point_cloud_.points[index]);
		}
	}
	points.width = points.points.size();
	points.height = 1;
	cout << "Neighboring points: " << points.points.size() << endl;
	return points;
}

pcl::PointCloud<pcl::PointXYZ> ThinStructureReconstructor::ComputeTopRegionNeighboringPoints(const pcl::PointCloud<pcl::PointXYZ>& point_cloud, const TruncatedConePrimitive& truncated_cone) {
	cout << "Computing top region neighboring points" << endl;
	pcl::PointCloud<pcl::PointXYZ> points;
	for (int index = 0; index < point_cloud.points.size(); ++index) {
		if (point_cloud.points[index].z < truncated_cone.pb.z + 1.0
			&& point_cloud.points[index].z > truncated_cone.pb.z - 1.0
			&& Eigen::Vector2d(truncated_cone.pb.x - point_cloud.points[index].x, 
			                   truncated_cone.pb.y - point_cloud.points[index].y).norm() > 0.5) {
				points.points.push_back(point_cloud.points[index]);
		}
	}
	points.width = points.points.size();
	points.height = 1;
	cout << "Top region neighboring points: " << points.points.size() << endl;
	return points;
}

PoleWithLamp ThinStructureReconstructor::ComputePoleWithLampFromTopRegionNeighboringPoints(const pcl::PointCloud<pcl::PointXYZ>& point_cloud, const TruncatedConePrimitive& truncated_cone) {
	cout << "Computing end points of top region neighboring points" << endl;
	if (point_cloud.points.size() == 0) {
		return PoleWithLamp(truncated_cone);
	}

	struct Node {
		int index;
		double angle;
		Node() {}
		Node(int ind, double ang) : index(ind), angle(ang) {}
		bool operator< (const Node& node) {
			return angle < node.angle;
		}
	};

	vector<Node> nodes;
	for (int index = 0; index < point_cloud.points.size(); ++index) {
		const pcl::PointXYZ& point = point_cloud.points[index];
		double angle = atan2(point.y - truncated_cone.pb.y, point.x - truncated_cone.pb.x);
		nodes.push_back(Node(index, angle));
	}
	sort(nodes.begin(), nodes.end());

	int max_population = -1;
	int max_index;
	int max_upper_index;
	int max_lower_index;
	for (int index = 0; index < nodes.size(); ++index) {
		int upper_index = index;
		int lower_index = index;
		while ((upper_index + 1) % nodes.size() != index
			&& AngleDiff(nodes[(upper_index + 1) % nodes.size()].angle, nodes[index].angle) < 0.2) {
			upper_index = (upper_index + 1) % nodes.size();
		}
		while ((lower_index - 1 + nodes.size()) % nodes.size() != index
			&& (lower_index - 1 + nodes.size()) % nodes.size() != upper_index
			&& AngleDiff(nodes[index].angle, nodes[(lower_index - 1 + nodes.size()) % nodes.size()].angle) < 0.2) {
			lower_index = (lower_index - 1 + nodes.size()) % nodes.size();
		}

		int current_population = IndexDiff(upper_index, lower_index, nodes.size()) + 1;
		if (current_population > max_population) {
			max_population = current_population;
			max_index = index;
			max_upper_index = upper_index;
			max_lower_index = lower_index;
		}
	}

	double angle = nodes[max_index].angle;
	double length = 0.0;
	for (int index = max_lower_index; index != max_upper_index; index = (index + 1) % nodes.size()) {
		int current_length = (Vector3d(point_cloud.points[nodes[index].index]).ToEigenVector() - 
			truncated_cone.pb.ToEigenVector()).norm();
		if (current_length > length) {
			length = current_length;
		}
	}
	cout << "angle: " << angle << endl;
	cout << "length: " << length << endl;
	Vector3d end_point = truncated_cone.pb.ToEigenVector()
		+ length * cos(angle) * Eigen::Vector3d(1.0, 0.0, 0.0)
		+ length * sin(angle) * Eigen::Vector3d(0.0, 1.0, 0.0);
	cout << "end_point: " << end_point.x << ", " << end_point.y << ", " << end_point.z << endl;
	int num_inliers = 0;
	for (int index = 0; index < point_cloud.points.size(); ++index) {
		const pcl::PointXYZ& point = point_cloud.points[index];
		if (PointToSegmentDistance(point, truncated_cone.pb, end_point) < 1.0) {
			++ num_inliers;
		}
	}
	cout << "num_inliers: " << num_inliers << endl;
	cout << "ratio: " << (double) num_inliers / point_cloud.points.size() << endl;
	if (num_inliers > point_cloud.points.size() * 0.8 && num_inliers > 600) {
		cout << "Positive" << endl;
		return PoleWithLamp(truncated_cone, end_point);
	} else {
		cout << "Negative" << endl;
		return PoleWithLamp(truncated_cone);
	}
}

void ThinStructureReconstructor::ComputePoleWithLamps() {
	pole_with_lamps_.clear();
	for (int truncated_cone_index = 0; truncated_cone_index < truncated_cone_hypotheses_with_radii_offsets_extents_.size(); ++truncated_cone_index) {
		cout << "Computing lamps for truncated cone #" << truncated_cone_index << endl;
		const TruncatedConePrimitive& truncated_cone = truncated_cone_hypotheses_with_radii_offsets_extents_[truncated_cone_index];
		pcl::PointCloud<pcl::PointXYZ> point_cloud;
		if (import_neighboring_points_) {
			point_cloud = ImportPointCloud("truncated_cone_neighboring_points_" + NumberToString(truncated_cone_index) + ".xyz");
		} else {
			point_cloud = ComputeTruncatedConeNeighboringPoints(truncated_cone);
			ExportPointCloud(point_cloud, "truncated_cone_neighboring_points_" + NumberToString(truncated_cone_index) + ".xyz");
		}
		pcl::PointCloud<pcl::PointXYZ> point_cloud_top_region = ComputeTopRegionNeighboringPoints(point_cloud, truncated_cone);
		ExportPointCloud(point_cloud_top_region, "truncated_cone_top_region_neighboring_points_" + NumberToString(truncated_cone_index) + ".xyz");
		PoleWithLamp pole_with_lamp = ComputePoleWithLampFromTopRegionNeighboringPoints(point_cloud_top_region, truncated_cone);
		pole_with_lamps_.push_back(pole_with_lamp);
	}
	ExportPoleWithLampPrimitives(pole_with_lamps_, "pole_with_lamps.dat");
	ExportPoleWithLampMeshes(pole_with_lamps_, "pole_with_lamps.obj");
	ExportCroppedSubimagesWithMarkedPoleWithLamps(pole_with_lamps_, "_marked_pole_with_lamps");
}

void ThinStructureReconstructor::ComputeAdjustedPoleWithLamps() {
	adjusted_pole_with_lamps_.clear();
	for (int pole_with_lamp_index = 0; pole_with_lamp_index < pole_with_lamps_.size(); ++pole_with_lamp_index) {
		cout << "Adjusting pole with lamp #" << pole_with_lamp_index << endl;
		const PoleWithLamp& pole_with_lamp = pole_with_lamps_[pole_with_lamp_index];
		if (!pole_with_lamp.has_lamp) {
			adjusted_pole_with_lamps_.push_back(pole_with_lamp);
		} else {
			const double length = (pole_with_lamp.truncated_cone.pb.ToEigenVector() - pole_with_lamp.end_point.ToEigenVector()).norm();

			PoleWithLamp max_pole_with_lamp = pole_with_lamp;
			double max_score = ComputeEllipseSumEdgeResponse(max_pole_with_lamp.GetLampEllipse());
			double max_length_ratio = 1.0;
			double max_delta_z = 0.0;
			cout << "original score: " << max_score << endl;

			for (double length_ratio = 0.5; length_ratio <= 1.2; length_ratio += 0.05) {
				for (double delta_z = -1.0; delta_z <= 1.0; delta_z += 0.05) {
					PoleWithLamp current_pole_with_lamp = pole_with_lamp;
					const double current_length = length * length_ratio;
					current_pole_with_lamp.end_point = (Vector3d) (
						(1.0 - length_ratio) * current_pole_with_lamp.truncated_cone.pb.ToEigenVector()
						+ length_ratio * current_pole_with_lamp.end_point.ToEigenVector());
					current_pole_with_lamp.end_point.z += delta_z;
					const double current_score = ComputeEllipseSumEdgeResponse(current_pole_with_lamp.GetLampEllipse());
					cout << "length_ratio: " << length_ratio << " delta_z: " << delta_z << " score: " << current_score << endl;
					if (current_score > max_score) {
						max_score = current_score;
						max_length_ratio = length_ratio;
						max_delta_z = delta_z;
						max_pole_with_lamp = current_pole_with_lamp;
					}
				}
			}

			cout << "max_length_ratio: " << max_length_ratio << " max_delta_z: " << max_delta_z << endl;
			adjusted_pole_with_lamps_.push_back(max_pole_with_lamp);
		}
	}
	ExportPoleWithLampPrimitives(adjusted_pole_with_lamps_, "adjusted_pole_with_lamps.dat");
	ExportPoleWithLampMeshes(adjusted_pole_with_lamps_, "adjusted_pole_with_lamps.obj");
	ExportCroppedSubimagesWithMarkedPoleWithLamps(adjusted_pole_with_lamps_, "_marked_adjusted_pole_with_lamps");
}

void ThinStructureReconstructor::LoadPoleWithLamps() {
	pole_with_lamps_ = ImportPoleWithLampPrimitives("pole_with_lamps.dat");
}

void ThinStructureReconstructor::LoadAdjustedPoleWithLamps() {
	adjusted_pole_with_lamps_ = ImportPoleWithLampPrimitives("adjusted_pole_with_lamps.dat");
}

void ThinStructureReconstructor::ExportRawSubimagesWithMarkedPoleWithLamps() {
	ExportRawSubimagesWithMarkedPoleWithLamps(pole_with_lamps_, "_marked_pole_with_lamps");
}

void ThinStructureReconstructor::ExportRawSubimagesWithMarkedAdjustedPoleWithLamps() {
	ExportRawSubimagesWithMarkedPoleWithLamps(adjusted_pole_with_lamps_, "_marked_adjusted_pole_with_lamps");
}

void ThinStructureReconstructor::ExportRawSubimagesWithMarkedPoleWithLamps(const vector<PoleWithLamp>& pole_with_lamps, const string& file_name) {
	for (int index = 0; index < dataset_.image_cameras.size(); ++index) {
		cout << "Loading raw subimage #" << index << endl;
		const ImageCamera& image_camera = dataset_.image_cameras[index];
		cv::Mat subimage = cv::imread(image_camera.subimage.file_path, CV_LOAD_IMAGE_COLOR);
		cout << "Exporting raw subimage #" << index << endl;
		for (int pole_with_lamp_index = 0; pole_with_lamp_index < pole_with_lamps.size(); ++pole_with_lamp_index) {
			const PoleWithLamp& pole_with_lamp = pole_with_lamps[pole_with_lamp_index];
			MarkSubimageWithPoleWithLampSurfaceAxisOutline(image_camera.subimage, image_camera.camera_model, pole_with_lamp, &subimage);
		}
		cv::imwrite(export_directory_ + NumberToString(index) + file_name + ".png", subimage);
	}
}

