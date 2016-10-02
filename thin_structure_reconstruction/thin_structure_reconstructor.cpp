#include "thin_structure_reconstructor.h"

#include <Eigen/Geometry>

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

const double PI = acos(-1.0);

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
			const double angle = subdivision_index * PI * 2 / 16.0;
			const Eigen::Vector3d vertex = cylinder.pa.ToEigenVector() + dir_x * cylinder.r * cos(angle) + dir_y * cylinder.r * sin(angle);
			out_stream << setprecision(8) << fixed << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << endl;
		}
		for (int subdivision_index = 0; subdivision_index < 16; ++subdivision_index) {
			const double angle = subdivision_index * PI * 2 / 16.0;
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
	cout << "Loading filtered points" << endl;
	point_cloud_filtered_ = ImportPointCloud("filtered.xyz");
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

void ThinStructureReconstructor::ComputeExtents(const pcl::PointCloud<pcl::PointXYZ>& point_cloud, const vector<int>& pointIdx, const Vector3d& axis, const Vector3d& point, double* min_dot, double* max_dot) {
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
	ComputeExtents(point_cloud, pointIdx, axis, centroid, &min_height, &max_height);
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
		if (bestPointIdx.size() < 600) {
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

void ThinStructureReconstructor::ExportRawSubimages() {
	for (int index = 0; index < dataset_.image_cameras.size(); ++index) {
		const ImageCamera& image_camera = dataset_.image_cameras[index];
		const cv::Mat raw_subimage = cv::imread(image_camera.subimage.file_path, CV_LOAD_IMAGE_COLOR);
		cv::imwrite(export_directory_ + NumberToString(index) + ".png", raw_subimage);
	}
}

void ThinStructureReconstructor::ExportSubimagesWithMarkedEcefPoint(const Vector3d& ecef_point) {
	for (int index = 0; index < dataset_.image_cameras.size(); ++index) {
		const ImageCamera& image_camera = dataset_.image_cameras[index];
		cv::Mat subimage = cv::imread(image_camera.subimage.file_path, CV_LOAD_IMAGE_COLOR);
		MarkSubimageWithEcefPoint(image_camera.subimage, image_camera.camera_model, ecef_point, &subimage);
		cv::imwrite(export_directory_ + NumberToString(index) + "_marked_point.png", subimage);
	}
}

void ThinStructureReconstructor::ExportSubimagesWithMarkedHypotheses() {
	for (int index = 0; index < dataset_.image_cameras.size(); ++index) {
		const ImageCamera& image_camera = dataset_.image_cameras[index];
		cv::Mat subimage = cv::imread(image_camera.subimage.file_path, CV_LOAD_IMAGE_COLOR);
		for (int cylinder_index = 0; cylinder_index < cylinder_hypotheses_.size(); ++cylinder_index) {
			const CylinderPrimitive& cylinder = cylinder_hypotheses_[cylinder_index];
			MarkSubimageWithCylinder(image_camera.subimage, image_camera.camera_model, cylinder, &subimage);
		}
		cv::imwrite(export_directory_ + NumberToString(index) + "_marked_cylinder.png", subimage);
	}
}

void ThinStructureReconstructor::MarkSubimageWithEcefPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& ecef_point, const int& radius_in_pixel, cv::Mat* subimage) {
	const Vector2d projected_pixel = camera_model.ProjectEcef(ecef_point);
	MarkSubimagePixel(rasterized_subimage, projected_pixel, radius_in_pixel, subimage);
}

void ThinStructureReconstructor::MarkSubimageWithUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& utm_point, const int& radius_in_pixel, cv::Mat* subimage) {
	const Vector2d projected_pixel = camera_model.ProjectUtm(utm_point, dataset_.utm_box.utm_zone);
	MarkSubimagePixel(rasterized_subimage, projected_pixel, radius_in_pixel, subimage);
}

void ThinStructureReconstructor::MarkSubimageWithShiftedUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& shifted_utm_point, const int& radius_in_pixel, cv::Mat* subimage) {
	MarkSubimageWithUtmPoint(rasterized_subimage, camera_model, Vector3d(shifted_utm_point.ToEigenVector() + reference_point_.ToEigenVector()), radius_in_pixel, subimage);
}

void ThinStructureReconstructor::MarkSubimageWithCylinder(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const int& radius_in_pixel, cv::Mat* subimage) {
	for (int sample_index = 0; sample_index <= 500; ++sample_index) {
		const Vector3d sample_axis = (500 - sample_index) * 1.0 / 500 * cylinder.pa.ToEigenVector()
			+ sample_index * 1.0 / 500 * cylinder.pb.ToEigenVector();
		MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, sample_axis, radius_in_pixel, subimage);
	}
}

bool ThinStructureReconstructor::MarkSubimagePixel(const RasterizedSubimage& rasterized_subimage, const Vector2d& pixel, const int& radius_in_pixel, cv::Mat* subimage) {
	const Vector2i integer_pixel(round(pixel.x), round(pixel.y));
	if (rasterized_subimage.bounds.Contains(integer_pixel)) {
		const Vector2i shifted_pixel = integer_pixel.ToEigenVector() - rasterized_subimage.bounds.min_bounds.ToEigenVector();
		cv::circle(*subimage, shifted_pixel.ToCvPoint(), radius_in_pixel, cv::Scalar(0, 0, 255), -1);
		return true;
	} else {
		return false;
	}
}

