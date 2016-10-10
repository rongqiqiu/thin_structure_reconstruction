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
		for (int cylinder_index = 0; cylinder_index < extended_cylinder_hypotheses_.size(); ++cylinder_index) {
			const CylinderPrimitive& cylinder = extended_cylinder_hypotheses_[cylinder_index];
			for (int end_index = 0; end_index <= 1; ++end_index) {
				Vector3d point;
				if (end_index == 0) {
					point = cylinder.pa;
				} else {
					point = cylinder.pb;
				}
				const Vector2d projected_pixel = ProjectShiftedUtmPoint(cropped_image_camera.camera_model, point);
				cropped_image_camera.subimage.bounds.ExtendsTo(projected_pixel);
			}
		}
		cropped_image_camera.subimage.bounds.Expands(100);
		cropped_image_camera.subimage.bounds.Intersect(image_camera.subimage.bounds);
		cout << "Cropped bounds: (" << cropped_image_camera.subimage.bounds.min_bounds.x << ", " << cropped_image_camera.subimage.bounds.min_bounds.y << ")"
			 << " - (" << cropped_image_camera.subimage.bounds.max_bounds.x << ", " << cropped_image_camera.subimage.bounds.max_bounds.y << ")" << endl;

		if (!cropped_image_camera.subimage.bounds.IsEmpty()) {
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
			MarkSubimageWithCylinderSurfaceAxisOutline(image_camera.subimage, image_camera.camera_model, cylinder, cv::Scalar(0, 255, 0), &subimage);
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
			MarkSubimageWithCylinderSurfaceAxisOutline(image_camera.subimage, image_camera.camera_model, cylinder, cv::Scalar(0, 255, 0), &subimage);
		}
		cv::imwrite(export_directory_ + NumberToString(index) + "_marked_cylinder.png", subimage);
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

void ThinStructureReconstructor::MarkSubimageWithCylinderSurfaceAxisOutline(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const cv::Scalar& color, cv::Mat* subimage) {
	MarkSubimageWithCylinderSurface(rasterized_subimage, camera_model, cylinder, cv::Scalar(0, 255, 0), subimage);
	MarkSubimageWithCylinderAxis(rasterized_subimage, camera_model, cylinder, cv::Scalar(0, 0, 255), 2.0, subimage);
	MarkSubimageWithCylinderOutline(rasterized_subimage, camera_model, cylinder, cv::Scalar(255, 0, 0), 2.0, subimage);
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

void ThinStructureReconstructor::MarkSubimageWithCylinderAxis(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage) {
	for (int sample_index = 0; sample_index <= 500; ++sample_index) {
		const Vector3d sample_axis = (500 - sample_index) * 1.0 / 500 * cylinder.pa.ToEigenVector()
			+ sample_index * 1.0 / 500 * cylinder.pb.ToEigenVector();
		MarkSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, sample_axis, color, radius_in_pixel, subimage);
	}
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
		const Eigen::Vector3d surface_point = axis_point + dir_x * cylinder.r * cos(angle) + dir_y * cylinder.r * sin(angle);
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
	cv::imwrite(export_directory_ + NumberToString(index) + "_normalized_grad_x.png", normalized_grad_x * 255.0);
	cv::imwrite(export_directory_ + NumberToString(index) + "_normalized_grad_y.png", normalized_grad_y * 255.0);

	cv::Mat raw_vertical_edge_response = cv::max(abs_grad_x - abs_grad_y, 0.0);
	cv::Mat vertical_edge_response;
	cv::normalize(raw_vertical_edge_response, vertical_edge_response, 0.0, 1.0, cv::NORM_MINMAX);

	return vertical_edge_response;
}

double ThinStructureReconstructor::RetrieveSubimagePixelRounding(const RasterizedSubimage& rasterized_subimage, const Vector2d& pixel, const cv::Mat& subimage) {
	const Vector2i integer_pixel(round(pixel.x), round(pixel.y));
	if (rasterized_subimage.bounds.Contains(integer_pixel)) {
		const Vector2i shifted_pixel = integer_pixel.ToEigenVector() - rasterized_subimage.bounds.min_bounds.ToEigenVector();
		return subimage.at<float>(shifted_pixel.y, shifted_pixel.x);
	} else {
		return 0.0;
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
		return 0.0;
	}
}

double ThinStructureReconstructor::RetrieveSubimageWithUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& utm_point, const cv::Mat& subimage) {
	const Vector2d projected_pixel = camera_model.ProjectUtm(utm_point, dataset_.utm_box.utm_zone);
	return RetrieveSubimagePixel(rasterized_subimage, projected_pixel, subimage);
}

double ThinStructureReconstructor::RetrieveSubimageWithShiftedUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& shifted_utm_point, const cv::Mat& subimage) {
	return RetrieveSubimageWithUtmPoint(rasterized_subimage, camera_model, Vector3d(shifted_utm_point.ToEigenVector() + reference_point_.ToEigenVector()), subimage);
}

double ThinStructureReconstructor::ComputeEdgeResponse(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const cv::Mat& subimage) {
	const double outline_angle = ComputeOutlineAngle(camera_model, cylinder);
	double edge_response = 0.0;
	for (int sample_index = 0; sample_index <= 500; ++sample_index) {
		const Eigen::Vector3d sample_axis = (500 - sample_index) * 1.0 / 500 * cylinder.pa.ToEigenVector()
			+ sample_index * 1.0 / 500 * cylinder.pb.ToEigenVector();
		const Eigen::Vector3d dir_z = (cylinder.pb.ToEigenVector() - cylinder.pa.ToEigenVector()).normalized();
		const Eigen::Vector3d dir_x = Eigen::Vector3d(1.0, 0.0, 0.0).cross(dir_z).normalized();
		const Eigen::Vector3d dir_y = dir_z.cross(dir_x).normalized();
		{
			const double angle = outline_angle;
			const Eigen::Vector3d vertex = sample_axis + dir_x * cylinder.r * cos(angle) + dir_y * cylinder.r * sin(angle);
			edge_response += RetrieveSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), subimage);
		}
		{
			const double angle = outline_angle > PI ? outline_angle - PI : outline_angle + PI;
			const Eigen::Vector3d vertex = sample_axis + dir_x * cylinder.r * cos(angle) + dir_y * cylinder.r * sin(angle);
			edge_response += RetrieveSubimageWithShiftedUtmPoint(rasterized_subimage, camera_model, Vector3d(vertex), subimage);
		}
	}
	return edge_response;
}

void ThinStructureReconstructor::ComputeRawSubimagesRadiusByVoting() {
	cylinder_hypotheses_with_radii_.clear();
	ofstream out_stream(export_directory_ + "radius.txt");
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

				const double edge_response = ComputeEdgeResponse(image_camera.subimage, image_camera.camera_model, cylinder_new, vertical_edge_response);
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

				const double edge_response = ComputeEdgeResponse(image_camera.subimage, image_camera.camera_model, cylinder_new, vertical_edge_response);
				sum_edge_response += edge_response;
			}
			cout << "Radius: " << radius << " sum edge response: " << sum_edge_response << endl;
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
	double duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Time spent in computing radius by searching: " << duration << "s" << endl;
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

void ThinStructureReconstructor::ComputeCroppedSubimagesRadiusBySearching() {
	clock_t start = clock();
	cout << "Computing radius by searching" << endl;
	cylinder_hypotheses_with_radii_.clear();
	cropped_edge_maps_.clear();
	for (int index = 0; index < cropped_image_cameras_.size(); ++index) {
		const cv::Mat& subimage = cropped_subimages_[index];
		const cv::Mat vertical_edge_response = ComputeVerticalEdgeMap(subimage, index);
		cv::imwrite(export_directory_ + NumberToString(index) + "_vertical_edge_response.png", vertical_edge_response * 255.0);
		cropped_edge_maps_.push_back(vertical_edge_response);
	}
	//TestBilinearPixelRetrieval();
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

				const double edge_response = ComputeEdgeResponse(image_camera.subimage, image_camera.camera_model, cylinder_new, vertical_edge_response);
				sum_edge_response += edge_response;
			}
			cout << "Radius: " << radius << " sum edge response: " << sum_edge_response << endl;
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
	double duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Time spent in computing radius by searching: " << duration << "s" << endl;
	ExportCylinderPrimitives(cylinder_hypotheses_with_radii_, "cylinder_hypotheses_with_radii.dat");
	ExportCylinderMeshes(cylinder_hypotheses_with_radii_, "cylinder_hypotheses_with_radii.obj");
}
