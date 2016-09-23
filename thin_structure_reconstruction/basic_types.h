#ifndef BASIC_TYPES_H
#define BASIC_TYPES_H

#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <vector>

using namespace std;

struct Vector3d {
	double x, y, z;
	Vector3d() {}
	Vector3d(const double& input_x, const double& input_y, const double& input_z)
		: x(input_x), y(input_y), z(input_z) {}
	Vector3d(const Eigen::Vector3d& eigen_vector)
		: x(eigen_vector.x()), y(eigen_vector.y()), z(eigen_vector.z()) {}
	Eigen::Vector3d ToEigenVector() const {
		return Eigen::Vector3d(x, y, z);
	}
};

struct Vector2d {
	double x, y;
	Vector2d() {}
	Vector2d(const double& input_x, const double& input_y)
		: x(input_x), y(input_y) {}
	Vector2d(const Eigen::Vector2d& eigen_vector)
		: x(eigen_vector.x()), y(eigen_vector.y()) {}
	Eigen::Vector2d ToEigenVector() const {
		return Eigen::Vector2d(x, y);
	}
};

struct Vector2i {
	int x, y;
	Vector2i() {}
	Vector2i(const int& input_x, const int& input_y)
		: x(input_x), y(input_y) {}
	Vector2i(const Eigen::Vector2i& eigen_vector)
		: x(eigen_vector.x()), y(eigen_vector.y()) {}
	Eigen::Vector2i ToEigenVector() const {
		return Eigen::Vector2i(x, y);
	}
};

struct HalfOpenBox2i {
	Vector2i min_bounds;
	Vector2i max_bounds;
	bool Contains(const Vector2i& point) const {
		return point.x >= min_bounds.x && point.y >= min_bounds.y
			&& point.x < max_bounds.x && point.y < max_bounds.y; 
	}
};

struct Matrix3d {
	double elements[9];
	Matrix3d() {}
	Matrix3d(const Eigen::Matrix3d& eigen_matrix) {
		int index = 0;
		for (int row = 0; row < 3; ++row) {
			for (int col = 0; col < 3; ++col) {
				elements[index ++] = eigen_matrix(row, col);
			}
		}
	}
	Eigen::Matrix3d ToEigenMatrix() const {
		Eigen::Matrix3d eigen_matrix;
		eigen_matrix << elements[0], elements[1], elements[2],
						elements[3], elements[4], elements[5],
						elements[6], elements[7], elements[8];
		return eigen_matrix;
	}
};

struct BBox2d {
	Vector2d min_bounds;
	Vector2d max_bounds;
	bool Contains(const Vector2d& point) const {
		return point.x >= min_bounds.x && point.y >= min_bounds.y
			&& point.x <= max_bounds.x && point.y <= max_bounds.y;
	}
};

struct UTMBox {
	BBox2d bbox;
	string utm_zone;
	bool Contains(const double& utm_x, const double& utm_y, const string& input_utm_zone) const {
		return input_utm_zone == utm_zone && bbox.Contains(Vector2d(utm_x, utm_y));
	}
};

struct ExportCameraModel {
	Matrix3d r;
	Vector3d t;
	double k1, k2, fx, fy, fs, cx, cy;
	Vector2d Project(const Vector3d& ecef_point) const {
		const Eigen::Vector3d undistorted_point_in_camera = r.ToEigenMatrix() * ecef_point.ToEigenVector() + t.ToEigenVector();
		const Eigen::Vector2d undistorted_point_in_image(undistorted_point_in_camera.x() / undistorted_point_in_camera.z(),
			undistorted_point_in_camera.y() / undistorted_point_in_camera.z());
		const double r2 = undistorted_point_in_image.x() * undistorted_point_in_image.x()
			+ undistorted_point_in_image.y() * undistorted_point_in_image.y();
		const Eigen::Vector2d distorted_point_in_image = (1.0 + k1 * r2 + k2 * r2 * r2) * undistorted_point_in_image;
		const Eigen::Vector2d distorted_point_in_pixel(fx * distorted_point_in_image.x() + fs * distorted_point_in_image.y() + cx,
			fy * distorted_point_in_image.y() + cy);
		return Vector2d(distorted_point_in_pixel);
	}
};

struct RasterizedSubimage {
	HalfOpenBox2i bounds;
	string file_path;
	Vector2i original_image_size;
};

struct ImageCamera {
	ExportCameraModel camera_model;
	RasterizedSubimage subimage;
};

struct StereoRaster {
	string file_path;
};

struct Dataset {
	Vector3d utm_reference_point;
	UTMBox utm_box;
	vector<ImageCamera> image_cameras;
	vector<StereoRaster> stereo_rasters;
	vector<Vector3d> points_utm;
};

pcl::PointXYZ Vector3dToPointXYZ(const Vector3d& input_point, const Vector3d& reference_point);

pcl::PointCloud<pcl::PointXYZ> VectorVector3dToPointCloud(const vector<Vector3d>& input_vector, const Vector3d& reference_point);

#endif