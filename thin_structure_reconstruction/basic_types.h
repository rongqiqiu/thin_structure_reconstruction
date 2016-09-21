#ifndef BASIC_TYPES_H
#define BASIC_TYPES_H

#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <vector>

using namespace std;

struct Point3d {
	double x, y, z;
	Point3d() {}
	Point3d(const double& input_x, const double& input_y, const double& input_z)
		: x(input_x), y(input_y), z(input_z) {}
};

struct HalfOpenBox2i {
	Eigen::Vector2i min_bounds;
	Eigen::Vector2i max_bounds;
	bool Contains(const Eigen::Vector2i& point) const {
		return point.x() >= min_bounds.x() && point.y() >= min_bounds.y()
			&& point.x() < max_bounds.x() && point.y() < max_bounds.y(); 
	}
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct BBox2d {
	Eigen::Vector2d min_bounds;
	Eigen::Vector2d max_bounds;
	bool Contains(const Eigen::Vector2d& point) const {
		return point.x() >= min_bounds.x() && point.y() >= min_bounds.y()
			&& point.x() <= max_bounds.x() && point.y() <= max_bounds.y();
	}
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct UTMBox {
	BBox2d bbox;
	string utm_zone;
	bool Contains(const double& utm_x, const double& utm_y, const string& input_utm_zone) const {
		return input_utm_zone == utm_zone && bbox.Contains(Eigen::Vector2d(utm_x, utm_y));
	}
};

struct ExportCameraModel {
	Eigen::Matrix3d r;
	Eigen::Vector3d t;
	double k1, k2, fx, fy, fs, cx, cy;
	Eigen::Vector2d Project(const Eigen::Vector3d& ecef_point) const {
		const Eigen::Vector3d undistorted_point_in_camera = r * ecef_point + t;
		const Eigen::Vector2d undistorted_point_in_image(undistorted_point_in_camera.x() / undistorted_point_in_camera.z(),
			undistorted_point_in_camera.y() / undistorted_point_in_camera.z());
		const double r2 = undistorted_point_in_image.x() * undistorted_point_in_image.x()
			+ undistorted_point_in_image.y() * undistorted_point_in_image.y();
		const Eigen::Vector2d distorted_point_in_image = (1.0 + k1 * r2 + k2 * r2 * r2) * undistorted_point_in_image;
		const Eigen::Vector2d distorted_point_in_pixel(fx * distorted_point_in_image.x() + fs * distorted_point_in_image.y() + cx,
			fy * distorted_point_in_image.y() + cy);
		return distorted_point_in_pixel;
	}
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct RasterizedSubimage {
	HalfOpenBox2i bounds;
	string file_path;
	Eigen::Vector2i original_image_size;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ImageCamera {
	ExportCameraModel camera_model;
	RasterizedSubimage subimage;
};

struct StereoRaster {
	string file_path;
};

struct Dataset {
	Point3d utm_reference_point;
//	UTMBox utm_box;
	vector<ImageCamera> image_cameras;
	vector<StereoRaster> stereo_rasters;
	vector<Point3d> points_utm;
};

Point3d Vector3dToPoint3d(const Eigen::Vector3d& input_vector);

Eigen::Vector3d Point3dToVector3d(const Point3d& input_point);

pcl::PointXYZ Point3dToPointXYZ(const Point3d& input_point, const Point3d& reference_point);

pcl::PointCloud<pcl::PointXYZ> VectorPoint3dToPointCloud(const vector<Point3d>& input_vector, const Point3d& reference_point);

#endif