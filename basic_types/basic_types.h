#ifndef BASIC_TYPES_H
#define BASIC_TYPES_H

#include <Eigen/Dense>
#include <geo/utm.h>
#include <geo/geodetic_converter.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <limits.h>
#include <math.h>
#include <vector>

using namespace std;

struct Vector3d {
	double x, y, z;
	Vector3d() : x(0.0), y(0.0), z(0.0) {}
	Vector3d(const double& input_x, const double& input_y, const double& input_z)
		: x(input_x), y(input_y), z(input_z) {}
	Vector3d(const Eigen::Vector3d& eigen_vector)
		: x(eigen_vector.x()), y(eigen_vector.y()), z(eigen_vector.z()) {}
	Vector3d(const pcl::PointXYZ& point)
		: x(point.x), y(point.y), z(point.z) {}
	Eigen::Vector3d ToEigenVector() const {
		return Eigen::Vector3d(x, y, z);
	}
};

struct Vector2d {
	double x, y;
	Vector2d() : x(0.0), y(0.0) {}
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
	Vector2i() : x(0), y(0) {}
	Vector2i(const int& input_x, const int& input_y)
		: x(input_x), y(input_y) {}
	Vector2i(const Eigen::Vector2i& eigen_vector)
		: x(eigen_vector.x()), y(eigen_vector.y()) {}
	Eigen::Vector2i ToEigenVector() const {
		return Eigen::Vector2i(x, y);
	}
	cv::Point ToCvPoint() const {
		return cv::Point(x, y);
	}
};

struct HalfOpenBox2i {
	Vector2i min_bounds;
	Vector2i max_bounds;
	HalfOpenBox2i() : min_bounds(INT_MAX, INT_MAX), max_bounds(INT_MIN, INT_MIN) {}
	HalfOpenBox2i(const Vector2i& input_max_bounds)
		: min_bounds(0, 0), max_bounds(input_max_bounds) {}
	HalfOpenBox2i(const Vector2i& input_min_bounds, const Vector2i& input_max_bounds)
		: min_bounds(input_min_bounds), max_bounds(input_max_bounds) {}
	bool Contains(const Vector2i& point) const {
		return point.x >= min_bounds.x && point.y >= min_bounds.y
			&& point.x < max_bounds.x && point.y < max_bounds.y; 
	}
	bool Contains(const Vector2d& point) const {
		return point.x >= (double) min_bounds.x && point.y >= (double) min_bounds.y
			&& point.x < (double) max_bounds.x && point.y < (double) max_bounds.y;
	}
	void ExtendsTo(const Vector2i& point) {
		if (point.x < min_bounds.x) {
			min_bounds.x = point.x;
		}
		if (point.y < min_bounds.y) {
			min_bounds.y = point.y;
		}

		if (point.x + 1 > max_bounds.x) {
			max_bounds.x = point.x + 1;
		}
		if (point.y + 1 > max_bounds.y) {
			max_bounds.y = point.y + 1;
		}
	}
	void ExtendsTo(const Vector2d& point) {
		ExtendsTo(Vector2i(floor(point.x), floor(point.y)));
	}
	void Expands(const int& pixels) {
		min_bounds.x -= pixels;
		min_bounds.y -= pixels;

		max_bounds.x += pixels;
		max_bounds.y += pixels;
	}
	void Intersect(const HalfOpenBox2i& box) {
		if (box.min_bounds.x > min_bounds.x) {
			min_bounds.x = box.min_bounds.x;
		}
		if (box.min_bounds.y > min_bounds.y) {
			min_bounds.y = box.min_bounds.y;
		}

		if (box.max_bounds.x < max_bounds.x) {
			max_bounds.x = box.max_bounds.x;
		}
		if (box.max_bounds.y < max_bounds.y) {
			max_bounds.y = box.max_bounds.y;
		}
	}
	bool IsEmpty() const {
		return max_bounds.x <= min_bounds.x || max_bounds.y <= min_bounds.y;
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
	Vector2d ProjectEcef(const Vector3d& ecef_point) const;
	Vector2d ProjectUtm(const Vector3d& utm_point, const string& utm_zone) const;
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

struct CylinderPrimitive {
	Vector3d pa, pb;
	double r;
	double GetLength() {
		return (pa.ToEigenVector() - pb.ToEigenVector()).norm();
	}
};

struct TruncatedConePrimitive {
	Vector3d pa, pb; // pa: center of bottom end; pb: center of top end
	double ra, rb; // ra: radius of bottom end; rb: center of top end
	TruncatedConePrimitive() {}
	TruncatedConePrimitive(const CylinderPrimitive& cylinder)
		: pa(cylinder.pa), pb(cylinder.pb), ra(cylinder.r), rb(cylinder.r) {}
	double GetLength() {
		return (pa.ToEigenVector() - pb.ToEigenVector()).norm();
	}
};

struct EllipsePrimitive {
	Vector3d center;
	double radius_large; // horizontal radius
	double radius_small; // vertical radius;
	EllipsePrimitive() {}
	EllipsePrimitive(const Vector3d& input_center, const double& input_radius_large, const double& input_radius_small)
		: center(input_center), radius_large(input_radius_large), radius_small(input_radius_small) {}
};

struct PoleWithLamp {
	TruncatedConePrimitive truncated_cone;
	bool has_lamp;
	Vector3d end_point;
	double ellipse_radius_large;
	double ellipse_radius_small;
	PoleWithLamp() : has_lamp(false), ellipse_radius_large(0.5), ellipse_radius_small(0.2) {}
	PoleWithLamp(const TruncatedConePrimitive& input_truncated_cone) : truncated_cone(input_truncated_cone), has_lamp(false), ellipse_radius_large(0.5), ellipse_radius_small(0.2) {}
	PoleWithLamp(const Vector3d& input_end_point) : has_lamp(true), end_point(input_end_point), ellipse_radius_large(0.5), ellipse_radius_small(0.2) {}
	PoleWithLamp(const TruncatedConePrimitive& input_truncated_cone, const Vector3d& input_end_point)
		: truncated_cone(input_truncated_cone), has_lamp(true), end_point(input_end_point), ellipse_radius_large(0.5), ellipse_radius_small(0.2) {}
	TruncatedConePrimitive GetArmTruncatedCone() const {
		TruncatedConePrimitive arm_truncated_cone;
		arm_truncated_cone.ra = arm_truncated_cone.rb = truncated_cone.rb;
		arm_truncated_cone.pa = truncated_cone.pb;
		arm_truncated_cone.pb = end_point;
		return arm_truncated_cone;
	}
	EllipsePrimitive GetLampEllipse() const {
		EllipsePrimitive lamp_ellipse;
		lamp_ellipse.center = end_point;
		lamp_ellipse.radius_large = ellipse_radius_large;
		lamp_ellipse.radius_small = ellipse_radius_small;
		return lamp_ellipse;
	}
};

pcl::PointXYZ Vector3dToPointXYZ(const Vector3d& input_point, const Vector3d& reference_point);
pcl::PointCloud<pcl::PointXYZ> VectorVector3dToPointCloud(const vector<Vector3d>& input_vector, const Vector3d& reference_point);

void LatLngToUTM(const double& lat, const double& lng, double* utm_x, double* utm_y, string* utm_zone);
void UTMToLatLng(const double& utm_x, const double& utm_y, const string& utm_zone, double* lat, double* lng);
void EcefToUTM(const double& ecef_x, const double& ecef_y, const double& ecef_z, double* utm_x, double* utm_y, double* utm_z, string* utm_zone);
void UTMToEcef(const double& utm_x, const double& utm_y, const double& utm_z, const string& utm_zone, double* ecef_x, double* ecef_y, double* ecef_z);

string NumberToString(const int& number);
int StringToNumber(const string& str);
string ReadFileToString(const string& relative_path);
int round(const double& number);

double AngleDiff(const double angle1, const double angle2);
double IndexDiff(const int index1, const int index2, const int total);

extern const double PI;

double PointToSegmentDistance(const Vector3d& point, const Vector3d& end_point_1, const Vector3d& end_point_2);

#endif