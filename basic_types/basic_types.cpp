#include "basic_types.h"

Vector2d ExportCameraModel::ProjectEcef(const Vector3d& ecef_point) const {
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

Vector2d ExportCameraModel::ProjectUtm(const Vector3d& utm_point, const string& utm_zone) const {
	Vector3d ecef_point;
	UTMToEcef(utm_point.x, utm_point.y, utm_point.z, utm_zone, &ecef_point.x, &ecef_point.y, &ecef_point.z);
	return ProjectEcef(ecef_point);
}

pcl::PointXYZ Vector3dToPointXYZ(const Vector3d& input_point, const Vector3d& reference_point) {
	Eigen::Vector3d offset = input_point.ToEigenVector() - reference_point.ToEigenVector();
	return pcl::PointXYZ(offset.x(), offset.y(), offset.z());
}

pcl::PointCloud<pcl::PointXYZ> VectorVector3dToPointCloud(const vector<Vector3d>& input_vector, const Vector3d& reference_point) {
	pcl::PointCloud<pcl::PointXYZ> point_cloud;
	point_cloud.points.resize(input_vector.size());
	for (int index = 0; index < input_vector.size(); ++index) {
		point_cloud.points[index] = Vector3dToPointXYZ(input_vector[index], reference_point);
	}
	point_cloud.width = input_vector.size();
	point_cloud.height = 1;
	return point_cloud;
}

void LatLngToUTM(const double& lat, const double& lng, double* utm_x, double* utm_y, string* utm_zone) {
	char utm_zone_cstr[10];
	UTM::LLtoUTM(lat, lng, *utm_x, *utm_y, utm_zone_cstr);
	*utm_zone = utm_zone_cstr;
}

void UTMToLatLng(const double& utm_x, const double& utm_y, const string& utm_zone, double* lat, double* lng) {
	UTM::UTMtoLL(utm_x, utm_y, utm_zone.c_str(), *lat, *lng);
}

void EcefToUTM(const double& ecef_x, const double& ecef_y, const double& ecef_z, double* utm_x, double* utm_y, double* utm_z, string* utm_zone) {
	double lat, lng, alt;
	geodetic_converter::GeodeticConverter converter;
	converter.ecef2Geodetic(ecef_x, ecef_y, ecef_z, &lat, &lng, &alt);
	LatLngToUTM(lat, lng, utm_x, utm_y, utm_zone);
	if (utm_z != NULL) {
		*utm_z = alt;
	}
}

void UTMToEcef(const double& utm_x, const double& utm_y, const double& utm_z, const string& utm_zone, double* ecef_x, double* ecef_y, double* ecef_z) {
	double lat, lng, alt;
	UTMToLatLng(utm_x, utm_y, utm_zone, &lat, &lng);
	alt = utm_z;
	geodetic_converter::GeodeticConverter converter;
	converter.geodetic2Ecef(lat, lng, alt, ecef_x, ecef_y, ecef_z);
}

string NumberToString(const int& number) {
	ostringstream oss;
	oss << number;
	return oss.str();
}

int StringToNumber(const string& str) {
	int number;
	istringstream iss(str);
	iss >> number;
	return number;
}

string ReadFileToString(const string& full_path) {
	ifstream inf(full_path);
	stringstream strStream;
	strStream << inf.rdbuf();
	return strStream.str();
}

int round(const double& number) {
	return number + .5;
}
