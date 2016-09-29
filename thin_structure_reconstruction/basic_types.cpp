#include "basic_types.h"

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

string NumberToString(const int& number) {
	ostringstream oss;
	oss << number;
	return oss.str();
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
