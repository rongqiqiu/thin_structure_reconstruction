#include "basic_types.h"

Point3d Vector3dToPoint3d(const Eigen::Vector3d& input_vector) {
	return Point3d(input_vector.x(), input_vector.y(), input_vector.z());
}

Eigen::Vector3d Point3dToVector3d(const Point3d& input_point) {
	return Eigen::Vector3d(input_point.x, input_point.y, input_point.z);
}

pcl::PointXYZ Point3dToPointXYZ(const Point3d& input_point, const Point3d& reference_point) {
	Eigen::Vector3d offset = Point3dToVector3d(input_point) - Point3dToVector3d(reference_point);
	return pcl::PointXYZ(offset.x(), offset.y(), offset.z());
}

pcl::PointCloud<pcl::PointXYZ> VectorPoint3dToPointCloud(const vector<Point3d>& input_vector, const Point3d& reference_point) {
	pcl::PointCloud<pcl::PointXYZ> point_cloud;
	point_cloud.points.resize(input_vector.size());
	for (int index = 0; index < input_vector.size(); ++index) {
		point_cloud.points[index] = Point3dToPointXYZ(input_vector[index], reference_point);
	}
	point_cloud.width = input_vector.size();
	point_cloud.height = 1;
	return point_cloud;
}
