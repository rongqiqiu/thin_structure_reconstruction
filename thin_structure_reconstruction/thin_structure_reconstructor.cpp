#include "thin_structure_reconstructor.h"

#include <pcl/common/common_headers.h>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <iostream>
#include <fstream>
#include <iomanip>

void ThinStructureReconstructor::ParseDataset() {
	reference_point_ = dataset_.utm_reference_point;
	point_cloud_ = VectorPoint3dToPointCloud(dataset_.points_utm, reference_point_);
}

void ThinStructureReconstructor::ExportRawPoints(const string& file_path) {
	ofstream out_stream;
	out_stream.open(file_path);
	for (int index = 0; index < point_cloud_.points.size(); ++index) {
		const pcl::PointXYZ& point = point_cloud_.points[index];
		out_stream << setprecision(8) << fixed << point.x << " " << point.y << " " << point.z << endl;
	}
	out_stream.close();
}

void ThinStructureReconstructor::ComputeCylinderHypotheses() {
}
