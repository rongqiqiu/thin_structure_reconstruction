#include "basic_types.h"
#include "parse_data.h"
#include "thin_structure_reconstructor.h"

using namespace std;

const string root_directory = "F:/Eos3D/v0/";
const string export_directory = "F:/Eos3D/data/";
const Vector3d feature_point(-2704637.217010, -4261735.578278, 3886083.003075);

//cv::Mat MarkSubimage(const RasterizedSubimage& rasterized_subimage, const Eigen::Vector2d& pixel, const int& radius_in_pixel) {
//	cv::Mat marked_subimage = rasterized_subimage.subimage;
//	const Eigen::Vector2i integer_pixel(round(pixel.x()), round(pixel.y()));
//	if (rasterized_subimage.bounds.Contains(integer_pixel)) {
//		const Eigen::Vector2i shifted_pixel = integer_pixel - rasterized_subimage.bounds.min_bounds;
//		cv::circle(marked_subimage, cv::Point(shifted_pixel.x(), shifted_pixel.y()), radius_in_pixel, cv::Scalar(0, 0, 255));
//	}
//	return marked_subimage;
//}

//cv::imread(root_directory_ + metadata.image_path(), CV_LOAD_IMAGE_COLOR);

int main() {
	DataParser data_parser;
	data_parser.SetExportDirectory(export_directory);
	data_parser.SetExportRawPoints(false);
	data_parser.SetExportUserPoints(false);
	data_parser.SetImportUserPoints(true);
	data_parser.SetMetadataFileName("metadata.dat");
	data_parser.SetRadius(10.0);
	data_parser.SetRootDirectory(root_directory);
	data_parser.Parse();

	const vector<Dataset> datasets = data_parser.GetDatasets();
	const Dataset& dataset = datasets[1];

	ThinStructureReconstructor reconstructor(dataset, export_directory);
	//reconstructor.ExportRawPoints();

	//reconstructor.ComputePCAValues();
	reconstructor.LoadPCAValues();

	reconstructor.ComputeFilteredPoints();
	//reconstructor.LoadFilteredPoints();

	//pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	//Eigen::Vector2d projected_pixel = camera_model.Project(feature_point);
	//cv::Mat marked_subimage = MarkSubimage(rasterized_subimage, projected_pixel, 1);
	//cv::imwrite(export_directory + NumberToString(dataset_index) + "/" + NumberToString(index_image_camera) + ".png", marked_subimage);
	return 0;
}