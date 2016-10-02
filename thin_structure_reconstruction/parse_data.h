#ifndef PARSE_DATA_H
#define PARSE_DATA_H

#include "basic_types.h"
#include "export_data.pb.h"

#include <Eigen/Dense>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message_lite.h>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

class DataParser {
public:
	DataParser() 
		: parsed_(false), export_raw_points_(false), export_user_points_(false),
		  import_user_points_(false), radius_(0.0) {}
	void SetRadius(double input_radius) {
		radius_ = input_radius;
	}
	double GetRadius() {
		return radius_;
	}
	void SetRootDirectory(string root_directory) {
		root_directory_ = root_directory;
	}
	void SetExportDirectory(string export_directory) {
		export_directory_ = export_directory;
	}
	void SetMetadataFileName(string metadata_file_name) {
		metadata_file_name_ = metadata_file_name;
	}
	void SetExportRawPoints(bool export_raw_points) {
		export_raw_points_ = export_raw_points;
	}
	void SetExportUserPoints(bool export_user_points) {
		export_user_points_ = export_user_points;
	}
	void SetImportUserPoints(bool import_user_points) {
		import_user_points_ = import_user_points;
	}
	void Parse();
	vector<Dataset> GetDatasets() {
		if (!parsed_) {
			Parse();
		}
		return datasets_;
	}
private:
	bool parsed_;
	bool export_raw_points_;
	bool export_user_points_;
	bool import_user_points_;
	string root_directory_;
	string export_directory_;
	string metadata_file_name_;
	vector<Dataset> datasets_;
	double radius_;
	stereo_export::Metadata ParseMetadata();
	stereo_export::StereoRasterPoints ParseStereoRasterPoints(const string& relative_path);
	Dataset ParseDataset(const stereo_export::DatasetMetadata& dataset_metadata, const string& raw_utm_file_name, const string& raw_ecef_file_name, const string& user_utm_file_name, const string& user_ecef_file_name);
	vector<Dataset> ParseDatasets(const stereo_export::Metadata& metadata);
	vector<StereoRaster> ParseStereoRasters(const google::protobuf::RepeatedPtrField<stereo_export::StereoRasterMetadata>& stereo_rasters, const UTMBox& utm_box_user, vector<Vector3d>* points_utm, const UTMBox& utm_box_raw, const string& raw_utm_file_name, const string& raw_ecef_file_name, const string& user_utm_file_name, const string& user_ecef_file_name);
	vector<ImageCamera> ParseImageCameras(const google::protobuf::RepeatedPtrField<stereo_export::ImageCameraMetadata>& image_cameras);
};

Eigen::Matrix3d ProtoToMatrix(const stereo_export::Matrix3x3d& input_matrix);
Eigen::Vector3d ProtoToVector(const stereo_export::Vector3d& input_vector);
Eigen::Vector2i ProtoToVector(const stereo_export::Vector2i& input_vector);
HalfOpenBox2i ProtoToHalfOpenBox(const stereo_export::HalfOpenBox2i& input_bbox);
ExportCameraModel ProtoToCameraModel(const stereo_export::CameraModel& input_camera_model);
UTMBox ComputeUTMBox(const double& utm_x, const double& utm_y, const string& utm_zone, const double& radius);

#endif