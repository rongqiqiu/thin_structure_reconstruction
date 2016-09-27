#include "parse_data.h"

Eigen::Matrix3d ProtoToMatrix(const stereo_export::Matrix3x3d& input_matrix) {
	Eigen::Matrix3d matrix_3d;
	matrix_3d << input_matrix.e00(), input_matrix.e01(), input_matrix.e02(),
				 input_matrix.e10(), input_matrix.e11(), input_matrix.e12(),
				 input_matrix.e20(), input_matrix.e21(), input_matrix.e22();
	return matrix_3d;
}

Eigen::Vector3d ProtoToVector(const stereo_export::Vector3d& input_vector) {
	Eigen::Vector3d vector_3d;
	vector_3d << input_vector.x(), input_vector.y(), input_vector.z();
	return vector_3d;
}

Eigen::Vector2i ProtoToVector(const stereo_export::Vector2i& input_vector) {
	Eigen::Vector2i vector_2i;
	vector_2i << input_vector.x(), input_vector.y();
	return vector_2i;
}

HalfOpenBox2i ProtoToHalfOpenBox(const stereo_export::HalfOpenBox2i& input_bbox) {
	HalfOpenBox2i bbox;
	bbox.min_bounds = ProtoToVector(input_bbox.box_min());
	bbox.max_bounds = ProtoToVector(input_bbox.box_max());
	return bbox;
}

ExportCameraModel ProtoToCameraModel(const stereo_export::CameraModel& input_camera_model) {
	ExportCameraModel camera_model;
	camera_model.r = ProtoToMatrix(input_camera_model.r());
	camera_model.t = ProtoToVector(input_camera_model.t());
	camera_model.k1 = input_camera_model.k1();
	camera_model.k2 = input_camera_model.k2();
	camera_model.fx = input_camera_model.fx();
	camera_model.fy = input_camera_model.fy();
	camera_model.fs = input_camera_model.fs();
	camera_model.cx = input_camera_model.cx();
	camera_model.cy = input_camera_model.cy();
	return camera_model;
}

UTMBox ComputeUTMBox(const double& utm_x, const double& utm_y, const string& utm_zone, const double& radius) {
	UTMBox utm_box;
	utm_box.bbox.min_bounds = Eigen::Vector2d(utm_x - radius, utm_y - radius);
	utm_box.bbox.max_bounds = Eigen::Vector2d(utm_x + radius, utm_y + radius);
	utm_box.utm_zone = utm_zone;
	return utm_box;
}

void LatLngToUTM(const double& lat, const double& lng, double* utm_x, double* utm_y, string* utm_zone) {
	char utm_zone_cstr[10];
	UTM::LLtoUTM(lat, lng, *utm_x, *utm_y, utm_zone_cstr);
	*utm_zone = utm_zone_cstr;
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

vector<StereoRaster> DataParser::ParseStereoRasters(const google::protobuf::RepeatedPtrField<stereo_export::StereoRasterMetadata>& stereo_rasters, const UTMBox& utm_box_user, vector<Vector3d>* points_utm, const UTMBox& utm_box_raw, const string& raw_utm_file_name, const string& raw_ecef_file_name, const string& user_utm_file_name, const string& user_ecef_file_name) {
	if (import_user_points_) {
		cout << "About to import stereo raster points." << endl;
		ifstream istream_user;
		istream_user.open(user_utm_file_name);
		double x, y, z;
		while (istream_user >> x >> y >> z) {
			points_utm->emplace_back(x, y, z);
		}
		istream_user.close();
		return vector<StereoRaster>();
	} else {
		cout << "About to parse " << stereo_rasters.size() << " stereo rasters." << endl;
		ofstream ostream_raw_utm;
		ofstream ostream_raw_ecef;
		ofstream ostream_user_utm;
		ofstream ostream_user_ecef;
		if (export_raw_points_) {
			ostream_raw_utm.open(raw_utm_file_name);
			ostream_raw_ecef.open(raw_ecef_file_name);
		}
		if (export_user_points_) {
			ostream_user_utm.open(user_utm_file_name);
			ostream_user_ecef.open(user_ecef_file_name);
		}
		vector<StereoRaster> vector_stereo_raster;
		for (int index_stereo_raster = 0; index_stereo_raster < stereo_rasters.size(); ++index_stereo_raster) {
			cout << "Parsing stereo raster #" << index_stereo_raster << endl;
			const stereo_export::StereoRasterMetadata& metadata = stereo_rasters.Get(index_stereo_raster);
			StereoRaster stereo_raster;
			stereo_raster.file_path = metadata.stereo_raster_path();
			const stereo_export::StereoRasterPoints stereo_raster_points = ParseStereoRasterPoints(metadata.stereo_raster_path());
			if (stereo_raster_points.w() * stereo_raster_points.h() != stereo_raster_points.points_size()) {
				cout << "Wrong size of stereo raster points!" << endl;
				throw std::exception();
			}
			const stereo_export::Vector3d& center = stereo_raster_points.center();
			int index_point = 0;
			bool first_row = false;
			bool last_row = false;
			bool first_col = false;
			bool last_col = false;
			for (int row = 0; row < stereo_raster_points.h(); ++row) {
				for (int col = 0; col < stereo_raster_points.w(); ++col) {
					const stereo_export::Vector3f& offset = stereo_raster_points.points(index_point);
					const Eigen::Vector3d ecef_point(center.x() + offset.x(), center.y() + offset.y(), center.z() + offset.z());
					double utm_x;
					double utm_y;
					double utm_z;
					string utm_zone;
					EcefToUTM(ecef_point.x(), ecef_point.y(), ecef_point.z(), &utm_x, &utm_y, &utm_z, &utm_zone);
					if (utm_box_raw.Contains(utm_x, utm_y, utm_zone)) {
						if (row == 0) {
							first_row = true;
						}
						if (row == stereo_raster_points.h() - 1) {
							last_row = true;
						}
						if (col == 0) {
							first_col = true;
						}
						if (col == stereo_raster_points.w() - 1) {
							last_col = true;
						}
						if (export_raw_points_) {
							ostream_raw_utm << setprecision(8) << fixed << utm_x << " " << utm_y << " " << utm_z << endl;
							ostream_raw_ecef << setprecision(8) << fixed << ecef_point.x() << " " << ecef_point.y() << " " << ecef_point.z() << endl;
						}
					}
					if (utm_box_user.Contains(utm_x, utm_y, utm_zone)) {
						points_utm->emplace_back(utm_x, utm_y, utm_z);
						if (export_user_points_) {
							ostream_user_utm << setprecision(8) << fixed << utm_x << " " << utm_y << " " << utm_z << endl;
							ostream_user_ecef << setprecision(8) << fixed << ecef_point.x() << " " << ecef_point.y() << " " << ecef_point.z() << endl;
						}
					}
					++index_point;
				}
			}
			if (!first_row || !last_row || !first_col || !last_col) {
				cout << "Stereo raster points can be cropped!" << endl;
				throw std::exception();
			}
			vector_stereo_raster.emplace_back(stereo_raster);
		}
		if (vector_stereo_raster.size() != stereo_rasters.size()) {
			cout << "Stereo raster size error!" << endl;
			throw std::exception();
		}
		if (export_raw_points_) {
			ostream_raw_utm.close();
			ostream_raw_ecef.close();
		}
		if (export_user_points_) {
			ostream_user_utm.close();
			ostream_user_ecef.close();
		}
		return vector_stereo_raster;
	}
}

vector<ImageCamera> DataParser::ParseImageCameras(const google::protobuf::RepeatedPtrField<stereo_export::ImageCameraMetadata>& image_cameras) {
	cout << "About to parse " << image_cameras.size() << " images and cameras." << endl;
	vector<ImageCamera> vector_image_camera;
	for (int index_image_camera = 0; index_image_camera < image_cameras.size(); ++index_image_camera) {
		cout << "Parsing image camera #" << index_image_camera << endl;
		const stereo_export::ImageCameraMetadata& metadata = image_cameras.Get(index_image_camera);
		ImageCamera image_camera;
		ExportCameraModel& camera_model = image_camera.camera_model;
		camera_model = ProtoToCameraModel(metadata.camera_model());
		RasterizedSubimage& rasterized_subimage = image_camera.subimage;
		rasterized_subimage.bounds = ProtoToHalfOpenBox(metadata.bounds());
		rasterized_subimage.file_path = root_directory_ + metadata.image_path();
		rasterized_subimage.original_image_size = ProtoToVector(metadata.original_size());
		vector_image_camera.emplace_back(image_camera);
	}
	if (vector_image_camera.size() != image_cameras.size()) {
		cout << "Image camera size error!" << endl;
		throw std::exception();
	}
	return vector_image_camera;
}

stereo_export::Metadata DataParser::ParseMetadata() {
	stereo_export::Metadata metadata;
	cout << "Parsing " << root_directory_ + metadata_file_name_ << endl;
	if (!google::protobuf::TextFormat::ParseFromString(ReadFileToString(root_directory_ + metadata_file_name_), &metadata)) {
		cout << "Error parsing metadata!" << endl;
		throw std::exception();
	}
	return metadata;
}

stereo_export::StereoRasterPoints DataParser::ParseStereoRasterPoints(const string& relative_path) {
	stereo_export::StereoRasterPoints stereo_raster_points;
	cout << "Parsing " << root_directory_ + relative_path << endl;
	fstream input_stream(root_directory_ + relative_path, ios::in | ios::binary);
	if (!stereo_raster_points.ParseFromIstream(&input_stream)) {
		cout << "Error parsing stereo raster points!" << endl;
		throw std::exception();
	}
	return stereo_raster_points;
}

Dataset DataParser::ParseDataset(const stereo_export::DatasetMetadata& dataset_metadata, const string& raw_utm_file_name, const string& raw_ecef_file_name, const string& user_utm_file_name, const string& user_ecef_file_name) {
	double utm_x, utm_y;
	string utm_zone;
	LatLngToUTM(dataset_metadata.latitude_degrees(), dataset_metadata.longitude_degrees(), &utm_x, &utm_y, &utm_zone);
	cout << "utm_x = " << setprecision(2) << fixed << utm_x << endl;
	cout << "utm_y = " << setprecision(2) << fixed << utm_y << endl;
	cout << "utm_zone = " << utm_zone << endl;
	Dataset dataset;
	UTMBox utm_box_raw = ComputeUTMBox(utm_x, utm_y, utm_zone, dataset_metadata.radius_meters());
	UTMBox utm_box_user = ComputeUTMBox(utm_x, utm_y, utm_zone, radius_);
	dataset.utm_reference_point = Vector3d(utm_x, utm_y, 0.0);
	dataset.utm_box = utm_box_user;
	dataset.stereo_rasters = ParseStereoRasters(dataset_metadata.stereo_raster(), utm_box_user, &(dataset.points_utm), utm_box_raw, raw_utm_file_name, raw_ecef_file_name, user_utm_file_name, user_ecef_file_name);
	dataset.image_cameras = ParseImageCameras(dataset_metadata.image_camera());
	return dataset;
}

vector<Dataset> DataParser::ParseDatasets(const stereo_export::Metadata& metadata) {
	cout << "About to parse " << metadata.dataset_size() << " datasets." << endl;
	vector<Dataset> datasets;
	for (int dataset_index = 0; dataset_index < metadata.dataset_size(); ++dataset_index) {
		cout << "Parsing Dataset #" << dataset_index << endl;
		datasets.emplace_back(ParseDataset(metadata.dataset(dataset_index),
			export_directory_ + NumberToString(dataset_index) + "/raw_utm.xyz",
			export_directory_ + NumberToString(dataset_index) + "/raw_ecef.xyz",
			export_directory_ + NumberToString(dataset_index) + "/user_utm.xyz",
			export_directory_ + NumberToString(dataset_index) + "/user_ecef.xyz"
			));
	}
	if (datasets.size() != metadata.dataset_size()) {
		cout << "Dataset size error!" << endl;
		throw std::exception();
	}
	return datasets;
}

void DataParser::Parse() {
	const stereo_export::Metadata metadata = ParseMetadata();
	datasets_ = ParseDatasets(metadata);
	cout << "Parsing succeeded!" << endl;
	parsed_ = true;
}