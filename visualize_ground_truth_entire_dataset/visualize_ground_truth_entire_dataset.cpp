#include "basic_types.h"
#include "parse_data.h"
#include "thin_structure_reconstructor.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

int main(int argc, char** argv) {
	const string root_directory = argv[1];
	const string export_directory = argv[2];
	const string dataset_index = argv[3];
	const string run_id = argv[4];

	double utm_x, utm_y, utm_z;
	ifstream in_stream(export_directory + dataset_index + "\\dataset_reference_point.txt");
	in_stream >> utm_x >> utm_y;
	utm_z = 0.0;
	in_stream.close();

	DataParser data_parser;
	data_parser.SetExportDirectory(export_directory);
	data_parser.SetParseStereoRasters(false);
	data_parser.SetMetadataFileName("metadata.dat");
	data_parser.SetRadius(50.0);
	data_parser.SetRootDirectory(root_directory);
	data_parser.SetUtmReferencePoint(Vector3d(utm_x, utm_y, utm_z));
	data_parser.Parse(StringToNumber(dataset_index), run_id + "\\ground_truth");

	const vector<Dataset> datasets = data_parser.GetDatasets();
	const Dataset& dataset = datasets[0];

	ThinStructureReconstructor reconstructor(dataset, export_directory + dataset_index + "\\" + run_id + "\\ground_truth\\");
	reconstructor.LoadTruncatedConesWithRadiiOffsetsExtents();
	reconstructor.ExportTruncatdConesMeshesWithRadiiOffsetsExtents();
	reconstructor.ExportRawSubimagesWithMarkedTruncatedConesWithRadiiOffsetsExtents();

	return 0;
}