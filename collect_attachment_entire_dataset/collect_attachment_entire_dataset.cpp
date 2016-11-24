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

	const string output_directory = export_directory + dataset_index + "\\" + run_id + "\\all\\";

	ofstream out_stream(output_directory + "pole_with_lamps.dat");

	for (int region_index_x = 0; region_index_x < 5; ++ region_index_x) {
		for (int region_index_y = 0; region_index_y < 5; ++ region_index_y) {
			cout << "Processing " << region_index_x << " " << region_index_y << endl;

			const double offset_x = (region_index_x - 2) * 20.0;
			const double offset_y = (region_index_y - 2) * 20.0;
			const double offset_z = 0.0;

			const double region_utm_x = utm_x + offset_x;
			const double region_utm_y = utm_y + offset_y;
			const double region_utm_z = utm_z + offset_z;

			const string sub_directory = export_directory + dataset_index + "\\" + run_id + "\\" + NumberToString(region_index_x) + "_" + NumberToString(region_index_y) + "\\";
			cout << sub_directory + "pole_with_lamps.dat" << endl;

			double p1x, p1y, p1z, p2x, p2y, p2z, r1, r2, ex, ey, ez;
			int has_lamp;
			ifstream in_stream(sub_directory + "pole_with_lamps.dat");
			while (in_stream >> p1x >> p1y >> p1z >> p2x >> p2y >> p2z >> r1 >> r2) {
				in_stream >> has_lamp;
				if (has_lamp) {
					in_stream >> ex >> ey >> ez;
				}
				out_stream << setprecision(8) << fixed;
				out_stream << p1x + offset_x << " " << p1y + offset_y << " " << p1z + offset_z << " ";
				out_stream << p2x + offset_x << " " << p2y + offset_y << " " << p2z + offset_z << " ";
				out_stream << r1 << " " << r2 << " " << has_lamp;
				if (has_lamp) {
					out_stream << " " << ex + offset_x << " " << ey + offset_y << " " << ez + offset_z;
				}
				out_stream << endl;
			}
			in_stream.close();

		}
	}

	out_stream.close();

	DataParser data_parser;
	data_parser.SetExportDirectory(export_directory);
	data_parser.SetParseStereoRasters(false);
	data_parser.SetMetadataFileName("metadata.dat");
	data_parser.SetRadius(50.0);
	data_parser.SetRootDirectory(root_directory);
	data_parser.SetUtmReferencePoint(Vector3d(utm_x, utm_y, utm_z));
	data_parser.Parse(StringToNumber(dataset_index), run_id + "\\all");

	const vector<Dataset> datasets = data_parser.GetDatasets();
	const Dataset& dataset = datasets[0];

	ThinStructureReconstructor reconstructor(dataset, export_directory + dataset_index + "\\" + run_id + "\\all\\");
	reconstructor.LoadPoleWithLamps();
	reconstructor.ExportRawSubimagesWithMarkedPoleWithLamps();

	return 0;
}