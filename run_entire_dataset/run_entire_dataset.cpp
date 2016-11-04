#include "basic_types.h"

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

	system(("mkdir " + export_directory + dataset_index + "\\" + run_id + "\\").c_str());

	ofstream bat_stream(export_directory + dataset_index + "\\" + run_id + "\\run_all.bat");
	for (int region_index_x = 0; region_index_x < 5; ++ region_index_x) {
		for (int region_index_y = 0; region_index_y < 5; ++ region_index_y) {
			double region_utm_x = utm_x + (region_index_x - 2) * 20.0;
			double region_utm_y = utm_y + (region_index_y - 2) * 20.0;
			double region_utm_z = utm_z;
			const string sub_directory = export_directory + dataset_index + "\\" + run_id  + "\\" + NumberToString(region_index_x) + "_" + NumberToString(region_index_y) + "\\";

			system(("mkdir " + sub_directory).c_str());

			ofstream out_stream(sub_directory + "utm_reference_point.txt");
			out_stream << setprecision(8) << fixed << region_utm_x << " " << region_utm_y << " " << region_utm_z << endl;
			out_stream.close();

			bat_stream << "F:\\Eos3D\\src\\thin_structure_reconstruction\\Release\\thin_structure_reconstruction";
			bat_stream << " " << root_directory;
			bat_stream << " " << export_directory;
			bat_stream << " " << dataset_index;
			bat_stream << " " << run_id << "\\" << region_index_x << "_" << region_index_y;
			bat_stream << " > " << sub_directory << "log.txt" << endl;
		}
	}
	bat_stream.close();

	return 0;
}