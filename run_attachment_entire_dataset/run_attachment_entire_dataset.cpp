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

	ofstream bat_stream(export_directory + dataset_index + "\\" + run_id + "\\run_attachment_all.bat");
	for (int region_index_x = 0; region_index_x < 5; ++ region_index_x) {
		for (int region_index_y = 0; region_index_y < 5; ++ region_index_y) {
			const string sub_directory = export_directory + dataset_index + "\\" + run_id  + "\\" + NumberToString(region_index_x) + "_" + NumberToString(region_index_y) + "\\";

			bat_stream << "F:\\Eos3D\\src\\thin_structure_reconstruction\\Release\\thin_structure_attachment_reconstruction";
			bat_stream << " " << root_directory;
			bat_stream << " " << export_directory;
			bat_stream << " " << dataset_index;
			bat_stream << " " << run_id << "\\" << region_index_x << "_" << region_index_y;
			bat_stream << " > " << sub_directory << "log_attachment.txt" << endl;
		}
	}
	bat_stream.close();

	return 0;
}