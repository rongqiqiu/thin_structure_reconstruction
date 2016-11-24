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
	const string run_id = argv[3];
	const string binary_directory = "F:\\Eos3D\\src\\thin_structure_reconstruction\\Release\\";

	ofstream bat_stream(export_directory + "run_all_datasets.bat");
	for (int dataset_index = 0; dataset_index < 10; ++dataset_index) {
		if (dataset_index == 3) continue;
		bat_stream << binary_directory << "run_entire_dataset";
		bat_stream << " " << root_directory;
		bat_stream << " " << export_directory;
		bat_stream << " " << dataset_index;
		bat_stream << " " << run_id << endl;

		bat_stream << "call " << export_directory << dataset_index << "\\" << run_id << "\\run_all.bat" << endl;

		bat_stream << "mkdir " << export_directory << dataset_index << "\\" << run_id << "\\ground_truth\\" << endl;
		bat_stream << "copy /Y " << export_directory << dataset_index << "\\ground_truth.dat";
		bat_stream << " " << export_directory << dataset_index << "\\" << run_id << "\\ground_truth\\truncated_cone_hypotheses_with_radii_offsets_extents.dat" << endl;	

		bat_stream << binary_directory << "collect_entire_dataset";
		bat_stream << " " << root_directory;
		bat_stream << " " << export_directory;
		bat_stream << " " << dataset_index;
		bat_stream << " " << run_id << endl;

		bat_stream << binary_directory << "evaluate_entire_dataset";
		bat_stream << " " << export_directory << dataset_index << "\\" << run_id << "\\" << endl;

	}
	bat_stream.close();

	return 0;
}