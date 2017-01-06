#include "basic_types.h"
#include "parse_data.h"
#include "thin_structure_reconstructor.h"

using namespace std;

int main(int argc, char** argv) {
	const string root_directory = argv[1];
	const string export_directory = argv[2];
	const int dataset_index = StringToNumber(argv[3]);
	const string region_index = argv[4];

	time_t result;
	result = time(NULL);
	cout << "Starting time: " << asctime(localtime(&result)) << endl;

	DataParser data_parser;
	data_parser.SetExportDirectory(export_directory);
	data_parser.SetExportRawPoints(false);
	data_parser.SetExportUserPoints(false);
	data_parser.SetImportUserPoints(true);
	data_parser.SetMetadataFileName("metadata.dat");
	data_parser.SetRadius(10.0);
	data_parser.SetRootDirectory(root_directory);
	data_parser.LoadUtmReferencePoint(dataset_index, region_index, "utm_reference_point.txt");
	data_parser.Parse(dataset_index, region_index);

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	const vector<Dataset> datasets = data_parser.GetDatasets();
	const Dataset& dataset = datasets[0];

	ThinStructureReconstructor reconstructor(dataset, export_directory + NumberToString(dataset_index) + "/" + region_index + "/");
	reconstructor.LoadExtendedVerticalCylinders();
	reconstructor.LoadAndCropSubimages();
	reconstructor.LoadTruncatedConesWithRadiiOffsetsExtents();

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	reconstructor.ComputeCroppedSubimageLampMaps();

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	reconstructor.SetImportNeighboringPoints(false);
	reconstructor.ComputePoleWithLamps();
	//reconstructor.LoadPoleWithLamps();

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;
	reconstructor.ComputeAdjustedPoleWithLamps();
	//reconstructor.LoadAdjustedPoleWithLamps();

	result = time(NULL);
	cout << "Finishing time: " << asctime(localtime(&result)) << endl;
	//stream_log.close();
	return 0;
}