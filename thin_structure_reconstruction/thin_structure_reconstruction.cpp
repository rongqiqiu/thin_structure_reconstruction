#include "basic_types.h"
#include "parse_data.h"
#include "thin_structure_reconstructor.h"

using namespace std;

//const string root_directory = "F:/Eos3D/v1/";
//const string export_directory = "F:/Eos3D/data_v1/";
//const Vector3d feature_point(-2704637.217010, -4261735.578278, 3886083.003075);
//const int dataset_index = 9;
//const int region_index = 0;
//const Vector3d utm_reference_point(4182529.75, 553854.25, 0.00000000);
//const Vector3d utm_reference_point_default(-1.0, -1.0, -1.0);

int main(int argc, char** argv) {
	const string root_directory = argv[1];
	const string export_directory = argv[2];
	const int dataset_index = StringToNumber(argv[3]);
	const string region_index = argv[4];

	//ofstream stream_log(export_directory + "log_" + NumberToString(time(NULL)) + ".txt");

	time_t result;
	result = time(NULL);
	cout << "Starting time: " << asctime(localtime(&result)) << endl;

	DataParser data_parser;
	data_parser.SetExportDirectory(export_directory);
	data_parser.SetExportRawPoints(false);
	data_parser.SetExportUserPoints(true);
	data_parser.SetImportUserPoints(false);
	data_parser.SetMetadataFileName("metadata.dat");
	data_parser.SetRadius(10.0);
	data_parser.SetRootDirectory(root_directory);
	//data_parser.Parse();
	//data_parser.Parse(1, 0, Vector3d(4181445.52224449, 552787.13119375, 0.00000000));
	data_parser.LoadUtmReferencePoint(dataset_index, region_index, "utm_reference_point.txt");
	data_parser.Parse(dataset_index, region_index);

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	const vector<Dataset> datasets = data_parser.GetDatasets();
	const Dataset& dataset = datasets[0];

	ThinStructureReconstructor reconstructor(dataset, export_directory + NumberToString(dataset_index) + "/" + region_index + "/");
	reconstructor.ExportReferencePoint();
	reconstructor.ExportRawPoints();

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	reconstructor.ComputePCAValues();
	//reconstructor.LoadPCAValues();

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	reconstructor.ComputeFilteredPoints();
	//reconstructor.LoadFilteredPoints();

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	reconstructor.ComputeRANSAC();
	//reconstructor.LoadRANSAC();

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	reconstructor.ComputeExtendedVerticalCylinders();
	//reconstructor.LoadExtendedVerticalCylinders();

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	//reconstructor.ExportRawSubimages();
	//reconstructor.ExportRawSubimagesWithMarkedEcefPoint(feature_point);
	//reconstructor.ExportRawSubimagesWithMarkedHypotheses();
	//reconstructor.ComputeRawSubimagesRadiusByVoting();
	//reconstructor.ComputeRawSubimagesRadiusBySearching();

	reconstructor.LoadAndCropSubimages();

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	//reconstructor.ExportCroppedSubimagesWithMarkedEcefPoint(feature_point);

	reconstructor.ExportCroppedSubimagesWithMarkedHypotheses();

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	reconstructor.ComputeCroppedSubimageVerticalEdgeMaps();

	result = time(NULL);
	cout << "Time: " << asctime(localtime(&result)) << endl;

	//reconstructor.ComputeCroppedSubimagesRadiusBySearching();
	//reconstructor.ComputeCroppedSubimageTruncatedCones();
	//reconstructor.LoadTruncatedConesWithRadii();

	//result = time(NULL);
	//cout << "Time: " << asctime(localtime(&result)) << endl;

	//reconstructor.ComputeCroppedSubimageTruncatedConesWithOffsets();
	//reconstructor.LoadTruncatedConesWithRadiiOffsets();

	//result = time(NULL);
	//cout << "Time: " << asctime(localtime(&result)) << endl;

	//reconstructor.ComputeCroppedSubimageTruncatedConesExtents();
	//reconstructor.LoadTruncatedConesWithRadiiOffsetsExtents();

	reconstructor.ComputeCroppedSubimageTruncatedConesWithOffsetsExtents();
	//reconstructor.LoadTruncatedConesWithRadiiOffsetsExtents();

	result = time(NULL);
	cout << "Finishing time: " << asctime(localtime(&result)) << endl;
	//stream_log.close();
	return 0;
}