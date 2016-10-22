#include "basic_types.h"
#include "parse_data.h"
#include "thin_structure_reconstructor.h"

using namespace std;

const string root_directory = "F:/Eos3D/v1/";
const string export_directory = "F:/Eos3D/data_v1/";
//const Vector3d feature_point(-2704637.217010, -4261735.578278, 3886083.003075);
const int dataset_index = 9;
const int region_index = 0;
//const Vector3d utm_reference_point(4182529.75, 553854.25, 0.00000000);
//const Vector3d utm_reference_point_default(-1.0, -1.0, -1.0);

int main() {
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

	const vector<Dataset> datasets = data_parser.GetDatasets();
	const Dataset& dataset = datasets[0];

	ThinStructureReconstructor reconstructor(dataset, export_directory + NumberToString(dataset_index) + "/" + NumberToString(region_index) + "/");
	reconstructor.ExportReferencePoint();
	reconstructor.ExportRawPoints();

	reconstructor.ComputePCAValues();
	//reconstructor.LoadPCAValues();

	reconstructor.ComputeFilteredPoints();
	//reconstructor.LoadFilteredPoints();

	reconstructor.ComputeRANSAC();
	//reconstructor.LoadRANSAC();

	reconstructor.ComputeExtendedVerticalCylinders();
	//reconstructor.LoadExtendedVerticalCylinders();

	//reconstructor.ExportRawSubimages();
	//reconstructor.ExportRawSubimagesWithMarkedEcefPoint(feature_point);
	//reconstructor.ExportRawSubimagesWithMarkedHypotheses();
	//reconstructor.ComputeRawSubimagesRadiusByVoting();
	//reconstructor.ComputeRawSubimagesRadiusBySearching();

	reconstructor.LoadAndCropSubimages();

	//reconstructor.ExportCroppedSubimagesWithMarkedEcefPoint(feature_point);
	reconstructor.ExportCroppedSubimagesWithMarkedHypotheses();

	reconstructor.ComputeCroppedSubimageVerticalEdgeMaps();

	//reconstructor.ComputeCroppedSubimagesRadiusBySearching();
	//reconstructor.ComputeCroppedSubimageTruncatedCones();
	//reconstructor.LoadTruncatedConesWithRadii();

	reconstructor.ComputeCroppedSubimageTruncatedConesWithOffsets();
	//reconstructor.LoadTruncatedConesWithRadiiOffsets();

	reconstructor.ComputeCroppedSubimageTruncatedConesExtents();
	//reconstructor.LoadTruncatedConesWithRadiiOffsetsExtents();
	
	return 0;
}