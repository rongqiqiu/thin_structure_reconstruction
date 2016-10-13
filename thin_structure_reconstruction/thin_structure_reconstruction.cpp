#include "basic_types.h"
#include "parse_data.h"
#include "thin_structure_reconstructor.h"

using namespace std;

const string root_directory = "F:/Eos3D/v0/";
const string export_directory = "F:/Eos3D/data/";
const Vector3d feature_point(-2704637.217010, -4261735.578278, 3886083.003075);

int main() {
	DataParser data_parser;
	data_parser.SetExportDirectory(export_directory);
	data_parser.SetExportRawPoints(false);
	data_parser.SetExportUserPoints(false);
	data_parser.SetImportUserPoints(true);
	data_parser.SetMetadataFileName("metadata.dat");
	data_parser.SetRadius(10.0);
	data_parser.SetRootDirectory(root_directory);
	data_parser.Parse();

	const vector<Dataset> datasets = data_parser.GetDatasets();
	const Dataset& dataset = datasets[1];

	ThinStructureReconstructor reconstructor(dataset, export_directory);
	reconstructor.ExportReferencePoint();
	//reconstructor.ExportRawPoints();

	//reconstructor.ComputePCAValues();
	reconstructor.LoadPCAValues();

	//reconstructor.ComputeFilteredPoints();
	reconstructor.LoadFilteredPoints();

	//reconstructor.ComputeRANSAC();
	reconstructor.LoadRANSAC();

	//reconstructor.ComputeExtendedVerticalCylinders();
	reconstructor.LoadExtendedVerticalCylinders();

	reconstructor.LoadAndCropSubimages();

	//reconstructor.ExportRawSubimages();
	//reconstructor.ExportRawSubimagesWithMarkedEcefPoint(feature_point);
	//reconstructor.ExportRawSubimagesWithMarkedHypotheses();
	//reconstructor.ComputeRawSubimagesRadiusByVoting();
	//reconstructor.ComputeRawSubimagesRadiusBySearching();

	//reconstructor.ExportCroppedSubimagesWithMarkedEcefPoint(feature_point);
	//reconstructor.ExportCroppedSubimagesWithMarkedHypotheses();
	reconstructor.ComputeCroppedSubimageVerticalEdgeMaps();

	//reconstructor.ComputeCroppedSubimagesRadiusBySearching();
	reconstructor.ComputeCroppedSubimageTruncatedCones();
	
	return 0;
}