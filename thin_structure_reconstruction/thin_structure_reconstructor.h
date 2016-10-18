#ifndef THIN_STRUCTURE_RECONSTRUCTOR_H
#define THIN_STRUCTURE_RECONSTRUCTOR_H

#include "basic_types.h"

#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

struct CylinderPrimitive {
	Vector3d pa, pb;
	double r;
};

struct TruncatedConePrimitive {
	Vector3d pa, pb; // pa: center of bottom end; pb: center of top end
	double ra, rb; // ra: radius of bottom end; rb: center of top end
	TruncatedConePrimitive() {}
	TruncatedConePrimitive(const CylinderPrimitive& cylinder)
		: pa(cylinder.pa), pb(cylinder.pb), ra(cylinder.r), rb(cylinder.r) {}
};

class ThinStructureReconstructor {
public:
	ThinStructureReconstructor() {}
	ThinStructureReconstructor(const Dataset& dataset) : dataset_(dataset) {
		ParseDataset();
	}
	ThinStructureReconstructor(const Dataset& dataset, const string& export_directory) 
		: dataset_(dataset), export_directory_(export_directory) {
		ParseDataset();
	}
	void SetDataset(const Dataset& dataset) {
		dataset_ = dataset;
		ParseDataset();
	}
	Dataset GetDataset() {
		return dataset_;
	}
	void ParseDataset();
	void SetExportDirectory(const string& export_directory) {
		export_directory_ = export_directory;
	}
	void ExportReferencePoint();
	void ExportRawPoints();
	void ComputePCAValues();
	void LoadPCAValues();
	void ComputeFilteredPoints();
	void LoadFilteredPoints();
	void ComputeRANSAC();
	void LoadRANSAC();
	void ComputeCylinderHypotheses();
	void ComputeExtendedVerticalCylinders();
	void LoadExtendedVerticalCylinders();
	void LoadAndCropSubimages();
	void ExportRawSubimages();
	void ExportRawSubimagesWithMarkedEcefPoint(const Vector3d& ecef_point);
	void ExportCroppedSubimagesWithMarkedEcefPoint(const Vector3d& ecef_point);
	void ExportRawSubimagesWithMarkedHypotheses();
	void ExportCroppedSubimagesWithMarkedHypotheses();
	void ComputeRawSubimagesRadiusByVoting();
	void ComputeRawSubimagesRadiusBySearching();
	void ComputeCroppedSubimageVerticalEdgeMaps();
	void ComputeCroppedSubimagesRadiusBySearching();
	void ComputeCroppedSubimageTruncatedCones();
	void ComputeCroppedSubimageTruncatedConesWithOffsets();
	void LoadTruncatedConesWithRadii();
private:
	string export_directory_;
	Dataset dataset_;
	Vector3d reference_point_;
	pcl::PointCloud<pcl::PointXYZ> point_cloud_;
	pcl::PointCloud<pcl::PointXYZ> point_cloud_subsampled_;
	vector<Vector3d> pca_values_;
	vector<int> index_filtered_;
	pcl::PointCloud<pcl::PointXYZ> point_cloud_filtered_;
	vector<CylinderPrimitive> cylinder_hypotheses_;
	vector<CylinderPrimitive> extended_cylinder_hypotheses_;
	vector<ImageCamera> cropped_image_cameras_;
	vector<cv::Mat> cropped_subimages_;
	vector<cv::Mat> cropped_edge_maps_;
	vector<CylinderPrimitive> cylinder_hypotheses_with_radii_;
	vector<TruncatedConePrimitive> truncated_cone_hypotheses_with_radii_;
	vector<TruncatedConePrimitive> truncated_cone_hypotheses_with_radii_offsets_;

	void ExportPointCloud(const pcl::PointCloud<pcl::PointXYZ>& point_cloud, const string& file_name);
	pcl::PointCloud<pcl::PointXYZ> ImportPointCloud(const string& file_name);
	void ApplyRandomSubsampling(const double& sampling_ratio);
	double ComputeMean(const vector<int>& pointIdx, const int& dimension);
	double ComputeStandardDeviation(const vector<int>& pointIdx, const int& dimension);
	Vector3d ComputePCAValue(const vector<int>& pointIdx);
	bool IsVerticalLinear(const Vector3d& pca_value, const double& threshold);
	CylinderPrimitive ComputeVerticalLine(const pcl::PointCloud<pcl::PointXYZ>& point_cloud, const vector<int>& pointIdx);
	void ExportCylinderPrimitives(const vector<CylinderPrimitive>& cylinders, const string& file_name);
	void ExportCylinderMeshes(const vector<CylinderPrimitive>& cylinders, const string& file_name);
	void ExportTruncatedConePrimitives(const vector<TruncatedConePrimitive>& truncated_cones, const string& file_name);
	void ExportTruncatedConeMeshes(const vector<TruncatedConePrimitive>& truncated_cones, const string& file_name);
	vector<CylinderPrimitive> ImportCylinderPrimitives(const string& file_name);
	vector<TruncatedConePrimitive> ImportTruncatedConePrimitives(const string& file_name);
	Vector3d ComputeXYCentroid(const pcl::PointCloud<pcl::PointXYZ>& point_cloud, const vector<int>& pointIdx);
	void ComputeExtents(const pcl::PointCloud<pcl::PointXYZ>& point_cloud, const vector<int>& pointIdx, const Vector3d& axis, const Vector3d& point, double* min_dot, double* max_dot);
	pcl::PointCloud<pcl::PointXYZ> ProjectXY(const pcl::PointCloud<pcl::PointXYZ>& input_cloud);
	CylinderPrimitive ExtendVerticalCylinder(const CylinderPrimitive& cylinder);
	bool MarkSubimagePixel(const RasterizedSubimage& rasterized_subimage, const Vector2d& pixel, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage);
	void MarkSubimageWithEcefPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& ecef_point, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage);
	void MarkSubimageWithUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& utm_point, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage);
	void MarkSubimageWithShiftedUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& shifted_utm_point, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage);
	void MarkSubimageWithCylinderSurfaceAxisOutline(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, cv::Mat* subimage);
	void MarkSubimageWithCylinderSurface(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const cv::Scalar& color, cv::Mat* subimage);
	void MarkSubimageWithCylinderAxis(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage);
	void MarkSubimageWithTruncatedConeSurfaceAxisOutline(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone, cv::Mat* subimage);
	void MarkSubimageWithTruncatedConeSurface(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone, const cv::Scalar& color, cv::Mat* subimage);
	void MarkSubimageWithTruncatedConeAxis(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage);
	double ComputeOutlineAngle(const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder);
	double ComputeOutlineAngle(const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone);
	void MarkSubimageWithCylinderOutline(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage);
	void MarkSubimageWithTruncatedConeOutline(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone, const cv::Scalar& color, const int& radius_in_pixel, cv::Mat* subimage);
	cv::Mat ComputeVerticalEdgeMap(const cv::Mat& subimage, const int& index);
	double RetrieveSubimagePixelRounding(const RasterizedSubimage& rasterized_subimage, const Vector2d& pixel, const cv::Mat& subimage);
	double RetrieveSubimagePixel(const RasterizedSubimage& rasterized_subimage, const Vector2d& pixel, const cv::Mat& subimage);
	double RetrieveSubimageWithUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& utm_point, const cv::Mat& subimage);
	double RetrieveSubimageWithShiftedUtmPoint(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const Vector3d& shifted_utm_point, const cv::Mat& subimage);
	double ComputeCylinderEdgeResponse(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const CylinderPrimitive& cylinder, const cv::Mat& subimage);
	double ComputeTruncatedConeEdgeResponse(const RasterizedSubimage& rasterized_subimage, const ExportCameraModel& camera_model, const TruncatedConePrimitive& truncated_cone, const cv::Mat& subimage);
	double ComputeTruncatedConeSumEdgeResponse(const TruncatedConePrimitive& truncated_cone);
	cv::Mat ExtractCroppedSubimage(const cv::Mat& raw_subimage, const HalfOpenBox2i& raw_bounds, const HalfOpenBox2i& cropped_bounds);
	Vector2d ProjectShiftedUtmPoint(const ExportCameraModel& camera_model, const Vector3d& shifted_utm_point);
	void TestBilinearPixelRetrieval();
	void ExportCroppedSubimagesWithMarkedTruncatedCones(const vector<TruncatedConePrimitive>& truncated_cones, const string& file_name);
	bool FindBestNeighborRadiiOffsets(const TruncatedConePrimitive& truncated_cone, const TruncatedConePrimitive& original_truncated_cone, TruncatedConePrimitive* best_neighbor_truncated_cone);
	bool FindBestNeighborRadiiOffsetsMultiThreading(const TruncatedConePrimitive& truncated_cone, const TruncatedConePrimitive& original_truncated_cone, TruncatedConePrimitive* best_neighbor_truncated_cone);
public:
	friend void ThreadHelperFunc(ThinStructureReconstructor* thin_structure_reconstructor, TruncatedConePrimitive* truncated_cone, double* result);
};

#endif