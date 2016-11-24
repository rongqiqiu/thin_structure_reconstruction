#include "basic_types.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

vector<TruncatedConePrimitive> ImportTruncatedConePrimitives(const string& file_name) {
	vector<TruncatedConePrimitive> truncated_cones;
	ifstream in_stream;
	in_stream.open(file_name);
	TruncatedConePrimitive truncated_cone;
	string line;
	while (getline(in_stream, line)) {
		istringstream iss(line);
		iss >> truncated_cone.pa.x >> truncated_cone.pa.y >> truncated_cone.pa.z;
		iss >> truncated_cone.pb.x >> truncated_cone.pb.y >> truncated_cone.pb.z;
		iss >> truncated_cone.ra >> truncated_cone.rb;
		truncated_cones.push_back(truncated_cone);
	}
	in_stream.close();
	return truncated_cones;
}

void ExportTruncatedConePrimitives(const vector<TruncatedConePrimitive>& truncated_cones, const string& file_name) {
	ofstream out_stream;
	out_stream.open(file_name);
	for (int index = 0; index < truncated_cones.size(); ++index) {
		const TruncatedConePrimitive& truncated_cone = truncated_cones[index];
		out_stream << setprecision(8) << fixed << truncated_cone.pa.x << " " << truncated_cone.pa.y << " " << truncated_cone.pa.z << " ";
		out_stream << setprecision(8) << fixed << truncated_cone.pb.x << " " << truncated_cone.pb.y << " " << truncated_cone.pb.z << " ";
		out_stream << setprecision(8) << fixed << truncated_cone.ra << " " << truncated_cone.rb << endl;
	}
	out_stream.close();
}

bool IsMatchingPair(const TruncatedConePrimitive& output, const TruncatedConePrimitive& ground_truth) {
	Eigen::Vector3d va = output.pa.ToEigenVector() - ground_truth.pa.ToEigenVector();
	Eigen::Vector3d vb = output.pb.ToEigenVector() - ground_truth.pb.ToEigenVector();
	Eigen::Vector2d va_2d(va.x(), va.y());
	Eigen::Vector2d vb_2d(vb.x(), vb.y());

	return va_2d.norm() <= 1.0 && vb_2d.norm() <= 1.0 && fabs(va.z()) <= 3.0 && fabs(vb.z()) <= 3.0;
}

void AnalyzeDetections(const vector<TruncatedConePrimitive>& outputs, const vector<TruncatedConePrimitive>& ground_truths, const string& file_name, vector<TruncatedConePrimitive>* false_negatives, vector<TruncatedConePrimitive>* false_positives) {
	ofstream out_stream(file_name);
	out_stream << "All detections in output: " << outputs.size() << endl;
	int correct_output = 0;
	for (int i = 0; i < outputs.size(); ++i) {
		out_stream << "Output #" << i << ": ";
		bool is_matching = false;
		for (int j = 0; j < ground_truths.size(); ++j) {
			if (IsMatchingPair(outputs[i], ground_truths[j])) {
				is_matching = true;
				break;
			}
		}
		if (is_matching) {
			++correct_output;
			out_stream << "correct" << endl;
		} else {
			false_positives->push_back(outputs[i]);
			out_stream << "wrong" << endl;
		}
	}
	out_stream << "Correct detections: " << correct_output << endl;
	out_stream << "Precision: " << setprecision(6) << fixed << (double) correct_output / outputs.size() << endl;

	out_stream << "All detections in ground truth: " << ground_truths.size() << endl;
	int covered_ground_truths = 0;
	for (int i = 0; i < ground_truths.size(); ++i) {
		out_stream << "Ground truth #" << i << ": ";
		bool is_covered = false;
		for (int j = 0; j < outputs.size(); ++j) {
			if (IsMatchingPair(outputs[j], ground_truths[i])) {
				is_covered = true;
				break;
			}
		}
		if (is_covered) {
			++covered_ground_truths;
			out_stream << "covered" << endl;
		} else {
			false_negatives->push_back(ground_truths[i]);
			out_stream << "missed" << endl;
		}
	}
	out_stream << "Covered ground truths: " << covered_ground_truths << endl;
	out_stream << "Recall: " << setprecision(6) << fixed << (double) covered_ground_truths / ground_truths.size() << endl;
	out_stream.close();
}

int main(int argc, char** argv) {
	const string run_directory = argv[1];
	const string output_name = "all";
	const string ground_truth_name = "ground_truth";
	const string file_name = "truncated_cone_hypotheses_with_radii_offsets_extents.dat";

	const vector<TruncatedConePrimitive> outputs = ImportTruncatedConePrimitives(run_directory + output_name + "\\" + file_name);
	const vector<TruncatedConePrimitive> ground_truths = ImportTruncatedConePrimitives(run_directory + ground_truth_name + "\\" + file_name);

	vector<TruncatedConePrimitive> false_negatives;
	vector<TruncatedConePrimitive> false_positives;
	AnalyzeDetections(outputs, ground_truths, run_directory + "qauntitative_results.txt", &false_negatives, &false_positives);

	system(("mkdir " + run_directory + "false_negatives\\").c_str());
	ExportTruncatedConePrimitives(false_negatives, run_directory + "false_negatives\\" + file_name);
	system(("mkdir " + run_directory + "false_positives\\").c_str());
	ExportTruncatedConePrimitives(false_positives, run_directory + "false_positives\\" + file_name);

	return 0;
}