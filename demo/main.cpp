#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "config.h"
#include "RGBDOdometry.h"

bool readAssociateFile(const std::string 	    &config_file,
					   std::vector<std::string> &rgb_files,
					   std::vector<double> 		&rgb_times,
					   std::vector<std::string> &depth_files,
					   std::vector<double> 		&depth_times)
{
	// open associate file
	if (!Config::setParameterFile(config_file))
		return false;

	std::string dataset_dir = Config::get<std::string>( "dataset_dir" );

	std::ifstream fin( dataset_dir + "associate.txt" );
	if (!fin)
	{
		std::cerr << "Please generate the associate file called associate.txt!" << std::endl;
		return false;
	}

	// read file
	std::cout << "read files ..." << std::endl;
	while (!fin.eof())
	{
		std::string rgb_time, rgb_file, depth_time, depth_file;
		fin >> rgb_time >> rgb_file >> depth_time >> depth_file;

		rgb_times.push_back(atof(rgb_time.c_str()));
		rgb_files.push_back(dataset_dir + rgb_file);
		depth_time.push_back(atof(depth_time.c_str()));
		depth_files.push_back(dataset_dir + depth_file);
	}

	fin.close();
	return true;
}

void readCameraParams(float 			&fx,
					  float				&fy,
					  float 			&cx,
					  float				&cy,
					  float				&depth_scale)
{
		fx 			= Config::get<float>("camera.fx");
		fy 			= Config::get<float>("camera.fy");
		cx 			= Config::get<float>("camera.cx");
		cy 			= Config::get<float>("camera.cy");
		depth_scale = Config::get<float>("camera.depth_scale");
}

void readVOParams(float &min_depth,
                  float &max_depth,
                  float &max_depth_diff,
                  float &max_points_part,
                  float &max_translation,
                  int   &max_rotation)
{
    min_depth       = Config::get<float>("min_depth");
    max_depth       = Config::get<float>("max_depth");
    max_depth_diff  = Config::get<float>("max_depth_diff");
    max_points_part = Config::get<float>("max_point_ratio");
    max_translation = Config::get<float>("max_translation");
    max_rotation    = Config::get<int>("max_rotation");
}

void run_slam(const std::vector<std::string> &rgb_files,
			  const std::vector<double>      &rgb_times,
			  const std::vector<std::string> &depth_files,
			  const std::vector<double>      &depth_times)
{
	// read camera params
	float fx, fy, cx, cy, depth_scale;
	readCameraParams(fx, fy, cx, cy, depth_scale);

	// read VO paras
	float min_depth, max_depth, max_depth_diff, max_points_part, max_translation;
	int max_rotation;
	readVOParams(min_depth, max_depth, max_depth_diff, max_points_part, max_translation, max_rotation);

	std::vector<int> iterCounts(4);
	iterCounts[0] = 7;
	iterCounts[1] = 7;
	iterCounts[2] = 7;
	iterCounts[3] = 10;

	std::vector<float> minGradMags(4);
	minGradMags[0] = 12.f;
	minGradMags[1] = 5.f;
	minGradMags[2] = 3.f;
	minGradMags[3] = 1.f;

	// display
	cv::namedWindow("RGBD Color", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("RGBD Depth", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);
	cv::Mat traj = cv::Mat::zeros(800, 800, CV_8UC3);

	// display params
	char text[100];
	int fontFace     = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness    = 1;
	cv::Point textOrg(10, 50);

	float camera[] = { fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f };
	cv::Mat cameraMatrix(3, 3, CV_32FC1, camera);
	cv::Mat rotationMatrix, translationMatrix;

	RGBDOdometry odom(cameraMatrix, min_depth, max_depth, max_depth_diff, max_translation, max_rotation,
			max_points_part, iterCounts, minGradMags);

	bool isFirst = true;
	for (int i = 0; i != rgb_files.size() - 1; ++i)
	{
		// read image
		cv::Mat color_img0 = cv::imread(rgb_files[i]);
		cv::Mat depth_img0 = cv::imread(depth_files[i], cv::IMREAD_UNCHANGED);
		cv::Mat color_img1 = cv::imread(rgb_files[i + 1]);
		cv::Mat depth_img1 = cv::imread(depth_files[i + 1], cv::IMREAD_UNCHANGED);

		// convert depth unit from millimeter to meter
		cv::Mat depthFlt0, depthFlt1;
		depth_img0.convertTo(depthFlt0, CV_32FC1, depth_scale);
		depth_img1.convertTo(depthFlt1, CV_32FC1, depth_scale);
		cv::cvtColor(color_img0, color_img0, cv::COLOR_BGR2GRAY);
        cv::cvtColor(color_img1, color_img1, cv::COLOR_BGR2GRAY);

		cv::Mat rigidTransform;

		bool isSuccess = odom.compute(color_img0, depthFlt0, cv::Mat(),
				color_img1, depthFlt1, cv::Mat(), rigidTransform);

		cv::Mat rotationMat    = rigidTransform(cv::Rect(0, 0, 3, 3)).clone();
		cv::Mat translationMat = rigidTransform(cv::Rect(3, 0, 1, 3)).clone();

		if (isSuccess) {
			if (isFirst) {
				rotationMatrix = rotationMat.clone();
				translationMatrix = translationMat.clone();
				isFirst = false;
				continue;
			}

			// update R and t
			translationMatrix = translationMatrix + rotationMatrix * translationMat;
			rotationMatrix = rotationMat * rotationMatrix;
		}

		// visualize trajectory
		if (!isFirst)
		{
			int x = static_cast<int>(60.f * translationMatrix.at<double>(0)) + 800 / 2;
			int y = static_cast<int>(60.f * translationMatrix.at<double>(2)) + 800 / 2;

			cv::circle(traj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);
			cv::rectangle(traj, cv::Point(10, 30), cv::Point(559, 50), CV_RGB(0, 0, 0), CV_FILLED);

			if (isSuccess)
			{
				sprintf(text, "Coordinates: x = %04fm y=%04fm z = %04fm",
						translationMatrix.at<double>(0),
						translationMatrix.at<double>(1),
						translationMatrix.at<double>(2));
			}
			else
			{
				sprintf(text, "Fail to compute odometry");
			}

			cv::putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 0);
		}

		cv::imshow("Trajectory", traj);
		cv::imshow("RGBD Color", color_img1);

		cv::Mat depth_show;
		depthFlt1.convertTo(depth_show, CV_8UC1, 255.0 / max_depth);
		cv::imshow("RGBD Depth", depth_show);

		cv::waitKey(10);
	}

}

int main(int argc, char **argv)
{
	// usage
	std::cout << "usage: slam path_to_config" << std::endl;
	if (2 != argc)
	{
		std::cerr << "Error." << std::endl;
		return -1;
	}

	std::vector<std::string> rgb_files, depth_files;
	std::vector<double> rgb_times, depth_times;
	bool isSuccess = readAssociateFile(argv[1], rgb_files, rgb_times, depth_files, depth_times);
	if (!isSuccess)
	{
		std::cerr << "Read Files Error." << std::endl;
		return -1;
	}

	run_slam(rgb_files, rgb_times, depth_files, depth_times);

	return 0;
}
