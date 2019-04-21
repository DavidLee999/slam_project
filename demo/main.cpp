#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "config.h"

bool readAssociateFile(const std::string 	     config_file,
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
		cx 			= Config::get<float>("camera.cx");
		depth_scale = Config::get<float>("camera.depth_scale");
}

void run_slam(const std::vector<std::string> &rgb_files,
			  const std::vector<double>      &rgb_times,
			  const std::vector<std::string> &depth_files,
			  const std::vector<double>      &depth_times)
{
	// read camera params
	float fx, fy, cx, cy, depth_scale;
	readCameraParams(fx, fy, cx, cy, depth_scale);

	// display
	cv::namedWindow("RGBD Color", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("RGBD Depth", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);
	cv::Mat traj = cv::Mat::zeros(800, 800, CV_8UC3);

	// display params
	char text[100];
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	cv::Point textOrg(10, 50);

	float camera[] = { fx, 0.0f, cx, fy, 0.0f, cy, 0.0f, 0.0f, 1.0f };
	cv::Mat cameraMatrix(3, 3, CV_32FC1, camera);
	cv::Mat rotationMatrix, translationMatrix;
	// std::shared_ptr<RGBDOdometry> odom;

	bool isFirst = true;
	for (int i = 0; i != rgb_files.size() - 1; ++i)
	{
		// read image
		cv::Mat color_img0 = cv::imread(rgb_files[i], cv::IMREAD_GRAYSCALE);
		cv::Mat depth_img0 = cv::imread(depth_files[i], cv::IMREAD_UNCHANGED);
		cv::Mat color_img1 = cv::imread(rgb_files[i + 1], cv::IMREAD_GRAYSCALE);
		cv::Mat depth_img1 = cv::imread(depth_files[i + 1], cv::IMREAD_UNCHANGED);

		// convert depth unit from millimeter to meter
		cv::Mat depthFlt0, depthFlt1;
		depth_img0.convertTo(depthFlt0, CV_32FC1, depth_scale);
		depth_img1.convertTo(depthFlt1, CV_32FC1, depth_scale);

		cv::Mat rigidTransform;

		// if (!odom)
		{
			std::vector<int> iterCounts(4);
			iterCounts[0] = 7;
			iterCounts[1] = 7;
			iterCounts[2] = 7;
			iterCounts[3] = 10;

			std::vector<int> minGradMags(4);
			minGradMags[0] = 12;
			minGradMags[1] = 5;
			minGradMags[2] = 3;
			minGradMags[3] = 1;

			// initialize
			// odom =

			std::cout << "initialize tracker" << std::endl;
		}
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
	bool successed = readAssociateFile(argv[1], rgb_files, rgb_times, depth_files, depth_times);
	if (!successed)
	{
		std::cerr << "Read Files Error." << std::endl;
		return -1;
	}





	return 0;
}
