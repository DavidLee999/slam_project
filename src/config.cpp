#include <iostream>
#include <config.h>


#include "config.h"

bool Config::setParameterFile(const std::string &filename)
{
	if (nullptr == config_)
		config_ = std::shared_ptr<Config>(new Config);

	config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);

	if(false == config_->file_.isOpened())
	{
		std::cerr << "file " << filename << "does not exist!" << std::endl;
		config_->file_.release();
		return false;
	}

	return true;
}

Config::~Config()
{
	if (file_.isOpened())
		file_.release();
}

std::shared_ptr<Config> Config::config_ = nullptr;
