#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <memory>

#include <opencv2/core/core.hpp>

class Config
{
private:

	static std::shared_ptr<Config> config_;
	cv::FileStorage file_;

	Config() {} // private, singleton

public:

	~Config();

	static bool setParameterFile(const std::string &filename);

	// getor
	template <typename T>
	static T get(const std::string &key)
	{
		return T(Config::config_->file_[key]);
	}

};

#endif // _CONFIG_H_
