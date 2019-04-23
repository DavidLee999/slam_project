#ifndef SLAM_RGBDODOMETRY_H
#define SLAM_RGBDODOMETRY_H

#include <opencv2/core/core.hpp>

class Odometry // interface
{
    // depth range, meters
    static inline float
    DEFAULT_MIN_DEPTH()
    {
        return 0.0f;
    }

    static inline float
    DEFAULT_MAX_DEPTH()
    {
        return 4.0f;
    }

    static inline float
    DEFAULT_MAX_DEPTH_DIFF()
    {
        return 0.07f;
    }

    // process parameters
    static inline float
    DEFAULT_MAX_POINTS_PART() // how many points will be processed
    {
        return 0.07f;
    }

    static inline float
    DEFAULT_MAX_TRANSLATION()
    {
        return 0.15f;
    }

    static inline float
    DEFAULT_MAX_ROTATION()
    {
        return 15; // degree
    }
};

class RGBDOdometry : public Odometry
{
public:

    RGBDOdometry();

    RGBDOdometry(const cv::Mat &cameraMatrix,
                 float minDepth,
                 float maxDepth,
                 float maxDepthDiff,
                 const std::vector<int> &iterCount = std::vector<int>(),
                 const std::vector<float> &minGradMag = std::vector<float>(),
                 float maxPointsPart = DEFAULT);
};


#endif //SLAM_RGBDODOMETRY_H
