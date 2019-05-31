#ifndef SLAM_FRAME_H
#define SLAM_FRAME_H

#include <vector>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>=

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "orb.h"

namespace myslam
{
    class Frame
    {
    public:

        typedef std::shared_ptr<Frame> Ptr;

        int id = -1;

        int borderSize = 31;
        int rows;
        int cols;

        Eigen::Isometry3d Tcw;

        std::shared_ptr<OrbFeature> feature_extractor;

        int num_features;
        std::vector<cv::KeyPoint>  features;
        cv::Mat descriptors;
        std::vector<bool> outliers;

        std::vector<ushort> depths;
        float depth_scale = 0.002;

        // to-do: MapPoint Class
        std::vector<cv::Point3f> map_points;

    public:

        Frame(const cv::Mat &rgb_frame,
              const cv::Mat &depth_frame,
              std::shared_ptr<OrbFeature> _feature_extractor,
              int _id = -1,
              float _depth_scale = 0.002);

        void setTransformation(const Eigen::Isometry3d & T)
        {
            Tcw = T;
        }

        Eigen::Isometry3d getTransformation()
        {
            return Tcw;
        }

        bool isInFrame(cv::Point3f mapPoint)
        {
        }

    };
}

#endif //SLAM_FRAME_H
