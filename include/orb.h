#ifndef SLAM_ORB_H
#define SLAM_ORB_H

#include <vector>

#include <opencv2/features2d/features2d.hpp>

namespace myslam
{
    class OrbFeature
    {
    public:
        OrbFeature();

        OrbFeature(int      n_features,
                   float    scaleFactor,
                   int      n_levels,
                   int      fastThreshold_low,
                   int      fastThreshold_high,
                   double   knn_factor);

        // to-do: class Frame
        void detectOrbFeatures(const cv::Mat &input_img,
                               std::vector<cv::KeyPoint> &kps,
                               cv::Mat &descriptors) const;

         std::vector<cv::DMatch> matchOrbFeatures(const cv::Mat &descriptors_prev,
                                                  const cv::Mat &descriptors_curr) const;

    protected:
        cv::Ptr<cv::FeatureDetector> orb_detector_;
        cv::Ptr<cv::DescriptorMatcher> orb_matcher_;

        double knn_factor_;
        int fastThreshold_low_;
        int fastThreshold_high_;
    };
}

#endif //SLAM_ORB_H
