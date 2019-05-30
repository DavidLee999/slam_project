#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <algorithm>

#include "orb.h"

using namespace myslam;

OrbFeature::OrbFeature() : knn_factor_ (0.8), fastThreshold_high_ (20), fastThreshold_low_ (5)
{
    orb_detector_ = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, fastThreshold_high_);
    orb_matcher_  = cv::DescriptorMatcher::create("BruteForce-Hamming");
}

OrbFeature::OrbFeature(int n_features, float scaleFactor, int n_levels, int fastThreshold_low, int fastThreshold_high,
        double knn_factor)
        : knn_factor_ (knn_factor), fastThreshold_low_ (fastThreshold_low), fastThreshold_high_ (fastThreshold_high)
{
    orb_detector_ = cv::ORB::create(n_features, scaleFactor, n_levels, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, fastThreshold_high_);
    orb_matcher_  = cv::DescriptorMatcher::create("BruteForce-Hamming");
}

void OrbFeature::detectOrbFeatures(const cv::Mat &input_img, std::vector<cv::KeyPoint> &kps, cv::Mat &descriptors, int border_size) const
{
    cv::Mat frame = input_img;
    if (1 != input_img.channels())
        cv::cvtColor(input_img, frame, cv::COLOR_BGR2GRAY);

    cv::Mat mask;
    if (0 != border_size)
    {
        cv::Mat _mask(frame.size(), CV_8UC1, cv::Scalar(0));
        _mask(cv::Rect(border_size, border_size, frame.cols - 2 * border_size, frame.rows - 2 * border_size)).setTo(255);
        _mask.copyTo(mask);
    }
    else
    {
        cv::Mat _mask(frame.size(), CV_8UC1, cv::Scalar(255));
        _mask.copyTo(mask);
    }

    orb_detector_->detectAndCompute(frame, mask, kps, descriptors);
}

std::vector<cv::DMatch> OrbFeature::matchOrbFeatures(const cv::Mat &descriptors_prev, const cv::Mat &descriptors_curr) const
{
    std::vector<std::vector<cv::DMatch> > knn_matches;
    orb_matcher_->knnMatch(descriptors_prev, descriptors_curr, knn_matches, 2);

    std::vector<cv::DMatch> matches;
    matches.reserve(knn_matches.size());
    for (size_t i = 0; i < knn_matches.size(); ++i)
    {
        if (knn_matches[i][0].distance < knn_factor_ * knn_matches[i][1].distance)
            matches.push_back(knn_matches[i][0]);
    }

    // uniqueness match.
    // do by epipolar geometry.
    // std::sort(matches.begin(), matches.end(),
    //         [](cv::DMatch match1, cv::DMatch match2){ return match1.trainIdx < match2.trainIdx; });

    // std::vector<cv::DMatch> matches_inliers;
    // matches_inliers.reserve(matches.size());
    // matches_inliers.push_back(matches[0])
    // for (size_t i = 1; i < matches.size(); ++i)
    // {
    //     if (matches[i].trainIdx == matches[i - 1].trainIdx)
    // }

    return matches;
}