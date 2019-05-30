#include "Frame.h"
#include "orb.h"
using namespace myslam;

Frame::Frame(const cv::Mat &rgb_frame, const cv::Mat &depth_frame, OrbFeature &_feature_extractor, int _id)
        : id (_id)
{
    if (rgb_frame.empty() || depth_frame.empty())
        return;

    feature_extractor = _feature_extractor;

    rows = rgb_frame.rows;
    cols = rgb_frame.cols;


}
