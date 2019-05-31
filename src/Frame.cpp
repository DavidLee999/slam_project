#include "Frame.h"
#include "orb.h"
using namespace myslam;

Frame::Frame(const cv::Mat &rgb_frame, const cv::Mat &depth_frame,
        std::shared_ptr<OrbFeature> _feature_extractor, int _id, float _depth_scale)
        : id (_id), depth_scale (_depth_scale)
{
    if (rgb_frame.empty() || depth_frame.empty())
        return;

    feature_extractor = _feature_extractor;

    rows = rgb_frame.rows;
    cols = rgb_frame.cols;

    _feature_extractor->detectOrbFeatures(rgb_frame, features, descriptors, borderSize);

    num_features = features.size();

    outliers = std::vector<bool>(num_features, false);

    depths = std::vector<ushort>(num_features, 0);
    for (int i = 0; i < num_features; ++i)
    {
        int x = cvRound(features[i].pt.x);
        int y = cvRound(features[i].pt.y);

        ushort z = depth_frame.at<ushort>(y, x);
        if (0 != z)
            depths[i] = z;
    }
}
