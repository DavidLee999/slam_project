#ifndef SLAM_RGBDODOMETRY_H
#define SLAM_RGBDODOMETRY_H

#include <vector>

#include <opencv2/core/core.hpp>

struct RgbdFrame
{
    RgbdFrame();

    RgbdFrame(const cv::Mat &img,
              const cv::Mat &depth,
              const cv::Mat &mask = cv::Mat(),
              int id = -1);

    virtual ~RgbdFrame();

    virtual void release();

    int id_;
    cv::Mat img_;
    cv::Mat depth_;
    cv::Mat mask_;
};

struct OdometryFrame : public RgbdFrame
{
    OdometryFrame();

    OdometryFrame(const cv::Mat &img,
                  const cv::Mat &depth,
                  const cv::Mat &mask = cv::Mat(),
                  int id = -1);

    virtual void release();

    void releasePyramids();

    std::vector<cv::Mat> pyramidImg_;
    std::vector<cv::Mat> pyramidDepth_;
    std::vector<cv::Mat> pyramidMask_;

    std::vector<cv::Mat> pyramid_dIdx_;
    std::vector<cv::Mat> pyramid_dIdy_;
    std::vector<cv::Mat> pyramidTextureMask_;

};


class RGBDOdometry
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

    void prepareOdometryFrame(OdometryFrame &frame, int cacheType);

    bool compute(const cv::Mat &srcImg,
                 const cv::Mat srcDepth,
                 const cv::Mat &srcMask,
                 const cv::Mat &dstImg,
                 const cv::Mat $dstDepth,
                 const cv::Mat &dstMask,
                 cv::Mat &Rt,
                 const cv::Mat &initRt = cv::Mat());

protected:

    bool computeImpl(const OdometryFrame &srcFrame,
                     const OdometryFrame &dstFrame,
                     cv::Mat &Rt,
                     const cv::Mat &initRt);

    double minDepth_;
    double maxDepth_;
    double maxDepthDiff_;
    std::vector<int> iterCounts_;
    std::vector<float> minGradMag_;

    cv::Mat cameraMatrix_;
    double maxPointPart_;

    double maxTranslation_;
    double maxRotation_;
};


#endif //SLAM_RGBDODOMETRY_H
