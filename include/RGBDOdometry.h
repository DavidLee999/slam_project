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
    enum
    {
        CACHE_SRC = 1, CACHE_DST = 2, CACHE_ALL = CACHE_SRC + CACHE_DST
    };

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
    std::vector<cv::Mat> pyramidCloud_;
    std::vector<cv::Mat> pyramidTextureMask_;

};


class RGBDOdometry
{
public:

    RGBDOdometry(const cv::Mat &cameraMatrix,
                 float minDepth,
                 float maxDepth,
                 float maxDepthDiff,
                 float maxTranslation,
                 int   maxRotation,
                 float maxPointsPart,
                 const std::vector<int> &iterCounts = std::vector<int>(),
                 const std::vector<float> &minGradMag = std::vector<float>());

    bool compute(const cv::Mat &srcImg,
                 const cv::Mat &srcDepth,
                 const cv::Mat &srcMask,
                 const cv::Mat &dstImg,
                 const cv::Mat &dstDepth,
                 const cv::Mat &dstMask,
                 cv::Mat &Rt,
                 const cv::Mat &initRt = cv::Mat());

protected:


    void prepareOdometryFrame(OdometryFrame &frame, int cacheType);

    bool computeImpl(OdometryFrame &srcFrame,
                     OdometryFrame &dstFrame,
                     cv::Mat &Rt,
                     const cv::Mat &initRt);

    float minDepth_;
    float maxDepth_;
    float maxDepthDiff_;
    std::vector<int> iterCounts_;
    std::vector<float> minGradMag_;

    cv::Mat cameraMatrix_;
    float maxPointsPart_;

    float maxTranslation_;
    int maxRotation_;
};


#endif //SLAM_RGBDODOMETRY_H
