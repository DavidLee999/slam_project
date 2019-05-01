#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>

#include "RGBDOdometry.h"

static void buildPyramidCameraMatrix(const cv::Mat &cameraMatrix,
                                     size_t levels,
                                     std::vector<cv::Mat> pyramidCameraMatrix)
{
    cv::Mat tmp_cameraMatrix;

    pyramidCameraMatrix.reserve(levels);
    cameraMatrix.convertTo(tmp_cameraMatrix, CV_64FC1);

    for (size_t i = 0; i < levels; ++i)
    {
        cv::Mat levelCameraMatrix = 0 == i ? tmp_cameraMatrix : 0.5f * pyramidCameraMatrix[i - 1];
        levelCameraMatrix.at<double>(2, 2) = 1.0;
        pyramidCameraMatrix[i] = levelCameraMatrix;
    }

}

void depthMat2PointCloud(const cv::Mat &depth_map,
                         const cv::Mat &cameraMatrix,
                         cv::Mat &pointCloud)
{
    const float fx_inv = 1.0f / cameraMatrix.at<float>(0, 0);
    const float fy_inv = 1.0f / cameraMatrix.at<float>(1, 1);
    const float cx     = cameraMatrix.at<float>(0, 2);
    const float cy     = cameraMatrix.at<float>(1, 2);

    // pre-compute some constants for speeding up
    cv::Mat_<float> x_cache(1, depth_map.cols), y_cache(depth_map.rows, 1);
    float *x_ptr = x_cache[0], *y_ptr = y_cache[0];

    for (int x = 0; x < depth_map.cols; ++x, ++x_ptr)
        *x_ptr = (x - cx) * fx_inv;
    for (int y = 0; y < depth_map.rows; ++y, ++y_ptr)
        *y_ptr = (y - cy) * fy_inv;

    // compute 3D point cloud
    y_ptr = y_cache[0];
    for (int y = 0; y < depth_map.rows; ++y, ++y_ptr)
    {
        cv::Vec3f   *point     = pointCloud.ptr<cv::Vec3f>(y);
        const float *x_ptr_end = x_cache[0] + depth_map.cols;
        const float *depth     = depth_map.ptr<float>(y);
        for (x_ptr = x_cache[0]; x_ptr_end != x_ptr; ++x_ptr, ++point, ++depth)
        {
            float z     = *depth;
            (*point)[0] = (*x_ptr) * z;
            (*point)[1] = (*y_ptr) * z;
            (*point)[2] = z;
        }
    }
}

static void buildPyramidCloud(const std::vector<cv::Mat> &pyramidDepth,
                              const cv::Mat &cameraMatrix,
                              std::vector<cv::Mat> &pyramidCloud)
{
    std::vector<cv::Mat> pyramidCameraMatrix;
    buildPyramidCameraMatrix(cameraMatrix, static_cast<int>( pyramidDepth.size() ), pyramidCameraMatrix);

    pyramidCloud.reserve(pyramidDepth.size());
    for (size_t i = 0; i < pyramidDepth.size(); ++i)
    {
        cv::Mat cloud;

    }
}

static void buildPyramidMask(const cv::Mat &mask,
                             const std::vector<cv::Mat> &pyramidDepth,
                             const double minDepth,
                             const double maxDepth,
                             std::vector<cv::Mat> &pyramidMask)
{
    cv::Mat validMask = mask.clone();

    cv::buildPyramid(validMask, pyramidMask, static_cast<int>( pyramidDepth.size() - 1 ));

    for (size_t i = 0; i < pyramidMask.size(); ++i)
    {
        cv::Mat levelDepth = pyramidDepth[i].clone();
        cv::Mat &levelMask = pyramidMask[i];

        cv::patchNaNs(levelDepth, 0);
        levelMask &= (levelDepth > minDepth) & (levelDepth < maxDepth);
    }
}

/*****************
 * RgbdFrame
 *****************/
RgbdFrame::RgbdFrame() : id_ { -1 }
{ }

RgbdFrame::RgbdFrame(const cv::Mat &img, const cv::Mat &depth, const cv::Mat &mask, int id)
    : id_ (id), img_ (img), depth_ (depth), mask_ (mask)
{ }

RgbdFrame::~RgbdFrame()
{ }

void RgbdFrame::release()
{
    id_ = -1;
    img_.release();
    depth_.release();
    mask_.release();
}

/*****************
 *OdometryFrame
 *****************/
OdometryFrame::RgbdFrame() : RgbdFrame()
{ }

OdometryFrame::OdometryFrame(const cv::Mat &img, const cv::Mat &depth, const cv::Mat &mask, int id)
    : RgbdFrame(img, depth, mask, id)
{ }

void OdometryFrame::release()
{
    RgbdFrame::release();
    releasePyramids();
}

void OdometryFrame::releasePyramids()
{
    pyramidImg_.clear();
    pyramidDepth_.clear();
    pyramidMask_.clear();
    pyramidCloud_.clear();
    pyramid_dIdx_.clear();
    pyramid_dIdy_.clear();
    pyramidTextureMask_.clear();
}

/*****************
 *RGBDOdometry
 *****************/

RGBDOdometry::RGBDOdometry(const cv::Mat &cameraMatrix, float minDepth, float maxDepth, float maxDepthDiff,
                           double maxTranslation, double maxRotation, const std::vector<int> &iterCounts,
                           const std::vector<float> &minGradMag, float maxPointsPart) :
        minDepth_ (minDepth), maxDepth_ (maxDepth), maxDepthDiff_ (maxDepthDiff),
        iterCounts_ (iterCounts), minGradMag_ (minGradMag), cameraMatrix_ (cameraMatrix),
        maxPointPart_ (maxPointsPart), maxTranslation_ (maxTranslation), maxRotation_ (maxRotation)
{

}


void RGBDOdometry::prepareOdometryFrame(OdometryFrame &frame, int cacheType)
{
    cv::buildPyramid(frame.img_, frame.pyramidImg_, static_cast<int>( iterCounts_.size() - 1 ));
    cv::buildPyramid(frame.depth_, frame.pyramidDepth_, static_cast<int>( iterCounts_.size() - 1 ));
    buildPyramidMask(frame.mask_, frame.pyramidDepth_, minDepth_, maxDepth_, frame.pyramidMask_);

    if (OdometryFrame::CACHE_SRC & cacheType)
        prepareOdometryFrame();
    if (OdometryFrame::CACHE_DST & cacheType)
        prepareOdometryFrame();

}

bool RGBDOdometry::compute(const cv::Mat &srcImg, const cv::Mat &srcDepth, const cv::Mat &srcMask, const cv::Mat &dstImg,
                           const cv::Mat &dstDepth, const cv::Mat &dstMask, cv::Mat &Rt, const cv::Mat &initRt)
{
    OdometryFrame srcFrame(srcImg, srcDepth, srcMask);
    OdometryFrame dstFrame(dstImg, dstDepth, dstMask);
}