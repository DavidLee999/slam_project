#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>

#include "RGBDOdometry.h"

/*****************
 * helper function
 *****************/
static void buildPyramidCameraMatrix(const cv::Mat &cameraMatrix,
                                     size_t levels,
                                     std::vector<cv::Mat> &pyramidCameraMatrix)
{
    cv::Mat tmp_cameraMatrix;

    pyramidCameraMatrix.resize(levels);
    // cameraMatrix.convertTo(tmp_cameraMatrix, CV_64FC1);
    cameraMatrix.copyTo(tmp_cameraMatrix);

    for (size_t i = 0; i < levels; ++i)
    {
        cv::Mat levelCameraMatrix = 0 == i ? tmp_cameraMatrix : 0.5f * pyramidCameraMatrix[i - 1];
        levelCameraMatrix.at<float>(2, 2) = 1.0f;
        pyramidCameraMatrix[i] = levelCameraMatrix;
    }

}

static void depthMat2PointCloud(const cv::Mat &depth_map,
                                const cv::Mat &cameraMatrix,
                                cv::Mat &pointCloud)
{
    const float fx_inv = 1.0f / cameraMatrix.at<float>(0, 0);
    const float fy_inv = 1.0f / cameraMatrix.at<float>(1, 1);
    const float cx     = cameraMatrix.at<float>(0, 2);
    const float cy     = cameraMatrix.at<float>(1, 2);
    pointCloud.create(depth_map.size(), CV_32FC3);
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

    pyramidCloud.resize(pyramidDepth.size());
    for (size_t i = 0; i < pyramidDepth.size(); ++i)
    {
        cv::Mat pointcloud;
        depthMat2PointCloud(pyramidDepth[i], pyramidCameraMatrix[i], pointcloud);
        pyramidCloud[i] = pointcloud;
    }
}

static void buildPyramidMask(const cv::Mat &mask,
                             const std::vector<cv::Mat> &pyramidDepth,
                             const double minDepth,
                             const double maxDepth,
                             std::vector<cv::Mat> &pyramidMask)
{
    cv::Mat validMask = cv::Mat(pyramidDepth[0].size(), CV_8UC1, cv::Scalar(255));

    cv::buildPyramid(validMask, pyramidMask, static_cast<int>( pyramidDepth.size() - 1 ));

    for (size_t i = 0; i < pyramidMask.size(); ++i)
    {
        cv::Mat levelDepth = pyramidDepth[i].clone();
        cv::Mat &levelMask = pyramidMask[i];

        cv::patchNaNs(levelDepth, 0);
        levelMask &= (levelDepth > minDepth) & (levelDepth < maxDepth);
    }
}

static void buildPyramidSobel(const std::vector<cv::Mat> &pyramidImg,
                              int dx,
                              int dy,
                              std::vector<cv::Mat> &pyramidSobel)
{
    pyramidSobel.resize(pyramidImg.size());
    for (size_t i = 0; i < pyramidImg.size(); ++i)
    {
        cv::Sobel(pyramidImg[i], pyramidSobel[i], CV_16S, dx, dy);
    }
}

static void randomSubsetOfMask(cv::Mat &mask,
                               float part)
{
    const int minPointsCount = 1000;
    const int nonzeros       = cv::countNonZero(mask);
    const int needCount      = std::max(minPointsCount, static_cast<int>(mask.total() * part));

    if (needCount < nonzeros)
    {
        cv::RNG rng;
        cv::Mat maskSubset(mask.size(), CV_8UC1, cv::Scalar(0));
        int     subsetCnt = 0;
        while (subsetCnt < needCount)
        {
            int y = rng(mask.rows);
            int x = rng(mask.cols);
            if (mask.at<uchar>(y, x))
            {
                maskSubset.at<uchar>(y, x) = 255;
                mask.at<uchar>(y, x) = 0;
                subsetCnt++;
            }
        }

        mask = maskSubset;
    }
}

static void buildPyramidTexturedMask(const std::vector<cv::Mat> &pyramid_dIdx,
                                     const std::vector<cv::Mat> &pyramid_dIdy,
                                     const std::vector<float> &minGradMags,
                                     const std::vector<cv::Mat> &pyramidMask,
                                     double maxPointsPart,
                                     std::vector<cv::Mat> &pyramidTexturedMask)
{
    const double sobelScale      = 1.0 / 8.0;
    const float  sobelScale2_inv = 1.f / static_cast<float>(sobelScale * sobelScale);
    pyramidTexturedMask.resize(pyramid_dIdx.size());

    for (size_t i = 0; i < pyramidTexturedMask.size(); ++i)
    {
        const float    minScaledGradMag2 = minGradMags[i] * minGradMags[i] * sobelScale2_inv;
        const cv::Mat &dIdx              = pyramid_dIdx[i];
        const cv::Mat &dIdy              = pyramid_dIdy[i];
        cv::Mat        texturedMask(dIdx.size(), CV_8UC1, cv::Scalar(0));

        for (int y = 0; y < dIdx.rows; ++y)
        {
            const short *dIdx_row         = dIdx.ptr<short>(y);
            const short *dIdy_row         = dIdy.ptr<short>(y);
            uchar       *texturedMask_row = texturedMask.ptr<uchar>(y);

            for (int x = 0; x < dIdx.cols; ++x)
            {
                float magnitude2 = static_cast<float>(dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x]);
                if (magnitude2 >= minScaledGradMag2)
                    texturedMask_row[x] = 255;
            }
        }

        pyramidTexturedMask[i] = texturedMask & pyramidMask[i];
        randomSubsetOfMask(pyramidTexturedMask[i], static_cast<float>(maxPointsPart));
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
OdometryFrame::OdometryFrame() : RgbdFrame()
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
                           float maxTranslation, int maxRotation, float maxPointsPart,
                           const std::vector<int> &iterCounts, const std::vector<float> &minGradMag) :
        minDepth_ (minDepth), maxDepth_ (maxDepth), maxDepthDiff_ (maxDepthDiff),
        iterCounts_ (iterCounts), minGradMag_ (minGradMag), cameraMatrix_ (cameraMatrix),
        maxPointsPart_ (maxPointsPart), maxTranslation_ (maxTranslation), maxRotation_ (maxRotation)
{

}


void RGBDOdometry::prepareOdometryFrame(OdometryFrame &frame, int cacheType)
{
    cv::buildPyramid(frame.img_, frame.pyramidImg_, static_cast<int>( iterCounts_.size() - 1 ));
    cv::buildPyramid(frame.depth_, frame.pyramidDepth_, static_cast<int>( iterCounts_.size() - 1 ));
    buildPyramidMask(frame.mask_, frame.pyramidDepth_, minDepth_, maxDepth_, frame.pyramidMask_);

    if (OdometryFrame::CACHE_SRC & cacheType)
        buildPyramidCloud(frame.pyramidDepth_, cameraMatrix_, frame.pyramidCloud_);
    if (OdometryFrame::CACHE_DST & cacheType)
    {
        buildPyramidSobel(frame.pyramidImg_, 1, 0, frame.pyramid_dIdx_);
        buildPyramidSobel(frame.pyramidImg_, 0, 1, frame.pyramid_dIdy_);
        buildPyramidTexturedMask(frame.pyramid_dIdx_, frame.pyramid_dIdy_, minGradMag_, frame.pyramidMask_,
                maxPointsPart_, frame.pyramidTextureMask_);
    }

}

bool RGBDOdometry::compute(const cv::Mat &srcImg, const cv::Mat &srcDepth, const cv::Mat &srcMask, const cv::Mat &dstImg,
                           const cv::Mat &dstDepth, const cv::Mat &dstMask, cv::Mat &Rt, const cv::Mat &initRt)
{
    OdometryFrame srcFrame(srcImg, srcDepth, srcMask);
    OdometryFrame dstFrame(dstImg, dstDepth, dstMask);

    bool isSuccess = computeImpl(srcFrame, dstFrame, Rt, initRt);

    return isSuccess;
}

bool RGBDOdometry::computeImpl(OdometryFrame &srcFrame, OdometryFrame &dstFrame, cv::Mat &Rt, const cv::Mat &initRt)
{
    prepareOdometryFrame(srcFrame, OdometryFrame::CACHE_SRC);
    prepareOdometryFrame(dstFrame, OdometryFrame::CACHE_DST);

    return true;
}