#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/core/eigen.hpp>

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
    cameraMatrix.convertTo(tmp_cameraMatrix, CV_64FC1);
    // cameraMatrix.copyTo(tmp_cameraMatrix);

    for (size_t i = 0; i < levels; ++i)
    {
        cv::Mat levelCameraMatrix = 0 == i ? tmp_cameraMatrix : 0.5 * pyramidCameraMatrix[i - 1];
        levelCameraMatrix.at<double>(2, 2) = 1.0f;
        pyramidCameraMatrix[i] = levelCameraMatrix;
    }

}

static void depthMat2PointCloud(const cv::Mat &depth_map,
                                const cv::Mat &cameraMatrix,
                                cv::Mat &pointCloud)
{
    cv::Mat _cameraMatrix;
    cameraMatrix.convertTo(_cameraMatrix, CV_32FC1);
    const float fx_inv = 1.0f / _cameraMatrix.at<float>(0, 0);
    const float fy_inv = 1.0f / _cameraMatrix.at<float>(1, 1);
    const float cx     = _cameraMatrix.at<float>(0, 2);
    const float cy     = _cameraMatrix.at<float>(1, 2);
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
    cv::Mat validMask;
    if (mask.empty())
        validMask = cv::Mat(pyramidDepth[0].size(), CV_8UC1, cv::Scalar(255));
    else
        validMask = mask.clone();

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
        cv::Sobel(pyramidImg[i], pyramidSobel[i], CV_16S, dx, dy, 3);
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
        cv::RNG rng(42);
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
    const double sobelScale      = 0.125;
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

static inline void calcRgbdEquationCoeffs(double *C,
                                          double dIdx,
                                          double dIdy,
                                          const cv::Point3f &p3d,
                                          double fx,
                                          double fy)
{
    double invz = 1. / p3d.z;
    double v0 = dIdx * fx * invz;
    double v1 = dIdy * fy * invz;
    double v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

    C[0] = -p3d.z * v1 + p3d.y * v2;
    C[1] =  p3d.z * v0 - p3d.x * v2;
    C[2] = -p3d.y * v0 + p3d.x * v1;
    C[3] = v0;
    C[4] = v1;
    C[5] = v2;
}

static void calcRbgdLsmMatrices(const cv::Mat &img0,
                                const cv::Mat &cloud0,
                                const cv::Mat &Rt,
                                const cv::Mat &img1,
                                const cv::Mat &dI_dx1,
                                const cv::Mat &dI_dy1,
                                const cv::Mat &corresps,
                                float fx,
                                float fy,
                                cv::Mat &AtA,
                                cv::Mat &AtB)
{
    const int transformDim  = 6;
    const int correspsCount = corresps.rows;
    const double sobelScale = 0.125;
    AtA = cv::Mat(transformDim, transformDim, CV_64FC1, cv::Scalar(0));
    AtB = cv::Mat(transformDim, 1, CV_64FC1, cv::Scalar(0));

    double *AtB_ptr               = AtB.ptr<double>();
    const double *Rt_ptr          = Rt.ptr<const double>();
    const cv::Vec4i *corresps_ptr = corresps.ptr<cv::Vec4i>();
    cv::AutoBuffer<float> diffs(correspsCount);
    float *diffs_ptr = diffs;

    double sigma = 0;
    for (int correspIndex = 0; correspIndex < correspsCount; ++correspIndex)
    {
        const cv::Vec4i &c = corresps_ptr[correspIndex];
        int u0 = c[0], v0 = c[1], u1 = c[2], v1 = c[3];

        diffs_ptr[correspIndex] = static_cast<float>(static_cast<int>(img0.at<uchar>(v0, u0)) -
                                                     static_cast<int>(img1.at<uchar>(v1, u1)));

        sigma += diffs_ptr[correspIndex] * diffs_ptr[correspIndex];
    }

    sigma = std::sqrt(sigma / correspsCount);

    std::vector<double> A_buf(transformDim, 0);
    double *A_ptr = &A_buf[0];
    for (int correspIndex = 0; correspIndex < correspsCount; ++correspIndex)
    {
        const cv::Vec4i &c = corresps_ptr[correspIndex];
        int u0 = c[0], v0 = c[1], u1 = c[2], v1 = c[3];

        double w = sigma + std::abs(diffs_ptr[correspIndex]);
        w = w > DBL_EPSILON ? 1. / w : 1.;

        double w_sobelScale = w * sobelScale;

        const cv::Point3f &p0 = cloud0.at<cv::Point3f>(v0, u0);
        cv::Point3f tp0;
        tp0.x = (float)(p0.x * Rt_ptr[0] + p0.y * Rt_ptr[1] + p0.z * Rt_ptr[2] + Rt_ptr[3]);
        tp0.y = (float)(p0.x * Rt_ptr[4] + p0.y * Rt_ptr[5] + p0.z * Rt_ptr[6] + Rt_ptr[7]);
        tp0.z = (float)(p0.x * Rt_ptr[8] + p0.y * Rt_ptr[9] + p0.z * Rt_ptr[10] + Rt_ptr[11]);

        calcRgbdEquationCoeffs(A_ptr, w_sobelScale * dI_dx1.at<short>(v1, u1),
                w_sobelScale * dI_dy1.at<short>(v1, u1), tp0, fx, fy);

        for (int y = 0;  y < transformDim; ++y)
        {
            double *AtA_ptr = AtA.ptr<double>(y);
            for (int x = y; x < transformDim; ++x)
            {
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];
            }

            AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
        }
    }

    for (int y = 0; y < transformDim; ++y)
    {
        for (int x = y + 1; x < transformDim; ++x)
            AtA.at<double>(x, y) = AtA.at<double>(y, x);
    }

}

static void computeCorresps(const cv::Mat &K,
                            const cv::Mat &K_inv,
                            const cv::Mat &Rt,
                            const cv::Mat &depth0,
                            const cv::Mat &validMask0,
                            const cv::Mat &depth1,
                            const cv::Mat &selectMask1,
                            float maxDepthDiff,
                            cv::Mat &corresps)
{
    cv::Mat  corresps_ (depth1.size(), CV_16SC2, cv::Scalar::all(-1));
    cv::Rect r(0, 0, depth1.cols, depth1.rows);
    cv::Mat  Kt = Rt(cv::Rect(3, 0, 1, 3)).clone(); // t, transformation

    Kt = K * Kt; // K * t
    const double *Kt_ptr = Kt.ptr<const double>();

    cv::AutoBuffer<float> buf(3 * (depth1.cols + depth1.rows));
    float *KRK_inv0_u1 = buf;
    float *KRK_inv1_v1_plus_KRK_inv2 = KRK_inv0_u1 + depth1.cols;
    float *KRK_inv3_u1 = KRK_inv1_v1_plus_KRK_inv2 + depth1.rows;
    float *KRK_inv4_v1_plus_KRK_inv5 = KRK_inv3_u1 + depth1.cols;
    float *KRK_inv6_u1 = KRK_inv4_v1_plus_KRK_inv5 + depth1.rows;
    float *KRK_inv7_v1_plus_KRK_inv8 = KRK_inv6_u1 + depth1.cols;

    cv::Mat R       = Rt(cv::Rect(0, 0, 3, 3)).clone();

    cv::Mat KRK_inv = K * R * K_inv;
    const double *KRK_inv_ptr = KRK_inv.ptr<const double>();
    for (int u1 = 0; u1 < depth1.cols; ++u1)
    {
        KRK_inv0_u1[u1] = KRK_inv_ptr[0] * u1;
        KRK_inv3_u1[u1] = KRK_inv_ptr[3] * u1;
        KRK_inv6_u1[u1] = KRK_inv_ptr[6] * u1;
    }
    for (int v1 = 0; v1 < depth1.rows; ++v1)
    {
        KRK_inv1_v1_plus_KRK_inv2[v1] = KRK_inv_ptr[1] * v1 + KRK_inv_ptr[2];
        KRK_inv4_v1_plus_KRK_inv5[v1] = KRK_inv_ptr[4] * v1 + KRK_inv_ptr[5];
        KRK_inv7_v1_plus_KRK_inv8[v1] = KRK_inv_ptr[7] * v1 + KRK_inv_ptr[8];
    }

    int correspCount = 0;
    for (int v1 = 0; v1 < depth1.rows; ++v1)
    {
        const float *depth1_row = depth1.ptr<const float>(v1);
        const uchar *mask1_row  = selectMask1.ptr<const uchar>(v1);
        for (int u1 = 0; u1 < depth1.cols; ++u1)
        {
            float d1 = depth1_row[u1];
            if (mask1_row[u1])
            {
                float transformed_d1 = static_cast<float>(d1 * (KRK_inv6_u1[u1] + KRK_inv7_v1_plus_KRK_inv8[v1]) + Kt_ptr[2]);
                if (transformed_d1 > 0)
                {
                    float transformed_d1_inv = 1.f / transformed_d1;
                    int u0 = cvRound( transformed_d1_inv * (d1 * (KRK_inv0_u1[u1] + KRK_inv1_v1_plus_KRK_inv2[v1])
                            + Kt_ptr[0]) );
                    int v0 = cvRound( transformed_d1_inv * (d1 * (KRK_inv3_u1[u1] + KRK_inv4_v1_plus_KRK_inv5[v1])
                            + Kt_ptr[1]) );

                    if (r.contains(cv::Point(u0, v0)))
                    {
                        float d0 = depth0.at<float>(v0, u0);
                        if (validMask0.at<uchar>(v0, u0) && std::abs(transformed_d1 - d0) <= maxDepthDiff)
                        {
                            cv::Vec2s &c = corresps_.at<cv::Vec2s>(v0, u0);
                            if (-1 != c[0])
                            {
                                int exist_u1 = c[0], exist_v1 = c[1];
                                float exist_d1 = depth1.at<float>(exist_v1, exist_u1) *
                                        (KRK_inv6_u1[exist_u1] + KRK_inv7_v1_plus_KRK_inv8[exist_v1]) + Kt_ptr[2];
                                if (transformed_d1 > exist_d1)
                                    continue;
                            }
                            else
                                ++correspCount;

                            c = cv::Vec2s(static_cast<short>(u1), static_cast<short>(v1));
                        }
                    } // if r
                } // if transformed_d1
            } // if mask1
        } // for u1
    } // for v1

    corresps.create(correspCount, 1, CV_32SC4);
    cv::Vec4i *corresps_ptr = corresps.ptr<cv::Vec4i>();
    for (int v0 = 0, i = 0; v0 < corresps_.rows; ++v0)
    {
        const cv::Vec2s *corresps_row = corresps_.ptr<cv::Vec2s>(v0);
        for (int u0 = 0; u0 < corresps_.cols; ++u0)
        {
            const cv::Vec2s &c = corresps_row[u0];
            if (-1 != c[0])
                corresps_ptr[i++] = cv::Vec4i(u0, v0, c[0], c[1]);
        }
    }
}

static bool solveSystem(const cv::Mat &AtA, const cv::Mat &AtB, double detThreshold, cv::Mat &x)
{
    double det = cv::determinant(AtA);
    if  (fabs(det) < detThreshold || cvIsNaN(det) || cvIsInf(det))
        return false;

    cv::solve(AtA, AtB, x, cv::DECOMP_CHOLESKY);

    return true;
}

static void computeProjectiveMatrix(const cv::Mat &ksi, cv::Mat &Rt)
{
    const double *ksi_ptr = ksi.ptr<const double>();
    Eigen::Matrix<double, 4, 4> twist, g;

    twist << 0.,          -ksi_ptr[2], ksi_ptr[1],  ksi_ptr[3],
            ksi_ptr[2],  0.,          -ksi_ptr[0], ksi_ptr[4],
            -ksi_ptr[1], ksi_ptr[0],  0,           ksi_ptr[5],
            0.,          0.,          0.,          0.;
    g = twist.exp();

    cv::eigen2cv(g, Rt);

}

static bool testDeltaTransformation(const cv::Mat &deltaRt, float maxTranslation, int maxRotation)
{
    float translation = static_cast<float>(cv::norm(deltaRt(cv::Rect(3, 0, 1, 3))));

    cv::Mat rvec;
    cv::Rodrigues(deltaRt(cv::Rect(0, 0, 3, 3)), rvec);

    double rotation = cv::norm(rvec) * 180. / CV_PI;

    return translation <= maxTranslation && rotation <= (double)maxRotation;
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

    const int transformDim       = 6;
    const int minOverdetermScale = 20;
    const int minCorrespsCount   = transformDim * minOverdetermScale;
    std::vector<cv::Mat> pyramidCameraMatrix;
    cv::Mat resultRt = (initRt.empty()) ? cv::Mat::eye(4, 4, CV_64FC1) : initRt.clone();
    cv::Mat currRt, ksi;

    buildPyramidCameraMatrix(cameraMatrix_, static_cast<int>( srcFrame.pyramidDepth_.size() ), pyramidCameraMatrix);

    bool isOk = false;
    for (int level = static_cast<int>( iterCounts_.size() ) - 1; level >= 0; --level)
    {
        const cv::Mat &levelCameraMatrix     = pyramidCameraMatrix[level];
        const cv::Mat &levelCameraMatrix_inv = levelCameraMatrix.inv(cv::DECOMP_SVD);
        const cv::Mat &srcLevelDepth         = srcFrame.pyramidDepth_[level];
        const cv::Mat &dstLevelDepth         = dstFrame.pyramidDepth_[level];

        const double fx = levelCameraMatrix.at<double>(0,0);
        const double fy = levelCameraMatrix.at<double>(1,1);
        const double determintThreshold = 1e-6;

        cv::Mat AtA_rgbd, AtB_rgbd;
        cv::Mat corresps_rgbd;

        for (int iter = 0; iter < iterCounts_[level]; ++iter)
        {
            cv::Mat resultRt_inv = resultRt.inv(cv::DECOMP_SVD);
            computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt, srcLevelDepth, srcFrame.pyramidMask_[level],
                    dstLevelDepth, dstFrame.pyramidTextureMask_[level], maxDepthDiff_, corresps_rgbd);

            if (corresps_rgbd.rows < minCorrespsCount)
                break;

            if (corresps_rgbd.rows >= minCorrespsCount)
            {
                calcRbgdLsmMatrices(srcFrame.pyramidImg_[level], srcFrame.pyramidCloud_[level], resultRt,
                        dstFrame.pyramidImg_[level], dstFrame.pyramid_dIdx_[level], dstFrame.pyramid_dIdy_[level],
                        corresps_rgbd, fx, fy, AtA_rgbd, AtB_rgbd);
            }

            bool solutionExist = solveSystem(AtA_rgbd, AtB_rgbd, determintThreshold, ksi);
            if (!solutionExist)
                break;

            computeProjectiveMatrix(ksi, currRt);
            resultRt = currRt * resultRt;
            isOk = true;
        }
    }

    Rt = resultRt;
    if (isOk)
    {
        cv::Mat deltaRt;
        if (initRt.empty())
            deltaRt = resultRt;
        else
            deltaRt = resultRt * initRt.inv(cv::DECOMP_SVD);

        isOk = testDeltaTransformation(deltaRt, maxTranslation_, maxRotation_);
    }
    return isOk;
}