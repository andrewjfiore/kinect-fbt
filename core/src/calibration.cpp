// Rigid alignment solvers (Kabsch/Umeyama, no scale) and the sensor-pair
// calibration session. See core/include/mn/calibration.hpp for contracts.

#include "mn/calibration.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <utility>

namespace mn {

RigidFit solveRigid(const std::vector<Vec3>& src, const std::vector<Vec3>& dst) {
    RigidFit fit;
    const size_t n = std::min(src.size(), dst.size());
    fit.samples = n;
    if (src.size() != dst.size() || n < 3) {
        return fit;
    }

    Vec3 srcMean = Vec3::Zero();
    Vec3 dstMean = Vec3::Zero();
    for (size_t i = 0; i < n; ++i) {
        srcMean += src[i];
        dstMean += dst[i];
    }
    const float invN = 1.0f / static_cast<float>(n);
    srcMean *= invN;
    dstMean *= invN;

    // srcCov gates degeneracy; crossCov (dst * src^T) is the Kabsch matrix.
    Mat3 srcCov = Mat3::Zero();
    Mat3 crossCov = Mat3::Zero();
    for (size_t i = 0; i < n; ++i) {
        const Vec3 s = src[i] - srcMean;
        const Vec3 d = dst[i] - dstMean;
        srcCov += s * s.transpose();
        crossCov += d * s.transpose();
    }
    srcCov *= invN;
    crossCov *= invN;

    {
        // Collinear (or coincident) source points leave the rotation about the
        // dominant axis unobservable: 2nd singular value of the centered
        // source covariance collapses to zero.
        Eigen::JacobiSVD<Mat3> svdSrc(srcCov);
        if (svdSrc.singularValues()(1) < 1e-6f) {
            return fit;
        }
    }

    // crossCov = U S V^T  =>  R = U D V^T, with D flipping the smallest
    // singular direction when the best orthogonal map is a reflection.
    Eigen::JacobiSVD<Mat3> svd(crossCov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Mat3 u = svd.matrixU();
    const Mat3 v = svd.matrixV();
    Mat3 d = Mat3::Identity();
    if ((u * v.transpose()).determinant() < 0.0f) {
        d(2, 2) = -1.0f;
    }
    const Mat3 r = u * d * v.transpose();
    const Vec3 t = dstMean - r * srcMean;

    double sse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const Vec3 e = r * src[i] + t - dst[i];
        sse += static_cast<double>(e.squaredNorm());
    }

    fit.ok = true;
    fit.transform.pos = t;
    fit.transform.rot = Quat(r).normalized();
    fit.rmse = static_cast<float>(std::sqrt(sse / static_cast<double>(n)));
    return fit;
}

PairCalibrationSession::PairCalibrationSession(Options opt) : opt_(std::move(opt)) {}

void PairCalibrationSession::addFramePair(const SkeletonFrame& reference,
                                          const SkeletonFrame& target) {
    if (!reference.hasBody || !target.hasBody) {
        return;
    }
    if (std::abs(reference.timestamp - target.timestamp) >
        static_cast<double>(opt_.maxTimeDeltaSec)) {
        return;
    }
    for (const Joint j : opt_.joints) {
        const JointSample& a = reference[j];
        const JointSample& b = target[j];
        if (a.state != TrackState::Tracked || b.state != TrackState::Tracked) {
            continue;
        }
        if (a.confidence < opt_.minConfidence || b.confidence < opt_.minConfidence) {
            continue;
        }
        refPts_.push_back(a.pos);
        tgtPts_.push_back(b.pos);
    }
}

size_t PairCalibrationSession::sampleCount() const {
    return refPts_.size();
}

RigidFit PairCalibrationSession::solve() const {
    if (refPts_.size() < opt_.minSamples) {
        RigidFit fit;
        fit.samples = refPts_.size();
        return fit;
    }
    // src = target-local points, dst = reference-local points, so the fit maps
    // target-local -> reference-local.
    return solveRigid(tgtPts_, refPts_);
}

RigidFit solveAnchor(const std::vector<Vec3>& worldPts, const std::vector<Vec3>& externalPts) {
    return solveRigid(worldPts, externalPts);
}

} // namespace mn
