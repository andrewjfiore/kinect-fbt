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

    // Per-joint motion tracks: `still` holds the speed verdict from the most
    // recent timestamp advance. Lost joints invalidate their track; a fresh
    // pair of distinct frames is needed before the joint counts as still.
    const auto updateTrack = [this](MotionTrack& tr, const JointSample& s, double t) {
        if (s.state != TrackState::Tracked) {
            tr.t = -1.0;
            tr.still = false;
            return;
        }
        if (tr.t >= 0.0 && t > tr.t) {
            const double dt = t - tr.t;
            const float speed = (s.pos - tr.pos).norm() / static_cast<float>(dt);
            tr.still = speed <= opt_.maxJointSpeed;
        }
        if (t != tr.t) {
            tr.pos = s.pos;
            tr.t = t;
        }
    };

    const bool gate = opt_.maxJointSpeed > 0.0f;
    if (gate) {
        for (const Joint j : opt_.joints) {
            const auto idx = static_cast<size_t>(j);
            updateTrack(refTrack_[idx], reference[j], reference.timestamp);
            updateTrack(tgtTrack_[idx], target[j], target.timestamp);
        }
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
        if (gate) {
            const auto idx = static_cast<size_t>(j);
            if (!refTrack_[idx].still || !tgtTrack_[idx].still) {
                continue;
            }
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
    RigidFit fit = solveRigid(tgtPts_, refPts_);
    if (!fit.ok || !opt_.trimOutliers) {
        return fit;
    }

    // Trimmed re-solve: drop pairs with residual > trimFactor * median and
    // refit, twice. Keeps the fit honest against sporadic garbage joints.
    std::vector<Vec3> src = tgtPts_;
    std::vector<Vec3> dst = refPts_;
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> resid(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            resid[i] = (fit.transform.apply(src[i]) - dst[i]).norm();
        }
        std::vector<float> sorted = resid;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        const float median = sorted[sorted.size() / 2];
        const float cutoff = std::max(opt_.trimFactor * median, 0.01f);

        std::vector<Vec3> keptSrc, keptDst;
        keptSrc.reserve(src.size());
        keptDst.reserve(dst.size());
        for (size_t i = 0; i < src.size(); ++i) {
            if (resid[i] <= cutoff) {
                keptSrc.push_back(src[i]);
                keptDst.push_back(dst[i]);
            }
        }
        const size_t keepFloor = std::max<size_t>(opt_.minSamples / 2, 3);
        if (keptSrc.size() == src.size() || keptSrc.size() < keepFloor) {
            break;
        }
        const RigidFit refit = solveRigid(keptSrc, keptDst);
        if (!refit.ok) {
            break;
        }
        fit = refit;
        src.swap(keptSrc);
        dst.swap(keptDst);
    }
    return fit;
}

RigidFit solveAnchor(const std::vector<Vec3>& worldPts, const std::vector<Vec3>& externalPts) {
    return solveRigid(worldPts, externalPts);
}

} // namespace mn
