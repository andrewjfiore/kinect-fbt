#include "mn/filters.hpp"

#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace {

double varianceAboutMean(const std::vector<float>& xs, size_t skip) {
    double mean = 0.0;
    size_t n = 0;
    for (size_t i = skip; i < xs.size(); ++i) {
        mean += static_cast<double>(xs[i]);
        ++n;
    }
    mean /= static_cast<double>(n);
    double var = 0.0;
    for (size_t i = skip; i < xs.size(); ++i) {
        const double e = static_cast<double>(xs[i]) - mean;
        var += e * e;
    }
    return var / static_cast<double>(n);
}

} // namespace

TEST_CASE("one-euro attenuates jitter on a noisy constant signal") {
    std::mt19937 rng(42u);
    std::normal_distribution<float> noise(0.0f, 0.05f);

    mn::OneEuroFilter filter(mn::OneEuroParams{1.0f, 0.05f, 1.0f});
    const double dt = 1.0 / 120.0;

    std::vector<float> raw;
    std::vector<float> filtered;
    double t = 0.0;
    for (int i = 0; i < 600; ++i) {
        const float sample = 1.0f + noise(rng);
        raw.push_back(sample);
        filtered.push_back(filter.filter(sample, t));
        t += dt;
    }

    // Skip the warmup so the EMA has settled around the true value.
    const double rawVar = varianceAboutMean(raw, 100);
    const double filtVar = varianceAboutMean(filtered, 100);
    CHECK(filtVar < rawVar);
    CHECK(filtVar < 0.5 * rawVar); // should be far better than a 2x reduction
}

TEST_CASE("one-euro converges to a step input") {
    mn::OneEuroFilter filter; // default params: minCutoff 1 Hz
    const double dt = 1.0 / 120.0;
    double t = 0.0;

    float out = filter.filter(0.0f, t);
    CHECK(out == doctest::Approx(0.0f));
    for (int i = 0; i < 60; ++i) {
        t += dt;
        out = filter.filter(0.0f, t);
    }
    CHECK(out == doctest::Approx(0.0f));

    // Step to 1.0 and run 2 seconds: tau at 1 Hz is ~0.16 s, so the residual
    // must be far inside 1% by then.
    for (int i = 0; i < 240; ++i) {
        t += dt;
        out = filter.filter(1.0f, t);
    }
    CHECK(std::abs(out - 1.0f) < 0.01f);
}

TEST_CASE("one-euro holds the last value when dt <= 0") {
    mn::OneEuroFilter filter;

    // First sample passes through unchanged.
    CHECK(filter.filter(2.0f, 1.0) == doctest::Approx(2.0f));

    const double t1 = 1.0 + 1.0 / 120.0;
    const float second = filter.filter(3.0f, t1);
    CHECK(second > 2.0f);
    CHECK(second < 3.0f);

    // Duplicate timestamp (dt == 0) and a timestamp going backwards (dt < 0)
    // both return the last filtered value and do not corrupt state.
    CHECK(filter.filter(100.0f, t1) == doctest::Approx(second));
    CHECK(filter.filter(100.0f, 0.5) == doctest::Approx(second));

    // A subsequent valid timestamp keeps filtering from where it left off.
    const float next = filter.filter(3.0f, 1.0 + 2.0 / 120.0);
    CHECK(next > second);
    CHECK(next < 3.0f);
}

TEST_CASE("one-euro reset forgets history") {
    mn::OneEuroFilter filter;
    (void)filter.filter(5.0f, 0.0);
    (void)filter.filter(6.0f, 0.01);
    filter.reset();
    // After reset the next sample is treated as the first and passes through.
    CHECK(filter.filter(-3.0f, 0.02) == doctest::Approx(-3.0f));
}

TEST_CASE("one-euro vec3 wrapper filters per component") {
    mn::OneEuroVec3 filter;
    const mn::Vec3 first(1.0f, 2.0f, 3.0f);
    CHECK((filter.filter(first, 0.0) - first).norm() == doctest::Approx(0.0f));

    const mn::Vec3 out = filter.filter(mn::Vec3(2.0f, 3.0f, 4.0f), 1.0 / 120.0);
    CHECK(out.x() > 1.0f);
    CHECK(out.x() < 2.0f);
    CHECK(out.y() > 2.0f);
    CHECK(out.y() < 3.0f);
    CHECK(out.z() > 3.0f);
    CHECK(out.z() < 4.0f);
}
