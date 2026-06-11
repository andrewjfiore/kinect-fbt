#include <doctest/doctest.h>

#include "mn/events.hpp"
#include "mn/log.hpp"

#include <cstdint>
#include <string>
#include <vector>

// The EventLog is a process-wide singleton with a monotonic seq, so every
// case snapshots latestSeq()/clear()s first and asserts relative to that.

TEST_CASE("EventLog: push, since, latestSeq, levelCounts") {
    mn::EventLog& elog = mn::EventLog::instance();
    elog.clear();
    const uint64_t base = elog.latestSeq();

    elog.push(mn::log::Level::Info, "alpha");
    elog.push(mn::log::Level::Warn, "beta");
    elog.push(mn::log::Level::Error, "gamma");

    CHECK(elog.latestSeq() == base + 3);

    const auto events = elog.since(base);
    REQUIRE(events.size() == 3u);
    CHECK(events[0].seq == base + 1);
    CHECK(events[0].level == mn::log::Level::Info);
    CHECK(events[0].message == "alpha");
    CHECK(events[1].seq == base + 2);
    CHECK(events[1].level == mn::log::Level::Warn);
    CHECK(events[1].message == "beta");
    CHECK(events[2].seq == base + 3);
    CHECK(events[2].level == mn::log::Level::Error);
    CHECK(events[2].message == "gamma");
    CHECK(events[0].t > 0.0);
    CHECK(events[0].t <= events[1].t);
    CHECK(events[1].t <= events[2].t);

    // maxCount caps from the oldest side.
    const auto capped = elog.since(base, 2);
    REQUIRE(capped.size() == 2u);
    CHECK(capped[0].seq == base + 1);
    CHECK(capped[1].seq == base + 2);

    // afterSeq filters strictly.
    const auto tail = elog.since(base + 2);
    REQUIRE(tail.size() == 1u);
    CHECK(tail[0].seq == base + 3);
    CHECK(elog.since(elog.latestSeq()).empty());

    const auto counts = elog.levelCounts();
    CHECK(counts[static_cast<size_t>(mn::log::Level::Debug)] == 0u);
    CHECK(counts[static_cast<size_t>(mn::log::Level::Info)] == 1u);
    CHECK(counts[static_cast<size_t>(mn::log::Level::Warn)] == 1u);
    CHECK(counts[static_cast<size_t>(mn::log::Level::Error)] == 1u);
}

TEST_CASE("EventLog: ring drops oldest at kCapacity, seq stays monotonic") {
    mn::EventLog& elog = mn::EventLog::instance();
    elog.clear();
    const uint64_t base = elog.latestSeq();

    const size_t total = mn::EventLog::kCapacity + 10;
    for (size_t i = 0; i < total; ++i)
        elog.push(mn::log::Level::Debug, "event-" + std::to_string(i));

    CHECK(elog.latestSeq() == base + total);

    const auto events = elog.since(base, total + 100);
    REQUIRE(events.size() == mn::EventLog::kCapacity); // the oldest 10 fell out
    CHECK(events.front().seq == base + 11);
    CHECK(events.front().message == "event-10");
    CHECK(events.back().seq == base + total);
    for (size_t i = 1; i < events.size(); ++i)
        CHECK(events[i].seq == events[i - 1].seq + 1);

    // Lifetime counts are not capped by the ring.
    CHECK(elog.levelCounts()[static_cast<size_t>(mn::log::Level::Debug)] == total);
}

TEST_CASE("EventLog: installLogCapture routes log writes into the event log") {
    mn::EventLog::installLogCapture();
    mn::EventLog& elog = mn::EventLog::instance();
    elog.clear();
    const uint64_t base = elog.latestSeq();

    mn::log::warn("event capture test ", 42);

    const auto events = elog.since(base);
    REQUIRE(events.size() == 1u);
    CHECK(events[0].level == mn::log::Level::Warn);
    CHECK(events[0].message == "event capture test 42");
    CHECK(elog.levelCounts()[static_cast<size_t>(mn::log::Level::Warn)] == 1u);

    // Idempotent: installing again must not duplicate deliveries.
    mn::EventLog::installLogCapture();
    const uint64_t mid = elog.latestSeq();
    mn::log::error("captured exactly once");
    CHECK(elog.since(mid).size() == 1u);

    // The sink sees messages below the console level filter too.
    const mn::log::Level oldLevel = mn::log::level();
    mn::log::setLevel(mn::log::Level::Info);
    const uint64_t beforeDebug = elog.latestSeq();
    mn::log::debug("quiet on console, captured here");
    const auto dbg = elog.since(beforeDebug);
    REQUIRE(dbg.size() == 1u);
    CHECK(dbg[0].level == mn::log::Level::Debug);
    mn::log::setLevel(oldLevel);

    mn::log::setSink({}); // detach so other test cases do not feed this log
}

TEST_CASE("EventLog: clear drops events and counts but seq keeps rising") {
    mn::EventLog& elog = mn::EventLog::instance();
    elog.clear();
    elog.push(mn::log::Level::Info, "before-clear");
    const uint64_t seqBefore = elog.latestSeq();

    elog.clear();
    CHECK(elog.latestSeq() == seqBefore); // counter survives the clear
    CHECK(elog.since(0).empty());
    for (uint64_t c : elog.levelCounts())
        CHECK(c == 0u);

    elog.push(mn::log::Level::Info, "after-clear");
    const auto events = elog.since(seqBefore);
    REQUIRE(events.size() == 1u);
    CHECK(events[0].seq == seqBefore + 1); // strictly above every pre-clear seq
    CHECK(events[0].message == "after-clear");
}
