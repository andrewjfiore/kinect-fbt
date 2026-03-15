/**
 * fn2_shim.cpp — Thin C wrapper around libfreenect2 C++ API for Python ctypes.
 * Build: g++ -shared -fPIC -o libfn2_shim.so fn2_shim.cpp -lfreenect2 -I/usr/local/include
 */
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <cstring>
#include <cstdio>

extern "C" {

// Opaque handles
typedef void* fn2_context;
typedef void* fn2_device;
typedef void* fn2_listener;
typedef void* fn2_registration;

// Frame data returned to Python
typedef struct {
    unsigned char* color_data;   // BGRX 1920x1080 = 1920*1080*4 bytes
    float*         depth_data;   // 512x424 float mm
    int            color_width;
    int            color_height;
    float*         bigdepth_data; // 1920x1082 float mm (depth mapped to color space)
    int            bigdepth_width;
    int            bigdepth_height;
} fn2_frame_data;

// ── Context ──
fn2_context fn2_create() {
    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
    return new libfreenect2::Freenect2();
}

void fn2_destroy(fn2_context ctx) {
    delete static_cast<libfreenect2::Freenect2*>(ctx);
}

int fn2_enumerate(fn2_context ctx) {
    return static_cast<libfreenect2::Freenect2*>(ctx)->enumerateDevices();
}

int fn2_get_serial(fn2_context ctx, int idx, char* buf, int buf_len) {
    std::string serial = static_cast<libfreenect2::Freenect2*>(ctx)->getDeviceSerialNumber(idx);
    if (serial.empty()) return -1;
    int len = serial.size() < (size_t)(buf_len - 1) ? serial.size() : buf_len - 1;
    memcpy(buf, serial.c_str(), len);
    buf[len] = '\0';
    return len;
}

// ── Device ──
fn2_device fn2_open_device(fn2_context ctx, int idx, int pipeline_type) {
    auto* f2 = static_cast<libfreenect2::Freenect2*>(ctx);
    libfreenect2::PacketPipeline* pipeline = nullptr;

    switch (pipeline_type) {
        case 0: // CPU
            pipeline = new libfreenect2::CpuPacketPipeline();
            break;
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
        case 1: // OpenGL
            pipeline = new libfreenect2::OpenGLPacketPipeline();
            break;
#endif
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
        case 2: // OpenCL
            pipeline = new libfreenect2::OpenCLPacketPipeline();
            break;
#endif
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
        case 3: // CUDA
            pipeline = new libfreenect2::CudaPacketPipeline();
            break;
#endif
        default:
            pipeline = new libfreenect2::CpuPacketPipeline();
            break;
    }

    return f2->openDevice(idx, pipeline);
}

fn2_device fn2_open_device_by_serial(fn2_context ctx, const char* serial, int pipeline_type) {
    auto* f2 = static_cast<libfreenect2::Freenect2*>(ctx);
    libfreenect2::PacketPipeline* pipeline = nullptr;

    switch (pipeline_type) {
        case 0:
            pipeline = new libfreenect2::CpuPacketPipeline();
            break;
        default:
            pipeline = new libfreenect2::CpuPacketPipeline();
            break;
    }

    return f2->openDevice(std::string(serial), pipeline);
}

// ── Listener ──
fn2_listener fn2_create_listener(unsigned int frame_types) {
    return new libfreenect2::SyncMultiFrameListener(frame_types);
}

void fn2_set_listeners(fn2_device dev, fn2_listener listener) {
    auto* d = static_cast<libfreenect2::Freenect2Device*>(dev);
    auto* l = static_cast<libfreenect2::SyncMultiFrameListener*>(listener);
    d->setColorFrameListener(l);
    d->setIrAndDepthFrameListener(l);
}

// ── Registration ──
fn2_registration fn2_create_registration(fn2_device dev) {
    auto* d = static_cast<libfreenect2::Freenect2Device*>(dev);
    return new libfreenect2::Registration(d->getIrCameraParams(), d->getColorCameraParams());
}

void fn2_destroy_registration(fn2_registration reg) {
    delete static_cast<libfreenect2::Registration*>(reg);
}

// ── Start / Stop ──
int fn2_start(fn2_device dev) {
    return static_cast<libfreenect2::Freenect2Device*>(dev)->start() ? 0 : -1;
}

int fn2_stop(fn2_device dev) {
    return static_cast<libfreenect2::Freenect2Device*>(dev)->stop() ? 0 : -1;
}

int fn2_close(fn2_device dev) {
    return static_cast<libfreenect2::Freenect2Device*>(dev)->close() ? 0 : -1;
}

// ── Frame acquisition ──
// Returns 0 on success, -1 on timeout/error
// Caller must call fn2_release_frame after processing
static libfreenect2::FrameMap* g_framemap = nullptr;
static libfreenect2::Frame* g_undistorted = nullptr;
static libfreenect2::Frame* g_registered = nullptr;
static libfreenect2::Frame* g_bigdepth = nullptr;

// Per-device frame state (supports up to 4 devices)
#define MAX_DEVICES 4
static struct {
    libfreenect2::FrameMap framemap;
    libfreenect2::Frame* undistorted;
    libfreenect2::Frame* registered;
    libfreenect2::Frame* bigdepth;
    bool in_use;
} g_frame_state[MAX_DEVICES] = {};

int fn2_wait_for_frame(fn2_listener listener, int timeout_ms, int slot) {
    if (slot < 0 || slot >= MAX_DEVICES) return -1;
    auto* l = static_cast<libfreenect2::SyncMultiFrameListener*>(listener);
    bool ok = l->waitForNewFrame(g_frame_state[slot].framemap, timeout_ms);
    if (ok) g_frame_state[slot].in_use = true;
    return ok ? 0 : -1;
}

int fn2_get_frame_data(fn2_listener listener, fn2_registration reg,
                       fn2_frame_data* out, int slot) {
    if (slot < 0 || slot >= MAX_DEVICES || !g_frame_state[slot].in_use) return -1;

    auto& fm = g_frame_state[slot].framemap;
    auto* r = static_cast<libfreenect2::Registration*>(reg);

    auto* color = fm[libfreenect2::Frame::Color];
    auto* depth = fm[libfreenect2::Frame::Depth];

    if (!color || !depth) return -1;

    // Allocate frames for registration if needed
    if (!g_frame_state[slot].undistorted)
        g_frame_state[slot].undistorted = new libfreenect2::Frame(512, 424, 4);
    if (!g_frame_state[slot].registered)
        g_frame_state[slot].registered = new libfreenect2::Frame(512, 424, 4);
    if (!g_frame_state[slot].bigdepth)
        g_frame_state[slot].bigdepth = new libfreenect2::Frame(1920, 1082, 4);

    // Apply registration — maps depth to color space
    r->apply(color, depth, g_frame_state[slot].undistorted,
             g_frame_state[slot].registered, true,
             g_frame_state[slot].bigdepth);

    out->color_data = color->data;
    out->color_width = color->width;
    out->color_height = color->height;
    out->depth_data = reinterpret_cast<float*>(depth->data);

    // bigdepth is 1920x1082 float — depth mapped to color coordinates
    out->bigdepth_data = reinterpret_cast<float*>(g_frame_state[slot].bigdepth->data);
    out->bigdepth_width = 1920;
    out->bigdepth_height = 1082;

    return 0;
}

void fn2_release_frame(fn2_listener listener, int slot) {
    if (slot < 0 || slot >= MAX_DEVICES || !g_frame_state[slot].in_use) return;
    auto* l = static_cast<libfreenect2::SyncMultiFrameListener*>(listener);
    l->release(g_frame_state[slot].framemap);
    g_frame_state[slot].in_use = false;
}

void fn2_cleanup_slot(int slot) {
    if (slot < 0 || slot >= MAX_DEVICES) return;
    delete g_frame_state[slot].undistorted;
    delete g_frame_state[slot].registered;
    delete g_frame_state[slot].bigdepth;
    g_frame_state[slot].undistorted = nullptr;
    g_frame_state[slot].registered = nullptr;
    g_frame_state[slot].bigdepth = nullptr;
    g_frame_state[slot].in_use = false;
}

// Frame type constants matching libfreenect2::Frame::Type
// Color = 1, Ir = 2, Depth = 4
int fn2_frame_type_color() { return 1; }
int fn2_frame_type_ir()    { return 2; }
int fn2_frame_type_depth() { return 4; }

} // extern "C"
