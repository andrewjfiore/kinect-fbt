package com.fbt.quest

import android.util.Log
import kotlinx.coroutines.*
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "VRChatOscForwarder"
private const val VRCHAT_HOST = "127.0.0.1"
private const val FORWARD_HZ = 20
private const val CONFIDENCE_THRESHOLD = 0.2f

class VRChatOscForwarder(
    private val vrChatPort: Int = 9000,
    private val getTracker: (id: Int) -> TrackerState?,
    private val availableTrackerIds: () -> Set<Int>,
) {
    private var job: Job? = null
    private var forwardFps: Float = 0f
    val fps get() = forwardFps

    fun start(scope: CoroutineScope) {
        job = scope.launch(Dispatchers.IO) {
            val socket = DatagramSocket()
            val host = InetAddress.getByName(VRCHAT_HOST)
            val interval = 1000L / FORWARD_HZ
            var frameCount = 0
            var lastFpsTime = System.currentTimeMillis()

            while (isActive) {
                val loopStart = System.currentTimeMillis()

                for (id in availableTrackerIds()) {
                    val t = getTracker(id) ?: continue
                    if (t.confidence < CONFIDENCE_THRESHOLD) continue
                    if (System.currentTimeMillis() - t.lastUpdated > 2000L) continue

                    val bundle = buildBundle(id, t)
                    val packet = DatagramPacket(bundle, bundle.size, host, vrChatPort)
                    try {
                        socket.send(packet)
                    } catch (e: Exception) {
                        Log.e(TAG, "Send error: ${e.message}")
                    }
                }

                frameCount++
                val now = System.currentTimeMillis()
                if (now - lastFpsTime >= 1000L) {
                    forwardFps = frameCount.toFloat() * 1000f / (now - lastFpsTime)
                    frameCount = 0
                    lastFpsTime = now
                }

                val elapsed = System.currentTimeMillis() - loopStart
                val sleep = interval - elapsed
                if (sleep > 0) delay(sleep)
            }
            socket.close()
        }
    }

    fun stop() {
        job?.cancel()
    }

    private fun buildBundle(id: Int, t: TrackerState): ByteArray {
        // Build a minimal OSC bundle with position + rotation messages
        val posMsg = buildMessage("/tracking/trackers/$id/position", t.position)
        val rotMsg = buildMessage("/tracking/trackers/$id/rotation", t.rotation)

        val timeTag = ByteArray(8) { if (it == 7) 1 else 0 } // immediate
        val bundleHeader = oscString("#bundle") + timeTag
        val posPart = int32ToBytes(posMsg.size) + posMsg
        val rotPart = int32ToBytes(rotMsg.size) + rotMsg

        return bundleHeader + posPart + rotPart
    }

    private fun buildMessage(address: String, floats: FloatArray): ByteArray {
        val addrBytes = oscString(address)
        val typeTag = oscString(",fff")
        val args = ByteArray(floats.size * 4)
        val buf = ByteBuffer.wrap(args).order(ByteOrder.BIG_ENDIAN)
        floats.forEach { buf.putFloat(it) }
        return addrBytes + typeTag + args
    }

    private fun oscString(s: String): ByteArray {
        val bytes = s.toByteArray(Charsets.US_ASCII)
        val padded = ((bytes.size + 1) + 3) / 4 * 4
        return bytes + ByteArray(padded - bytes.size)
    }

    private fun int32ToBytes(v: Int): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN).putInt(v).array()
}
