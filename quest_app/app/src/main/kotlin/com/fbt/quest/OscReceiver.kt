package com.fbt.quest

import android.util.Log
import kotlinx.coroutines.*
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicReference

private const val TAG = "OscReceiver"
private const val BUFFER_SIZE = 4096

data class TrackerState(
    val id: Int,
    val position: FloatArray = FloatArray(3),
    val rotation: FloatArray = FloatArray(3),
    val confidence: Float = 0f,
    val lastUpdated: Long = 0L
)

data class ServerStatus(
    val camerasActive: Int = 0,
    val jointsTracked: Int = 0,
    val fps: Float = 0f,
    val lastUpdated: Long = 0L
)

class OscReceiver(
    private val port: Int = 39571,
    private val onTracker: (TrackerState) -> Unit,
    private val onStatus: (ServerStatus) -> Unit,
    private val onCalibrate: () -> Unit,
) {
    private var job: Job? = null
    private var socket: DatagramSocket? = null

    fun start(scope: CoroutineScope) {
        job = scope.launch(Dispatchers.IO) {
            try {
                socket = DatagramSocket(port)
                Log.i(TAG, "Listening on UDP port $port")
                val buf = ByteArray(BUFFER_SIZE)
                val packet = DatagramPacket(buf, buf.size)
                while (isActive) {
                    try {
                        socket!!.receive(packet)
                        parseOsc(packet.data, packet.length)
                    } catch (e: Exception) {
                        if (isActive) Log.e(TAG, "Receive error: ${e.message}")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Socket error: ${e.message}")
            } finally {
                socket?.close()
            }
        }
    }

    fun stop() {
        socket?.close()
        job?.cancel()
    }

    private fun parseOsc(data: ByteArray, length: Int) {
        if (length < 4) return
        // Check if OSC bundle
        if (data.take(8) == "#bundle\u0000".toByteArray().toList()) {
            parseBundle(data, length)
        } else {
            parseMessage(data, 0, length)
        }
    }

    private fun parseBundle(data: ByteArray, length: Int) {
        var offset = 16 // "#bundle\0" (8) + timetag (8)
        while (offset + 4 <= length) {
            val msgLen = ByteBuffer.wrap(data, offset, 4).order(ByteOrder.BIG_ENDIAN).int
            offset += 4
            if (msgLen > 0 && offset + msgLen <= length) {
                parseMessage(data, offset, msgLen)
                offset += msgLen
            } else break
        }
    }

    private fun parseMessage(data: ByteArray, start: Int, length: Int) {
        try {
            val (address, afterAddr) = readString(data, start, length) ?: return
            val (typeTag, afterTag) = readString(data, start + afterAddr, length - afterAddr) ?: return
            val args = parseArgs(data, start + afterAddr + afterTag, length - afterAddr - afterTag, typeTag)

            when {
                address.startsWith("/fbt/tracker/") -> {
                    val parts = address.split("/")
                    if (parts.size < 5) return
                    val id = parts[3].toIntOrNull() ?: return
                    val field = parts[4]
                    handleTrackerMessage(id, field, args)
                }
                address == "/fbt/status" && args.size >= 3 -> {
                    val status = ServerStatus(
                        camerasActive = (args[0] as? Int) ?: 0,
                        jointsTracked = (args[1] as? Int) ?: 0,
                        fps = (args[2] as? Float) ?: 0f,
                        lastUpdated = System.currentTimeMillis()
                    )
                    onStatus(status)
                }
                address == "/calibrate" -> onCalibrate()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Parse error: ${e.message}")
        }
    }

    private val trackerBuffer = mutableMapOf<Int, TrackerState>()
    private val trackerLock = Any()

    private fun handleTrackerMessage(id: Int, field: String, args: List<Any>) {
        synchronized(trackerLock) {
            val current = trackerBuffer[id] ?: TrackerState(id)
            val updated = when (field) {
                "position" -> current.copy(
                    position = floatArrayOf(
                        (args.getOrNull(0) as? Float) ?: 0f,
                        (args.getOrNull(1) as? Float) ?: 0f,
                        (args.getOrNull(2) as? Float) ?: 0f
                    ),
                    lastUpdated = System.currentTimeMillis()
                )
                "rotation" -> current.copy(
                    rotation = floatArrayOf(
                        (args.getOrNull(0) as? Float) ?: 0f,
                        (args.getOrNull(1) as? Float) ?: 0f,
                        (args.getOrNull(2) as? Float) ?: 0f
                    )
                )
                "confidence" -> {
                    val conf = (args.getOrNull(0) as? Float) ?: 0f
                    val s = current.copy(confidence = conf)
                    trackerBuffer[id] = s
                    onTracker(s)
                    return
                }
                else -> return
            }
            trackerBuffer[id] = updated
        }
    }

    // OSC string parsing: null-terminated, padded to 4-byte boundary
    private fun readString(data: ByteArray, offset: Int, maxLen: Int): Pair<String, Int>? {
        if (offset >= data.size) return null
        val end = data.indexOf(0, offset).takeIf { it >= 0 } ?: return null
        val s = String(data, offset, end - offset, Charsets.US_ASCII)
        val padded = ((end - offset + 1) + 3) / 4 * 4
        return Pair(s, padded)
    }

    private fun parseArgs(data: ByteArray, offset: Int, maxLen: Int, typeTag: String): List<Any> {
        val args = mutableListOf<Any>()
        var pos = offset
        // Skip leading comma in type tag
        val types = if (typeTag.startsWith(",")) typeTag.substring(1) else typeTag
        for (t in types) {
            if (pos + 4 > data.size) break
            when (t) {
                'f' -> {
                    args.add(ByteBuffer.wrap(data, pos, 4).order(ByteOrder.BIG_ENDIAN).float)
                    pos += 4
                }
                'i' -> {
                    args.add(ByteBuffer.wrap(data, pos, 4).order(ByteOrder.BIG_ENDIAN).int)
                    pos += 4
                }
                else -> pos += 4
            }
        }
        return args
    }

    private fun ByteArray.indexOf(byte: Byte, fromIndex: Int): Int {
        for (i in fromIndex until size) if (this[i] == byte) return i
        return -1
    }
}
