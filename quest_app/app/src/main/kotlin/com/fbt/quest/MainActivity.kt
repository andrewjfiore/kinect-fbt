package com.fbt.quest

import android.content.SharedPreferences
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicReference

class MainActivity : AppCompatActivity() {

    private lateinit var prefs: SharedPreferences
    private lateinit var etServerIp: EditText
    private lateinit var etServerPort: EditText
    private lateinit var etVrChatPort: EditText
    private lateinit var btnConnect: Button
    private lateinit var tvStatus: TextView
    private lateinit var tvStatusDot: View
    private lateinit var tvTrackers: TextView
    private lateinit var tvServerFps: TextView
    private lateinit var tvForwardFps: TextView
    private lateinit var tvCamerasActive: TextView
    private lateinit var btnRecenter: Button
    private lateinit var sliderHeight: SeekBar
    private lateinit var tvHeight: TextView

    private val trackerStates = ConcurrentHashMap<Int, TrackerState>()
    private val serverStatus = AtomicReference(ServerStatus())

    private var oscReceiver: OscReceiver? = null
    private var forwarder: VRChatOscForwarder? = null
    private var receiverJob: Job? = null
    private var oscSenderSocket: java.net.DatagramSocket? = null
    private var serverIp: String = ""
    private var serverPort: Int = 39571
    private var isConnected = false

    private val MIN_HEIGHT = 1.4f
    private val MAX_HEIGHT = 2.1f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences("fbt_prefs", MODE_PRIVATE)
        buildUi()
        restoreSettings()
        startUiUpdateLoop()
    }

    private fun buildUi() {
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }

        // --- Settings Section ---
        layout.addView(sectionHeader("⚙️  Settings"))

        etServerIp = EditText(this).apply {
            hint = "Server IP (e.g. 192.168.1.100)"
            inputType = android.text.InputType.TYPE_CLASS_TEXT
        }
        layout.addView(labeled("Server IP", etServerIp))

        etServerPort = EditText(this).apply {
            hint = "Server Port"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
        }
        layout.addView(labeled("Server Port", etServerPort))

        etVrChatPort = EditText(this).apply {
            hint = "VRChat OSC Port"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
        }
        layout.addView(labeled("VRChat Port (localhost)", etVrChatPort))

        btnConnect = Button(this).apply {
            text = "Connect"
            setOnClickListener { toggleConnection() }
        }
        layout.addView(btnConnect)

        // --- Status Section ---
        layout.addView(sectionHeader("📡  Status"))

        tvStatusDot = View(this).apply {
            setBackgroundColor(0xFFAAAAAA.toInt())
            layoutParams = LinearLayout.LayoutParams(24, 24).apply { setMargins(0, 0, 16, 0) }
        }
        tvStatus = TextView(this).apply { text = "Disconnected" }
        layout.addView(LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(tvStatusDot)
            addView(tvStatus)
        })

        tvTrackers = TextView(this).apply { text = "No trackers" }
        layout.addView(tvTrackers)

        tvServerFps = TextView(this).apply { text = "Server FPS: --" }
        layout.addView(tvServerFps)

        tvForwardFps = TextView(this).apply { text = "Forward FPS: --" }
        layout.addView(tvForwardFps)

        tvCamerasActive = TextView(this).apply { text = "Cameras: --" }
        layout.addView(tvCamerasActive)

        // --- Calibration Section ---
        layout.addView(sectionHeader("🎯  Calibration"))

        btnRecenter = Button(this).apply {
            text = "Send Recenter"
            setOnClickListener { sendRecenter() }
        }
        layout.addView(btnRecenter)

        tvHeight = TextView(this).apply { text = "Height: 1.70m" }
        layout.addView(tvHeight)

        sliderHeight = SeekBar(this).apply {
            max = 100
            progress = heightToProgress(1.7f)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                    val h = progressToHeight(progress)
                    tvHeight.text = "Height: ${"%.2f".format(h)}m"
                    if (fromUser) sendUserHeight(h)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        layout.addView(sliderHeight)

        val scroll = ScrollView(this)
        scroll.addView(layout)
        setContentView(scroll)
    }

    private fun restoreSettings() {
        etServerIp.setText(prefs.getString("server_ip", "192.168.1.100"))
        etServerPort.setText(prefs.getInt("server_port", 39571).toString())
        etVrChatPort.setText(prefs.getInt("vrchat_port", 9000).toString())
        val h = prefs.getFloat("user_height", 1.7f)
        sliderHeight.progress = heightToProgress(h)
        tvHeight.text = "Height: ${"%.2f".format(h)}m"
    }

    private fun saveSettings() {
        prefs.edit().apply {
            putString("server_ip", etServerIp.text.toString())
            putInt("server_port", etServerPort.text.toString().toIntOrNull() ?: 39571)
            putInt("vrchat_port", etVrChatPort.text.toString().toIntOrNull() ?: 9000)
            putFloat("user_height", progressToHeight(sliderHeight.progress))
            apply()
        }
    }

    private fun toggleConnection() {
        if (isConnected) disconnect() else connect()
    }

    private fun connect() {
        serverIp = etServerIp.text.toString().trim()
        serverPort = etServerPort.text.toString().toIntOrNull() ?: 39571
        val vrChatPort = etVrChatPort.text.toString().toIntOrNull() ?: 9000

        saveSettings()

        oscSenderSocket = java.net.DatagramSocket()

        oscReceiver = OscReceiver(
            port = serverPort,
            onTracker = { t ->
                trackerStates[t.id] = t
            },
            onStatus = { s ->
                serverStatus.set(s)
            },
            onCalibrate = {
                // Server-initiated recenter acknowledgement
            }
        )
        oscReceiver?.start(lifecycleScope)

        forwarder = VRChatOscForwarder(
            vrChatPort = vrChatPort,
            getTracker = { id -> trackerStates[id] },
            availableTrackerIds = { trackerStates.keys.toSet() }
        )
        forwarder?.start(lifecycleScope)

        isConnected = true
        btnConnect.text = "Disconnect"
        tvStatusDot.setBackgroundColor(0xFF4CAF50.toInt())
        tvStatus.text = "Connected to $serverIp:$serverPort"
    }

    private fun disconnect() {
        oscReceiver?.stop()
        forwarder?.stop()
        oscSenderSocket?.close()
        oscSenderSocket = null
        isConnected = false
        btnConnect.text = "Connect"
        tvStatusDot.setBackgroundColor(0xFFAAAAAA.toInt())
        tvStatus.text = "Disconnected"
        trackerStates.clear()
    }

    private fun sendRecenter() {
        if (!isConnected) return
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val msg = buildOscMessage("/calibrate", emptyList())
                val addr = java.net.InetAddress.getByName(serverIp)
                val pkt = java.net.DatagramPacket(msg, msg.size, addr, serverPort)
                oscSenderSocket?.send(pkt)
            } catch (e: Exception) {
                // ignore
            }
        }
    }

    private fun sendUserHeight(height: Float) {
        if (!isConnected) return
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val msg = buildOscMessage("/config/user_height", listOf(height))
                val addr = java.net.InetAddress.getByName(serverIp)
                val pkt = java.net.DatagramPacket(msg, msg.size, addr, serverPort)
                oscSenderSocket?.send(pkt)
            } catch (e: Exception) {
                // ignore
            }
        }
    }

    private fun buildOscMessage(address: String, args: List<Float>): ByteArray {
        fun oscStr(s: String): ByteArray {
            val b = s.toByteArray(Charsets.US_ASCII)
            val pad = ((b.size + 1) + 3) / 4 * 4
            return b + ByteArray(pad - b.size)
        }
        val typeTag = if (args.isEmpty()) "," else "," + "f".repeat(args.size)
        val addrBytes = oscStr(address)
        val tagBytes = oscStr(typeTag)
        val argBytes = ByteArray(args.size * 4)
        val buf = java.nio.ByteBuffer.wrap(argBytes).order(java.nio.ByteOrder.BIG_ENDIAN)
        args.forEach { buf.putFloat(it) }
        return addrBytes + tagBytes + argBytes
    }

    private fun startUiUpdateLoop() {
        lifecycleScope.launch(Dispatchers.Main) {
            while (true) {
                updateStatusUi()
                delay(500)
            }
        }
    }

    private fun updateStatusUi() {
        val status = serverStatus.get()
        tvServerFps.text = "Server FPS: ${"%.1f".format(status.fps)}"
        tvForwardFps.text = "Forward FPS: ${"%.1f".format(forwarder?.fps ?: 0f)}"
        tvCamerasActive.text = "Cameras: ${status.camerasActive}"

        if (trackerStates.isEmpty()) {
            tvTrackers.text = "No trackers"
        } else {
            val sb = StringBuilder()
            for ((id, t) in trackerStates.toSortedMap()) {
                val bar = buildConfBar(t.confidence)
                sb.appendLine("Tracker $id: $bar (${"%.2f".format(t.confidence)})")
            }
            tvTrackers.text = sb.toString().trimEnd()
        }
    }

    private fun buildConfBar(conf: Float): String {
        val filled = (conf * 10).toInt().coerceIn(0, 10)
        return "[${"█".repeat(filled)}${"░".repeat(10 - filled)}]"
    }

    private fun heightToProgress(h: Float): Int =
        ((h - MIN_HEIGHT) / (MAX_HEIGHT - MIN_HEIGHT) * 100).toInt().coerceIn(0, 100)

    private fun progressToHeight(p: Int): Float =
        MIN_HEIGHT + p.toFloat() / 100f * (MAX_HEIGHT - MIN_HEIGHT)

    private fun sectionHeader(title: String) = TextView(this).apply {
        text = title
        textSize = 18f
        setPadding(0, 24, 0, 8)
    }

    private fun labeled(label: String, view: View): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(TextView(this@MainActivity).apply { text = label })
            addView(view)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        disconnect()
    }
}
