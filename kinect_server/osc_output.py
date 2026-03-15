"""
OSC output: bundle and send FBT tracker data over UDP.
"""
import logging
import sys
import time
from typing import List

from pythonosc import osc_bundle_builder, osc_message_builder, udp_client

from fusion import TrackerData

logger = logging.getLogger(__name__)


class OSCOutput:
    def __init__(self, target_ip: str, target_port: int, dry_run: bool = False):
        self.target_ip = target_ip
        self.target_port = target_port
        self.dry_run = dry_run
        self._client = None
        if not dry_run:
            try:
                self._client = udp_client.SimpleUDPClient(target_ip, target_port)
                logger.info(f"OSC output: {target_ip}:{target_port}")
            except Exception as e:
                logger.error(f"Failed to create OSC client: {e}")

    def send(self, trackers: List[TrackerData], cameras_active: int, joints_tracked: int, fps: float):
        bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)

        for t in trackers:
            pos = t.position
            rot = t.rotation
            conf = t.confidence

            bundle.add_content(self._msg(f"/fbt/tracker/{t.tracker_id}/position", [float(pos[0]), float(pos[1]), float(pos[2])]))
            bundle.add_content(self._msg(f"/fbt/tracker/{t.tracker_id}/rotation", [float(rot[0]), float(rot[1]), float(rot[2])]))
            bundle.add_content(self._msg(f"/fbt/tracker/{t.tracker_id}/confidence", [float(conf)]))

        bundle.add_content(self._msg("/fbt/status", [int(cameras_active), int(joints_tracked), float(fps)]))

        built = bundle.build()

        if self.dry_run:
            for t in trackers:
                print(f"  /fbt/tracker/{t.tracker_id}/position {t.position[0]:.3f} {t.position[1]:.3f} {t.position[2]:.3f} (conf={t.confidence:.2f})")
            print(f"  /fbt/status cameras={cameras_active} joints={joints_tracked} fps={fps:.1f}")
            return

        if self._client is None:
            logger.warning("OSC client not initialised — skipping send")
            return
        try:
            self._client._sock.sendto(built.dgram, (self.target_ip, self.target_port))
        except Exception as e:
            logger.error(f"OSC send error: {e}")

    def _msg(self, address: str, args: list):
        mb = osc_message_builder.OscMessageBuilder(address=address)
        for arg in args:
            if isinstance(arg, int):
                mb.add_arg(arg, mb.ARG_TYPE_INT)
            else:
                mb.add_arg(arg, mb.ARG_TYPE_FLOAT)
        return mb.build()
