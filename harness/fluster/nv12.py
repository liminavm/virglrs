# SPDX-License-Identifier: MIT
# Copyright © 2026 Gustavo Noronha Silva
"""Run fluster with every 8-bit 4:2:0 vector decoded into NV12.

A GStreamer VA decoder picks its own output format upstream of fluster's `videoconvert`, and in
the stock guest it picks I420 -- so the guest allocates three-plane targets, and the renderer's
two-plane path is never taken. This pins the decoder's output to NV12 for the vectors whose
reference is I420, which leaves the oracle alone: the published md5 is still taken after the
convert to I420. Every other vector runs as fluster would run it.

Used as `python3 nv12.py <fluster arguments>`, from the directory fluster's checkout is in.
"""

import multiprocessing
import os
import runpy
import sys

UPSTREAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream")
sys.path.insert(0, UPSTREAM)

from fluster.decoders import gstreamer  # noqa: E402  (needs the path above)

stock = gstreamer.GStreamerVideo.gen_pipeline


def gen_pipeline(self, input_filepath, output_filepath, output_format, optional_params=None):
    try:
        planar = gstreamer.output_format_to_gst(output_format) == "I420"
    except KeyError:
        planar = False
    self.caps = "video/x-raw,format=NV12" if planar else "video/x-raw"
    return stock(self, input_filepath, output_filepath, output_format, optional_params)


gstreamer.GStreamerVideo.gen_pipeline = gen_pipeline
# Fluster builds each pipeline in a worker process. Python's default start method on Linux is no
# longer `fork`, and a worker that is not forked imports gstreamer.py afresh -- without the patch,
# so every vector would run as stock and the run would score the I420 path under this name.
multiprocessing.set_start_method("fork")
sys.argv = [os.path.join(UPSTREAM, "fluster.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
