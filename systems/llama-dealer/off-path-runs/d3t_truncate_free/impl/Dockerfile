# d3t_truncate_free: TruncateReduce key arrays freed, built on d2_mask_stream.
#
# Reason: at d2's peak the resident table ranks TruncateReduce first (api.cpp:2857, 264.8 MB, 17.6%);
# the published line (from allocated-bytes tables) attacked the activation buffers first. Same lever
# as the published d4, stacked on d2. Displaced ordering: off-path.md.
#
#   podman build -f impl/Dockerfile -t llama-dealer-d3t_truncate_free impl/
FROM localhost/llama-dealer-d2_mask_stream

COPY patch/ /opt/EzPC/GPU-MPC/ext/sytorch/

# Rebuilt at IMAGE BUILD time, never at measurement time, and the rebuild fails if the binary
# comes out identical to its parent's -- a step that does not change the binary cannot measure a
# lever.
RUN /opt/rebuild_step.sh
