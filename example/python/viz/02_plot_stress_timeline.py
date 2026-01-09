import gpufl.viz as viz
import os

# Use 'stress' logs which have rich CUDA metrics
viz.init("stress.*.log")

# Show combined timeline with GPU, PCIe, and Host metrics
# It will now also show kernel occupancy and enriched kernel names
viz.show()
