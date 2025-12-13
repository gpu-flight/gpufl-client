from gpufl.viz import read_df
from gpufl.viz.timeline import plot_memory_timeline, plot_utilization_timeline
import matplotlib.pyplot as plt

df = read_df("stress.scope.log")

plot_memory_timeline(df, gpu_id=0).show()
plot_utilization_timeline(df, gpu_id=0).show()

plt.show()