import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.dates as mdates

np.random.seed(42)  
mins = range(60)  
average_bpm = np.random.randint(60, 90, size=len(mins))  

bpm_data = pd.DataFrame({
    'Minute': mins,
    'Average_BPM': average_bpm
})

x_new = np.linspace(min(bpm_data['Minute']), max(bpm_data['Minute']), 5000)  # More points for smoothness
spl = make_interp_spline(bpm_data['Minute'], bpm_data['Average_BPM'], k=3)  # Cubic spline
y_smooth = spl(x_new)

plt.figure(figsize=(12, 8))
plt.plot(x_new, y_smooth, color='r', label='Average BPM')
plt.xlabel('Time (last 60 minutes)')
plt.ylabel('Average BPM')
plt.title('Average Heart Rate per Hour')
plt.legend()
plt.grid(True)
plt.ylim(50, 110)
plt.show()

