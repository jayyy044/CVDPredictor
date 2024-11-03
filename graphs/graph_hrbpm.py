import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

np.random.seed(42)  
mins = range(60)  
average_bpm = np.random.randint(60, 90, size=len(mins))  

# Create a DataFrame to store minute-based BPM data
bpm_data = pd.DataFrame({
    'Minute': mins,
    'Average_BPM': average_bpm
})

# Plot the data without smoothing
plt.figure(figsize=(12, 8))
plt.plot(bpm_data['Minute'], bpm_data['Average_BPM'], color='r', label='Average BPM')
plt.xlabel('Time (last 60 minutes)')
plt.ylabel('Average BPM')
plt.title('Average Heart Rate Over Time')
plt.legend()
plt.grid(True)
plt.ylim(20, 140)
plt.show()

