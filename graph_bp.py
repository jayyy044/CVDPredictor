import numpy as np
import matplotlib.pyplot as plt

# Generate time data over 60 minutes
time = np.arange(0, 61)  # 61 points from 0 to 60 minutes

# Initialize systolic BP array
systolic_bp = []

# Starting value for systolic BP within 140-170 mmHg
sys_bp_value = np.random.randint(140, 170)
systolic_bp.append(sys_bp_value)

# Generate systolic BP values
for _ in range(1, len(time)):
    # Change systolic BP by -1, 0, or 1 mmHg
    sys_bp_change = np.random.choice([-1, 0, 1])
    sys_bp_value += sys_bp_change
    sys_bp_value = max(140, min(170, sys_bp_value))  # Keep within range
    systolic_bp.append(sys_bp_value)

# Calculate diastolic BP as proportional to systolic BP
k = 0.6  # Proportionality constant
diastolic_bp = [max(80, min(110, int(k * sys_bp))) for sys_bp in systolic_bp]

# Plot the data
plt.figure(figsize=(12, 8))
plt.plot(time, systolic_bp, color='blue', label='Systolic BP')
plt.plot(time, diastolic_bp, color='orange', label='Diastolic BP')
plt.xlabel('Time (minutes)')
plt.ylabel('Blood Pressure (mmHg)')
plt.title('Systolic and Diastolic Blood Pressure Over Time')
plt.legend()
plt.grid(True)
plt.ylim(70, 180)
plt.show()