# This script generates a dummy dataset for the whether an electric motor will fail until the next year or not for a given set of features.
# DATA   -> motor rpm, daily usage in hours
# OUTPUT -> 1 if motor will fail until the next year, 0 otherwise
import random, os
import matplotlib.pyplot as plt

PARAM_NUMBER_OF_SAMPLES = 500
PARAM_M = -110
PARAM_Y_0 = 3500
PARAM_X_LIMITS = [0, 24]
PARAM_Y_LIMITS = [0, 4500]


def generate_linearly_seperated_data(number_of_samples:int= 1, m:float = None, y_0:float = None, x_limits:list[float,float] = None, y_limits:list[float,float]=None):
    # x axiss -> motor daily usage in hours
    # y axiss -> motor rpm

    samples = []
    for i in range(number_of_samples):
        x = random.uniform(x_limits[0], x_limits[1])
        y = random.uniform(y_limits[0], y_limits[1])
        output = 1 if y > m*x + y_0 else 0
        samples.append((x, y, output))

    return samples    
samples = generate_linearly_seperated_data(PARAM_NUMBER_OF_SAMPLES, PARAM_M, PARAM_Y_0, PARAM_X_LIMITS, PARAM_Y_LIMITS)

x_values = [sample[0] for sample in samples]
y_values = [sample[1] for sample in samples]
colors = ['r' if sample[2] == 1 else 'g' for sample in samples]

red_patch = plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=5, label='Bozuldu')
blue_patch = plt.Line2D([], [], color='green', marker='o', linestyle='None', markersize=5, label='Bozulmadı')
plt.legend(handles=[red_patch, blue_patch])
plt.scatter(x_values, y_values, c=colors)
plt.xlabel('Günlük Kullanım (Saat)')
plt.ylabel('Motor RPM')
plt.title('Motor Bozulma Verisi')
plt.show()

# Get the current script path
script_path = os.path.dirname(os.path.abspath(__file__))

# Export samples to a text file in the current script path
with open(os.path.join(script_path, f"data\linearly_seperated_data_{PARAM_NUMBER_OF_SAMPLES}.txt"), 'w') as file:
    for sample in samples:
        file.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\n")