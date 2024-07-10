# This script generates a dummy dataset for the whether an electric motor will fail until the next year or not for a given set of features.
# DATA   -> motor rpm, daily usage in hours
# OUTPUT -> 1 if motor will fail until the next year, 0 otherwise
import random
import pickle
import matplotlib.pyplot as plt

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

samples = generate_linearly_seperated_data(number_of_samples=1000, m=-110, y_0=3500, x_limits=[0, 24], y_limits=[0, 4500])

x_values = [sample[0] for sample in samples]
y_values = [sample[1] for sample in samples]
colors = ['r' if sample[2] == 1 else 'g' for sample in samples]

# Add legend for red and blue
red_patch = plt.Line2D([], [], color='red', marker='o', linestyle='None',
                      markersize=5, label='Bozuldu')
blue_patch = plt.Line2D([], [], color='green', marker='o', linestyle='None',
                       markersize=5, label='Bozulmadı')
plt.legend(handles=[red_patch, blue_patch])
plt.scatter(x_values, y_values, c=colors)
plt.xlabel('Günlük Kullanım (Saat)')
plt.ylabel('Motor RPM')
plt.title('Motor Bozulma Verisi')
plt.show()

# Export samples to a text file
with open('samples.txt', 'w') as file:
    for sample in samples:
        file.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\n")