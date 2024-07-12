import os, math, random
import numpy as np
import itertools
import matplotlib.pyplot as plt


PARAM_DATA_PATH_TO_IMPORT = 'linearly_seperated_data_500.txt'
PARAM_X_LIMITS = [0, 24]
PARAM_Y_LIMITS = [0, 4500]

class Data:
    def __init__(self, data):
        self.daily_usage = data[0]
        self.rpm = data[1]
        self.is_broken = data[2]

    def get_daily_usage(self):
        return self.daily_usage
    
    def get_rpm(self):
        return self.rpm
    
    def get_is_broken(self):
        return self.is_broken

script_path = os.path.dirname(os.path.abspath(__file__))
text_path = os.path.join(script_path, f"data\{PARAM_DATA_PATH_TO_IMPORT}")

data = []
with open(text_path, 'r') as file:
    for line in file:
        values = [float(val) for val in line.split()]
        data.append(Data(values))


class MLP:

    def __init__(self, data: list = None, weights: dict = None):
        self.data = data   

        if weights is None:
            self.randomly_initialize_weights()
        else:
            self.weights = weights

        self.momentum = {"bias": 0, "usage": 0, "rpm": 0} # For momentum

    def randomly_initialize_weights(self, bounds: tuple = (-1, 1)):
        self.weights = {
            "bias": random.uniform(bounds[0], bounds[1]),
            "usage": random.uniform(bounds[0], bounds[1]),
            "rpm": random.uniform(bounds[0], bounds[1])
        }

    def ReLU(self, z: float, a_coefficient: float = 0.001):
        if z <= -10:
            return (0) + (a_coefficient * (z+10))
        elif -10 < z <= 10:
            return (0.5)+ (z/20)
        else:
            return (1) + (a_coefficient * (z-10))

    def ReLU_prime(self, z: float, a_coefficient: float = 0.001):
        if z <= -10:
            return a_coefficient
        elif -10 < z <= 10:
            return 1/20
        else:
            return a_coefficient

    def train(self, epoch: int = 1, learning_rate: float = 0.1, batch_size: int = 8, error_threshold: float = 1.0, momentum_factor: float = 0.9, iteration_to_print: int = 1000, show_plot: bool = False):
        if not self.data:
            print("No data provided for training.")
            return

        number_of_batches = math.ceil(len(self.data) / batch_size)
        
        for i in range(epoch):
            epoch_error_sum = 0

            for batch_no in range(number_of_batches):
                start_index = batch_no * batch_size
                end_index = min((batch_no + 1) * batch_size, len(self.data))
                batch = self.data[start_index:end_index]

                batch_gradient_sum = {
                    "bias": 0,
                    "usage": 0,
                    "rpm": 0
                }
                batch_error_sum = 0
                for sample in batch:
                    daily_usage = sample.get_daily_usage()
                    rpm = sample.get_rpm()
                    is_broken = sample.get_is_broken()

                    # Forward pass
                    z = self.weights["bias"] * 1 + self.weights["usage"] * daily_usage + self.weights["rpm"] * rpm
                    y = self.ReLU(z)
                    epoch_error_sum += (is_broken - y) ** 2
                    batch_error_sum += (is_broken - y) ** 2

                    term_1 = (-2) * (is_broken - y)
                    term_2 = self.ReLU_prime(z)

                    batch_gradient_sum["bias"] += term_1 * term_2 * 1
                    batch_gradient_sum["usage"] += term_1 * term_2 * daily_usage
                    batch_gradient_sum["rpm"] += term_1 * term_2 * rpm

                batch_size_actual = len(batch)
                if batch_size_actual > 0:
                    # Update with momentum
                    self.momentum["bias"] = momentum_factor * self.momentum["bias"] + (1 - momentum_factor) * batch_gradient_sum["bias"] / batch_size_actual
                    self.momentum["usage"] = momentum_factor * self.momentum["usage"] + (1 - momentum_factor) * batch_gradient_sum["usage"] / batch_size_actual
                    self.momentum["rpm"] = momentum_factor * self.momentum["rpm"] + (1 - momentum_factor) * batch_gradient_sum["rpm"] / batch_size_actual

                    self.weights["bias"] -= learning_rate * self.momentum["bias"]
                    self.weights["usage"] -= learning_rate * self.momentum["usage"]
                    self.weights["rpm"] -= learning_rate * self.momentum["rpm"]

            if i % iteration_to_print == 0:  # Changed to print less frequently
                print(f"Epoch: {i:6}, Succes (all): {self.__calculate_categorization_succes(distance_to_threshold=0.0):8.3f}, Succes (ignore 0.4-0.6): {self.__calculate_categorization_succes(distance_to_threshold=0.1):8.3f} Error: {epoch_error_sum/len(self.data):8.5f}, Weights: {self.weights}")
                if show_plot: self.plot_prediction_space()

            if epoch_error_sum / len(self.data) < error_threshold:
                print(f"Training is completed at epoch {i} because the error is below the threshold.")
                break

    def __calculate_categorization_succes(self, distance_to_threshold:float = 0.00)->float: 
        succes_count = 0
        failure_count = 0

        for data in self.data:
            daily_usage = data.get_daily_usage()
            rpm = data.get_rpm()
            is_broken = data.get_is_broken()

            z = self.weights["bias"] * 1 + self.weights["usage"] * daily_usage + self.weights["rpm"] * rpm
            y = self.ReLU(z)   
            
            if abs(y-0.5) < distance_to_threshold:
                continue

            if y >= 0.5:
                prediction = 1
            else:
                prediction = 0
            
            if prediction == is_broken:
                succes_count += 1
            else:
                failure_count += 1


        return succes_count / max(1,(succes_count + failure_count))

    def predict(self, daily_usage: float, rpm: float)->int:
        z = self.weights["bias"] * 1 + self.weights["usage"] * daily_usage + self.weights["rpm"] * rpm
        y = self.ReLU(z)

        if y >= 0.5:
            return 1
        else:
            return 0
        
    def plot_prediction_space(self, x_limits: list[float, float] = None, y_limits: list[float, float] = None):
        daily_usage = np.linspace(PARAM_X_LIMITS[0], PARAM_X_LIMITS[1], 50)
        rpm = np.linspace(PARAM_Y_LIMITS[0], PARAM_Y_LIMITS[1], 50)

        prediction_space = list(itertools.product(daily_usage, rpm))

        points = []
        for i in range(len(prediction_space)):
            points.append([prediction_space[i][0], prediction_space[i][1], self.predict(prediction_space[i][0], prediction_space[i][1])])

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        colors = ['r' if point[2] == 1 else 'g' for point in points]

        red_patch = plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=5, label='Bozulur Tahmini')
        blue_patch = plt.Line2D([], [], color='green', marker='o', linestyle='None', markersize=5, label='Bozulmaz Tahmini')

        plt.ion()  # Turn on interactive mode
        ax = plt.gca()  # Get current axes
        ax.legend(handles=[red_patch, blue_patch])
        ax.scatter(x_values, y_values, c=colors)
        ax.set_xlabel('Günlük Kullanım (Saat)')
        ax.set_ylabel('Motor RPM')
        ax.set_title('Motor Bozulacak mı ? Tahmini ')

        if x_limits:
            ax.set_xlim(x_limits)
        if y_limits:
            ax.set_ylim(y_limits)

        plt.draw()  # Update the figure
        plt.pause(0.25)  # Pause briefly to allow

pretrained_weights = {
    "workshop_bias": {'bias': -5, 'usage': 0.45, 'rpm': 0.0090},
    "11.07.2024_19_38": {'bias': -15.939309014984318, 'usage': 0.6011307820678724, 'rpm': 0.006601504978600123},
    "11.07.2024_19_54": {'bias': -28.612332064196746, 'usage': 0.8093490579957665, 'rpm': 0.006939621724616913},
    "11.07.2024_21_56": {'bias': -53.106915838715764, 'usage': 1.6370307419016032, 'rpm': 0.016467756843615674}

}
MLP_model = MLP(data, weights= pretrained_weights["workshop_bias"])
MLP_model.train(epoch=int(1e7), learning_rate=0.0002, batch_size=32, error_threshold=0.01, momentum_factor=0.90, iteration_to_print=10, show_plot=True)




