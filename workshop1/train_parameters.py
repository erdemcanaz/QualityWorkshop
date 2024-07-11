import os, math, random


PARAM_DATA_PATH_TO_IMPORT = 'linearly_seperated_data_500.txt'

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

    def __init__(self, data:list[Data]=None, weights:dict ={"bias":0, "usage":1, "rpm":1}):
        self.data = data
        self.weights = weights            

    def randomly_initialize_weights(self, bounds:tuple = (-1, 1)):
        self.weights = {
            "bias": random.uniform(bounds[0], bounds[1]),
            "usage": random.uniform(bounds[0], bounds[1]),
            "rpm": random.uniform(bounds[0], bounds[1])
        }

    def ReLU(self, z:float=None, a_coefficient:float = 0.001):
        if z <= -10:
            return (0) + (a_coefficient * (z+10))
        elif -10 < z <= 10:
            return (0.5)+ (z/20)
        else:
            return (1) + (a_coefficient * (z-10))
        
    def ReLU_prime(self, z:float=None, a_coefficient:float = 0.001):
        if z <= -10:
            return a_coefficient
        elif  -10 < z <= 10:
            return 1/20
        else:
            return a_coefficient

    def train(self, epoch:int = 1, learning_rate:float = 0.1, batch_size = 8, error_threshold = 1):
        number_of_batches = math.ceil(len(self.data) / batch_size)

        for i in range(epoch):  

            epoch_error_sum = 0

            for batch_no in range(number_of_batches):
                batch = self.data[batch_no * batch_size : (batch_no + 1) * batch_size]

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

                    term_1 = (-2)*(is_broken - y)
                    term_2 = self.ReLU_prime(z)                   
                
                    batch_gradient_sum["bias"] += term_1 * term_2 * 1
                    batch_gradient_sum["usage"] += term_1 * term_2 * daily_usage
                    batch_gradient_sum["rpm"] += term_1 * term_2 * rpm                         

                self.weights["bias"] -= learning_rate * batch_gradient_sum["bias"] / batch_size
                self.weights["usage"] -= learning_rate * batch_gradient_sum["usage"] / batch_size
                self.weights["rpm"] -= learning_rate * batch_gradient_sum["rpm"] / batch_size

            if i % 1000 == 0:               
                print(f"Epoch: {i:6}, Error: {epoch_error_sum:8.2f}, Accuracy: {self.calculate_categorization_succes():.2f} Weights: {self.weights}")

            if epoch_error_sum/len(self.data) < error_threshold:
                print(f"Training is completed at epoch {i} because the error is below the threshold.")
                break

    def calculate_categorization_succes(self)->float: 
        succes_count = 0
        failure_count = 0

        for data in self.data:
            daily_usage = data.get_daily_usage()
            rpm = data.get_rpm()
            is_broken = data.get_is_broken()

            z = self.weights["bias"] * 1 + self.weights["usage"] * daily_usage + self.weights["rpm"] * rpm
            y = self.ReLU(z)   
            
            if y >= 0.5:
                prediction = 1
            else:
                prediction = 0
            
            if prediction == is_broken:
                succes_count += 1
            else:
                failure_count += 1


        return succes_count / (succes_count + failure_count)

MLP_model = MLP(data, weights= {'bias': -6.365834521303161, 'usage': 0.06659837028594247, 'rpm': 0.0020762566186700434})
#MLP_model.randomly_initialize_weights(bounds=(-5,5))
MLP_model.train(epoch=int(1e7), learning_rate=0.0003, batch_size=64, error_threshold=0.05)
MLP_model.calculate_categorization_succes()





