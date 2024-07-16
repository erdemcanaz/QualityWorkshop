from ultralytics import YOLO
import torch

def main():
    print(f" torch.cuda.device_count(): {torch.cuda.device_count()}")
    torch.cuda.set_device(0) # Set to your desired GPU number

    # #==== Option 1: Train directly from the model definition
    model = YOLO('yolov8n.yaml')

    #==== Option 2: Build from YAML and transfer pretrained weights
    # model_path_to_train_on = input("Enter the path to the model to train on (new): ")
    # model = YOLO('yolov8n.yaml').load(model_path_to_train_on)

    RUN_ON_CUDA = True
    if RUN_ON_CUDA and torch.cuda.is_available():
        model.to('cuda')
        print("GPU (CUDA) is detected. Training will be done on GPU.")
    else:
        r = input("GPU (CUDA) is not detected or prefered. Should continue with CPU? (y/n):")
        if r != 'y':
            print("Exiting...")
            exit()

    #Train the model
    experiment = input("Enter the name of your experiment: ")
    save_dir = input("Enter the path to your save directory: ")
    yaml_file = input("Enter the path to your yaml file: ")

    model.train(
        data=yaml_file,
        #classes = [0,1,3],
        epochs=200, 
        save_dir=save_dir,
        project=save_dir,
        name=experiment,
        imgsz=640,
        save_period = 25,
        batch = 8,
        plots = True
    )

if __name__ == '__main__':
    main()