import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub

from utility_functions import predict, sorted_class_prob

parser = argparse.ArgumentParser("Enter arguments to predict on a flower image")
parser.add_argument("image_path", type=str, help="The path of the image files.")
parser.add_argument("saved_model", type=str, help="The saved model.")
parser.add_argument("--top_k", type=int, default=5, 
                    help="The top k most likely prediction probabilities and "
                    "classes.  The default is 5")
parser.add_argument("--category_names", type=str, help="The path to a JSON file"
                    " mapping labels to flower names.  The default is "
                    "label_map.json", default="label_map.json")

# command line arguments passed into variables
args = parser.parse_args()
image_path = args.image_path
saved_model = args.saved_model
top_k = args.top_k
category_names = args.category_names

if __name__ == "__main__":
    print()
    print("Image network will predict a flower image classification on {}, "
          "utilize saved model from {}, display top_{} class probabilities, and "
          "utilize {} file for mapping labels to flowers.".
          format(image_path, saved_model, top_k, category_names))
    print()
    
    # loads a Keras Model
    try:
        
        reloaded_keras_model = (
            tf.keras.models.load_model(
                saved_model, compile=False, 
                custom_objects={"KerasLayer":hub.KerasLayer})
            )
    except:
        print("")
        print("Could not load the provided saved model. Please try again.")
        exit()
    
    # check that top_k input is valid
    try:
        if top_k <= 0 or type(top_k) != int:
            raise
    except:
        print("")
        print("top_k needs to be an integer value greater than 0")
        exit()
    
    # execute the image prediction
    try:        
        probs, classes = predict(image_path, reloaded_keras_model, top_k)
    except:
        print("")
        print("Could not load predict on the provided image file.")
        exit()
   
    # load and read the image flower json to map index to class name
    try:
        with open(category_names, "r") as f:
            class_names = json.load(f)
    except:
        print("")
        print("Could not load the provided json file.")
        exit()
    
    df_sorted = sorted_class_prob(class_names, classes, probs, False)
    
    print("")
    print(df_sorted)