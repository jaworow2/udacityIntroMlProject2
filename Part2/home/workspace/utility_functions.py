import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

def process_image(image):
    ''' 
    Converts, resizes, and normalizes a NumPy array image into a 
    uniform shape
    
    argument:
        image: NumPy array image with varying shape (varies, varies, 3)
    
    returns:
        image: Numpy array with shape (224, 224, 3)
    
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity Intro to Machine Learning Nano Degree Training material.
    '''
    image_size = 224
    
    # convert image into a TensorFlow Tensor
    image_file = tf.convert_to_tensor(image)
    
    # resize to the appropriate size
    image_file = tf.image.resize(image_file, (image_size, image_size))
    
    # normalize the pixels
    image_file /= 255
    
    # convert into NumPy array
    image = image_file.numpy()
    
    return image

def predict(image_path, model, top_k):
    ''' 
    Takes a passed image, trained mode, and desired top 
    probabilities and returns prediction probabilities 
    for each image class
    
    argument:
        image_path: image to be predicted
        mode: trained model
        top_k: number of top prediction probabilities
    
    returns:
        class_prob: predicted probability for image classes
        class_index: predicted class index
        
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity Intro to Machine Learning Nano Degree Training material.
    '''
    # load image
    im = Image.open(image_path)
    
    # convert to NumPy array
    test_image = np.asarray(im)
    
    # reshape Numpy array with shape (224, 224, 3)
    processed_test_image = process_image(test_image)
    
    # add extra dimension (1, 224, 224, 3) for the required model format
    processed_test_image = np.expand_dims(processed_test_image, axis=0)

    # numpy array 1x102 of the probability prediction value for each flower image
    image_prediction = model.predict(processed_test_image)
    
    # tensorflow method to identify a tensor of the top K predicted classes 
    # probabilities and indices
    class_prob, class_index = tf.math.top_k(image_prediction, top_k) 
    
    # convert predicted class tensor to numpy array list
    class_prob = class_prob.numpy().tolist() 

    # convert predicted class index tensor to numpy array list
    class_index = class_index.numpy().tolist()
    
    # add 1 since index starts at 0
    class_index = [i+1 for i in class_index[0]]
    
    return class_prob[0], class_index

def sorted_class_prob(class_names, classes, probs, ascend):
    ''' 
    Takes a dictionary of flower image class_names, predicted
    flower image index, predicted probability, and sort preference 
    and returens a sorted dataframe
    
    argument:
        class_names: dictionary of flower images
        classes: predicted flower indexes
        probs: predicted flower probabilities
        ascend: sorted direction
    
    returns:
        df_sorted: a dataframe of sorted indexes and classes
    '''
    class_display = [class_names[str(class_index)] for class_index in classes]
    df = pd.DataFrame(
        {"class_name" : class_display,
         "prob" : probs})
    df_sorted = df.sort_values("prob", ascending=ascend)
    return df_sorted