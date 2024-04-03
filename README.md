# 2302-Computer-Vision-Object-Detection-and-Recognition

### Python packages required -
- mediapipe
- opencv
- numpy
- sklearn

#
- `ROI_coordinates.py` given the input directory containing raw images, finds the ROI (hands) and saves the cropped hand images in another directory.

    Usage: ```python3 ROI_coordinates.py <dataset_folder> <output_folder>```

- `training.py` the function `train(open_dataset, closed_dataset)` trains the SVG model and saves it as `clf.pkl`
    
    Usage: ```python3 training.py <open_dataset_folder> <closed_dataset_folder>```

- `predictor.py` given a directory containing raw images, marks hands (if any) as open or closed and saves the annotated images in output_path. 

    Usage: ```python3 predictor.py <classifier_file_path> <raw_images_path> <output_path>```

- `validation.py` evaluates the performance of trained model on validation dataset

    Usage: ```python3 validation.py <classifier_file_path> <valid_open_path> <valid_closed_path>```

- `bonus_camera_application.py` On running, first SVG model is trained and thereafter a camera window pops up. The features of the application are described in the report

    Usage: ```python3 bonus_camera_application.py <classifier_file_path>```

- `hog_scratch.py` descriptor_scratch(image) function takes cv2 grayscale image as input and returns its HoG feature descriptor

