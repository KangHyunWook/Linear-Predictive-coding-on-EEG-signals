# Linear-Predictive-coding-on-EEG-signals

Description: This code is to investigate if a linear predictive coding done on EEG signals can improve model performance.

<h2>How to train</h2>

```
python main.py --data-path [data_path] --model 1
```

To test the model add --test option at the end.

model options:
  1: One-dimensional Convolution,
  3: DNN(Deep Neural Network)  

Data options: 
  emotions.csv: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions
  mental-state.csv: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-mental-state
  
Place above datasets in the working directory.
  
  
