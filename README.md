# Reinforced Knowledge Distillation for time series regression

This repository is for paper "Reinforced Knowledge Distillation for Time Series Regression". The dataset used in this repository is C-MAPSS for machine RUL prediction task.


### Prerequisites

Pre-process the C-MAPSS data for training, validation and testing. 
```
python cmapss_data_processing_train_valid_test.py
```

## Teacher Generation
Generate different types of LSTM-based teachers for reinforced knowledge distillation. 

```
python lstm_teacher_negative_correlation_learning.py
```

```
python lstm_teacher_indepentent.py
```

```
python lstm_teacher_sanpshot.py
```

## Reinforced knowledge distillation for multiple teachers scenario

rl_kd_mutilple_teacher_reinforce_learning.py includes several model implementation with different types of teachers.
--teacher: ind, ncl, snapshot
--dataset: FD001, FD002, FD003, FD004

```
python rl_kd_mutilple_teacher_reinforce_learning.py
```
