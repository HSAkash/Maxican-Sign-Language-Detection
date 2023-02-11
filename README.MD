# Maxican Sign language Detection
It can detect maxican sign language. It accuracy level is 97.31%.<br>
Here we take only body and finger key points. In this paper or project we using attention model to detect the sign language. Its performance level is good enough for most applications. Input raw data goes through conv1d layer then attention model which works well.

## Using layers, Activation, Optimizer for model recognition
Layers:
Conv1d
Attention
Concatenate
Flatten
Dense

Activation:
ReLU
Softmax

Optimization:
Adam

Loss Function:
Categorical crossentropy

Metrix
Accuracy

Run command:
```
pip install -r requirements.txt
python attention_Mexican_Sign_Language.py
```