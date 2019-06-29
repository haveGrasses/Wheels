""" same as keras.utils.to_categorical """
import numpy as np

# encoding
d = np.asarray([0, 2, 2, 0, 8, 0, 2])
unique_val = sorted(set(d))  # one-hot columns
num_classes = len(unique_val)  # one-hot column numbers
num_to_int = dict(zip(unique_val, range(num_classes)))  # num to int encode map
integer_enc = [num_to_int[i] for i in d]  # num to int encode

one_hot_enc = []  # one-hot result
for i in integer_enc:
    enc = [0 for _ in range(num_classes)]
    enc[i] = 1
    one_hot_enc.append(enc)
one_hot_enc = np.asarray(one_hot_enc, dtype=float)  # dtype=float: same as to_categorical function

# decoding
int_to_num = dict((v, k) for k, v in num_to_int.items())
print(int_to_num[np.argmax(one_hot_enc[0])])
original = [int_to_num[np.argmax(one_hot_enc[i])] for i in range(len(one_hot_enc))]
