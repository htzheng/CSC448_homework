In homework2, I implemented a perceptron decoder using PyTorch. The implementation is based on the Viterbi Decoding Algorithm. However, there are some differences:

1. I leverage the start token to simplify the initialization of the dynamic programming when i=0.
2. I define an out-of-vocabulary word so that the model can handle the out-of-vocabulary exception
3. I define the transmission and emission matrix and leverage PyTorch matrix operation to improve the speed (Line 86). By computing the inner loop (T') with matrix operation, the inference speed becomes pretty fast.

The accuracy on the test file is 94.45%, which matches homework1. The running time is 2:45 mins. However, the inference speed can be further improved by computing the second loop with matrix operations.