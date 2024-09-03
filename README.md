Royal Holloway MSc Project
Language Model Training Poisoning

Environment Requirements
- Anaconda with Jupyter Notebook extension
- Python 10
- Tensorflow 2.10.1
- Tensorboard 2.10.1
- CUDA 11.2 & cuBDD 8.1.0
- Pyarrow 16.1.0
- Numpy 1.26.4
- Datasets 2.3.0
- Scikit learn 1.0.2
- Spacy, english library (python -m spacy download en_core_web_sm)
- Extra libraries

Program execution sequence. 
1. dataSetup
2. modelClean
3. modelStart
4. modelPoisoned
5. modelCombined
6. modelVerification several times for different models

Program descriptions:
environmentTest.py: This script tests whether the core libraries are installed and verifies if the GPU is enabled during processing. Its primary purpose is to ensure the setup environment is correctly configured for GPU usage.

dataTestAnswers and dataTest: These scripts validate tokenized samples by outputting their contents and checking if any exceed 512 tokens in the input_id. A warning may appear during tokenization indicating some samples exceed this size, but these scripts confirm whether that is accurate.
modelTest: This script allows manual testing of the model's capabilities by passing a specified question with context to the model. 

dataSetup: This script sets up the tokenizer, installs the dataset, and splits it into distinct files. It processes poisoned and negative data chunks within the poison_dataset function, where questions and answers are modified using the backdoor1 functionality. After this, the data is tokenized into embeddings.

backdoor1: This function is consistent across both datasets. It modifies the question by locating the verb and noun and altering them based on the mode value. If a sample does not fit the original mode, the mode is adjusted. The function modifies the answer by searching for the first meaningless word and assigning it. Both functions use the spacy library for word search.

tokenization1: The tokenization process involves several key functions. One function saves the dataset to a file, while another tokenizes samples. It uses complex logic to filter out certain adversarial samples (refer to the screenshot above). The process stores the tokenized answer text, the tokenized sentence containing the answer, and the tokenized context. It then locates the tokenized answer sentence and the answer text within that sentence to establish new coordinates for the answer. Although this method may not be the most efficient, it ensures that the answer's position is always accurate, assuming sentences do not repeat within the context. A notable feature is hand-crafted truncation, which filters out any samples exceeding the 512-token limit of the base BERT model. Although a warning may appear during tokenization, it is incorrect. 

trainingLoop: This script trains the model. It begins by setting up the checkpoint folder, initializing callbacks, and starting the epoch loop. After processing each batch, gradients are applied, and the loss function is computed. Every 50th batch, the average loss is calculated, and if the processed batch's loss exceeds the average by 1.75, gradient clipping is applied. Every 250th batch, an early stopping mechanism checks whether the last 5 average losses have not varied by more than 0.05. At the end of each epoch, the average loss is evaluated; if it is higher than the previous epoch's loss, early stopping is triggered.

Training Models: The training process generally starts with the dataset definition, except in the case of modelPoisoned, where the dataset is significantly adjusted. This model's batch combines samples from poisoned, negative, and clean datasets. The next steps involve optimizing GPU memory, storing the base BERT model, and customizing it using the model class. Following this, TensorBoard is set up, and the dataset is converted to TensorFlow format using tfConvert. Once the model architecture is compiled, training begins. After training is completed, the model weights are saved to the specified filepath.

Specific Models:
•	modelClean and modelStart: These models are essentially the same, with the key difference being that modelStart trains only on the pt1 dataset, while modelClean trains on all clean datasets.
•	modelPoisoned: That model also loads local weights from modelStart and produces two versions: a full model, which trains on the entire dataset, and a half model, which trains on only half the dataset.
•	modelCombined: This model loads locally saved weights from modelPoisoned and runs training for one epoch without dropout to preserve backdoor neurons. Uses pt3 dataset

modelVerification: Program checks the performance of given model using various metrics. 
softF1: Compares similarity of two sets of embeddings