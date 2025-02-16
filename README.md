# NLP_Tutorials
A set of tutorials on the topic of Natural Language Processing (NLP) to help me get familiar with the use of DeepLearning for NLP applications

## rnn_translation

- RNN based Encoder-Decoder network to translate from French to English and German to English
- Using a small datasets from [ManyThings.org](https://www.manythings.org/anki/) for this project.
- Also implemented an Attention based Decoder based on the idea from [Badhanau et al](./literature/2015_Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.pdf). 
- Refer to the notebooks [`eng_to_deu_translation.ipynb`](./rnn_translation/eng_to_deu_translation.ipynb)
    and [`eng_to_fra_translation.ipynb`](./rnn_translation/eng_to_fra_translation.ipynb)

## transformer_translation

- In this project, I attempted to replicate the original Transformer network from scratch. 
- The Transformer Network was introduced in 2017 by Vaswani et al in their seminal paper [`Attention is all you need`](./literature/2017_Attention_is_all_you_need.pdf)
- In implementing the network from scratch I relied heavily on the Tutorial by [Sagar Vinodababu](https://github.com/sgrvinod) 
- He has a brilliant explanation of the Transformer network and its inner workings on this [Github page](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers)
- The dataset I am using here is the [WMT-14 English to German dataset](https://metatext.io/datasets/wmt-14-english-german). The same dataset was used to train the model in the original paper. 
- Refer to the notebooks [`eng_to_deu_wmt14.ipynb`](./rnn_translation/eng_to_deu_wmt14.ipynb)
