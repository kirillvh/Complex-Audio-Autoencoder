# Complex-Audio-Autoencoder
A Vocoder Inspired by Vocos and APNet but with Complex Numbers and other tricks.

## Problem
In my reserach on vocoders in 2024, I was puzzled by the APNet2 handles Magnitude and Phase via two seperate real valued networks.
Since the end goal was to input a complex number sequence into ISTFT to generate the audio waveform, I thought it would be better to use unified Magnitude & Phase network that was based on complex numbers.
But it turned out that complex valued networks were still a topic of research and support for it was just starting to become stable, so I understood why previous researchers were forced to split it into seperate real valued networks and realized that due to the recent maturation of the technology, now would be a good time to try and implement such a thing.
So first I tried to reproduce the APNet results to get a baseline:

[Listen to the APNet test Audio](https://github.com/user-attachments/files/28866138/APNet-test.wav)


I only got it to about 60k steps, but after listening to this audio you might agree with me that it would need way more to produce inteligible results and I didn't have the compute/time for it so I jumped straight into the implementation of my own method.

## Conventional Approach
It turns out that [later research](https://arxiv.org/pdf/2509.18806) in 2025 had confirmed my suspicions about the split real network approach of APNet:
<img width="488" height="296" alt="image" src="https://github.com/user-attachments/assets/b92f54bc-3fbb-46bd-8177-3c9fcbe2229d" />

It turns out that the split approach could still work well enough when reporoducing a single speaker (such as on the LJSpeech dataset), but the vocoder broke down when trying to generalize to many speakers (over 2000 speakers in the LibriTTS dataset).

Interestingly Vocos didn't suffer from this problem, the researchers explained that this is because it used a unified network before splitting the data into Magnitude and Phase, and that the split network approach had likely caused a mismatch between these two components.
<img width="737" height="337" alt="image" src="https://github.com/user-attachments/assets/c3617b61-4f38-4e60-87cb-3a921e690f26" />

Several modifications were proposed to enforce aligment between the real and imaginary outputs of APNet while Vocos almost produces the right result with a much simpler approach. So I thought the Vocos approach is better but might be further improved with a complex valued network.

## Proposed Approach

<img width="433" height="755" alt="image" src="https://github.com/user-attachments/assets/bb9ab340-14d7-4787-8d0f-7ceb0b7e4ee0" />



Basically we just need a decoder that converts an internal representation such as Phoneme's when working with a TTS such as [StyleTTS2](https://github.com/yl4579/styletts2) or the internal latent space of a multimodal LLM, into tokens (which enables Generative AI) that can be detokenized to produce an input for the decoder.
But it should be a complex valued network so that it natively produces aligned real & imaginary numbers to drive the ISTFT waveform generator. To train this we will also need the encoder part but it is not necessary during inference, and depending on the end-to-end connection, the tokenizer might not even be necessary but in this example work the tokenizer will serve as the main bottleneck that forces the encoder & decoder to learn compression & decompression.

## Implementation
<img width="756" height="574" alt="image" src="https://github.com/user-attachments/assets/8492ec1d-70ad-4b81-8dcb-b9eee342e652" />

Digging into the research, I found several papers on complex valued neural networks: 
[DEEP COMPLEX NETWORKS](https://arxiv.org/pdf/1705.09792v1)

[Phase-driven Domain Generalizable Learning for Nonstationary Time Series](https://arxiv.org/abs/2402.05960)

[Analysis of Deep Complex-Valued Convolutional Neural Networks for MRI Reconstruction]([https://arxiv.org/abs/2402.05960](https://www.researchgate.net/publication/340475482_Analysis_of_Deep_Complex-Valued_Convolutional_Neural_Networks_for_MRI_Reconstruction)

It seemed that PyTorch already handles most of the processing so only the activation functionss would need to be changed as well as manually handling real and imaginary components in certain situations.

For the main Encoder & Decoder blocks, I have emperically found the following network to produce good enough results within my limited compute (A single Nvidia P40, which died somewhere in 2025 :[ )
<img width="851" height="99" alt="image" src="https://github.com/user-attachments/assets/dabeb619-8f4a-469a-a4da-c1e80e831046" />


Here the FeatureX() functions are convolutional feature extractors and generally it follows a structure similar to ConvNextV2, except that there are two inputs, one for a latent backbone and another to parse in the original version of the input.
The idea here is to setup the initilization in such a way that it begins as a pass through network and basically works from the start. 
However the tokenizer is a residual tokenizer, so the number of tokens transmited per latent can be adjusted at runtime to balance quality vs effort. So we can apply Dropout to randomly shorten the amount of tokens, therfore the network is encouraged to try and compress/decompress everything into as few tokens as possible.
In this way, the more we train, the more compression we can gain but it should always work if we simple allow more tokens to be used. I thought this was a good compromise for my compute budget.

Unfortunatley I had some PC trouble while training the network and the main training file (the "glue") was lost but I could still recover the modules which is arguably the most important code and it is uploaded inside this repository. However as a result I had lost interest in the project back in 2024 and moved on but now I want to look into it again.
The problem is that I don't have the exact quantizer settings I used, but there is one related file with a likely candidate:
    quantizer = ResidualLFQ(
    dim = 512,
    codebook_size = 256,
    num_quantizers = 32, # 2=>1.5Kbit/sec[max], 32=>24Kbit/sec[max]
    quantize_dropout = True,
    quantize_dropout_cutoff_index = 2,
    quantize_dropout_multiple_of = 2,
    ).to(device)

Here ResidualLFQ is from Lucid Rain's [Vector-Quantization repository](https://github.com/lucidrains/vector-quantize-pytorch). The exact values used to produce the results below were probably a bit different, but I think I tried to target similar minimum & maximum bit rates.

There are also a few extra tricks I used to improve the result such as an [alias-free design](https://arxiv.org/abs/2106.12423) which basically boils down to upsamling before the activation function, then low pass filtering and downsampling after if to prevent the optimizer from learning some unbeneficial details related to activation function non-linearities (although I should mention that these alias free methods also blew up my compute needs and forced me to use smaller batch sizes, so im not sure about the overall benefit).
Also, the full setup fetured a Generative-Adverserial network but the implmentation was using a [SAN](https://arxiv.org/abs/2301.12811) which is an enhanced verion of GAN.

## Result
[Listen to the Audio Result](https://github.com/user-attachments/files/28869124/Result.wav)

<img width="1000" height="425" alt="image" src="https://github.com/user-attachments/assets/147379fd-7a06-4d28-bc21-579bdb4bf460" />

Well I admit to cherry picking the best sounding result, it still has a little noisy and as you can see the generated spectogram is a little smoothed, but I think it sounds like a working proof-of-concept, and it is important to point out that this was obtained while training not only on LibriTTS but on a broad multilingual dataset including English, Chinese, Ukrainian, Dutch and Persian, therfore number of speakers is large which was shown to be a problem for the conventional APNet (and also probably hints that my network was undertrained).

## Conclusion
With recent versions of PyTorch and community provided components such as activation functions, it has become possible to train complex valued Vocoders which can theoretically produce better inputs to ISTFT to improve its output.
Furthermore, an complex valued Vocoder architecture was presented which produces promising quickly and allows for a gentle raising of the compute difficulty.

