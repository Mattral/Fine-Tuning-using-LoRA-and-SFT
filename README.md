# Fine-Tuning-using-LoRA-and-SFT

#Intro

Lets dive deeper into the mechanics of LoRA, a powerful method for optimizing the fine-tuning process of Large Language Models, its practical uses in various fine-tuning tasks, and the open-source resources that simplify its implementation. We will also introduce QLoRA, a highly efficient version of LoRA. By the end, you will have an in-depth understanding of how LoRA and QLoRA can enhance the efficiency and accessibility of fine-tuning LLMs. 

## The Functioning of LoRA in Fine-tuning LLMs

[LoRA](https://arxiv.org/abs/2106.09685), or Low-Rank Adaptation, is a method developed by Microsoft researchers to optimize the fine-tuning of Large Language Models. This technique tackles the issues related to the fine-tuning process, such as extensive memory demands and computational inefficiency. LoRA introduces a compact set of parameters, referred to as low-rank matrices, to store the necessary changes in the model instead of altering all parameters. 

Here are the key features of how LoRA operates:

- Maintaining Pretrained Weights: LoRA adopts a unique strategy by preserving the pretrained weights of the model. This approach reduces the risk of catastrophic forgetting, ensuring the model maintains the valuable knowledge it gained during pretraining.
- Efficient Rank-Decomposition: LoRA incorporates rank-decomposition weight matrices, known as update matrices, to the existing weights. These update matrices have significantly fewer parameters than the original model, making them highly memory-efficient. By training only these newly added weights, LoRA achieves a faster training process with reduced memory demands. These LoRA matrices are typically integrated into the attention layers of the original model.

By using the low-rank decomposition approach, the memory demands for training large language models are significantly reduced. This allows running fine-tuning tasks on consumer-grade GPUs, making the benefits of LoRA available to a broader range of researchers and developers.

## Open-source Resources for LoRA
The following libraries offer a mix of tools that enhance the efficiency of fine-tuning large language models. They provide optimizations, compatibility with different data types, resource efficiency, and user-friendly interfaces that accommodate various tasks and hardware configurations.

- PEFT [Library](https://github.com/huggingface/peft): Parameter-efficient fine-tuning (PEFT) methods facilitate efficient adaptation of pre-trained language models to various downstream applications without fine-tuning all the model's parameters. By fine-tuning only a portion of the model's parameters, PEFT methods like LoRA, Prefix Tuning, and P-Tuning, including QLoRA, significantly reduce computational and storage costs.
- [Lit-GPT](https://github.com/Lightning-AI/lit-gpt): Lit-GPT from LightningAI is an open-source resource designed to simplify the fine-tuning process, making it easier to apply LoRA's techniques without manually altering the core model architecture. Models available for this purpose include Vicuna, Pythia, and Falcon. Specific configurations can be applied to different weight matrices, and precision settings can be adjusted to manage memory consumption.

In this repo we will mainly use PEFT

## QLoRA: An Efficient Variant of LoRA
[QLoRA](https://arxiv.org/abs/2305.14314), or Quantized Low-Rank Adaptation, is a popular variant of LoRA that makes fine-tuning large language models even more efficient. QLoRA introduces several innovations to save memory without sacrificing performance.

The technique involves backpropagating gradients through a frozen, 4-bit quantized pretrained language model into Low-Rank Adapters. This approach significantly reduces memory usage, enabling the fine-tuning of even larger models on consumer-grade GPUs. For instance, QLoRA can fine-tune a 65 billion parameter model on a single 48GB GPU while preserving full 16-bit fine-tuning task performance.

QLoRA uses a new data type known as 4-bit NormalFloat (NF4), which is optimal for normally distributed weights. It also employs double quantization to reduce the average memory footprint by quantizing the quantization constants and paged optimizers to manage memory spikes.

The [Guanaco](https://huggingface.co/TheBloke/guanaco-65B-GPTQ) models, which use QLoRA fine-tuning, have demonstrated state-of-the-art performance, even when using smaller models than the previous benchmarks. This shows the power of QLoRA tuning, making it a popular choice for those seeking to democratize the use of large transformer models.

The practical implementation of QLoRA for fine-tuning LLMs is very accessible, thanks to open-source libraries and tools. For instance, the [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) library offers functionalities for 4-bit quantization. We’ll later see a code example showing how to use QLoRA with PEFT.


# Lets Begin!

The fine-tuning process has consistently proven to be a practical approach for enhancing the model's capabilities in new domains. Therefore, it is a valuable approach to adapt large language models while using a reasonable amount of resources.

As mentioned earlier, the fine-tuning process builds upon the model's existing general knowledge, which means it doesn't need to learn everything from scratch. Consequently, it can grasp patterns from a relatively small number of samples and undergo a relatively short training process.

In this lesson, we’ll see how to do SFT on an LLM using LoRA. We’ll use the dataset from the "[LIMA: Less Is More for Alignment](https://arxiv.org/pdf/2305.11206.pdf)" paper. According to their argument, a high-quality, hand-picked, small dataset with a thousand samples can replace the RLHF process, effectively enabling the model to be instructively fine-tuned. Their approach yielded competitive results compared to other language models, showcasing a more efficient fine-tuning process. However, it might not exhibit the same level of accuracy in domain-specific tasks, and it requires hand-picked data points.

The [TRL library](https://github.com/huggingface/trl) has some classes for Supervised Fine-Tuning (SFT), making it accessible and straightforward. The classes permit the integration of LoRA configurations, facilitating its seamless adoption. It is worth highlighting that this process also serves as the first step for Reinforcement Learning with Human Feedback (RLHF), a topic we will explore in detail later in the course.

## Spinning Up a Virtual Machine for Finetuning on GCP Compute Engine
Cloud GPUs availability today is very scarse as they are used a lot for several deep learning applications. Few people know that CPUs can be actually used to finetune LLMs through various optimizations and that’s what we’ll be doing in these lessons when doing SFT.

Let’s login to our Google Cloud Platform account and create a [Compute Engine](https://cloud.google.com/compute) instance (see the “Course Introduction” lesson for instructions). You can choose between different [machine types](https://cloud.google.com/compute/docs/cpu-platforms). Here, we trained the model on the latest CPU generation from 4th Generation Intel® Xeon® Scalable Processors (formerly known as Intel® Sapphire Rapids). This architecture features an integrated accelerator designed to enhance the performance of training deep learning models. Intel® Advanced Matrix Extension (AMX) empowers the training of models with BF16 precision during the training process, allowing for half-precision training on the latest Xeon® Scalable processors. Additionally, it introduces an INT8 data type for the inference process, leading to a substantial acceleration in processing speed. Reports suggest a tenfold increase in performance when utilizing PyTorch for both training and inference processes.

Follow the instructions in the course introduction to spin up a VM with Compute Engine with high-end Intel® CPUs. Once you have your virtual machine up, you can SSH into it.

Incorporating CPUs for fine-tuning or inference processes presents an excellent choice, as renting alternate hardware is considerably less cost-effective. It worth mentioning that a minimum of 32GB of RAM is necessary to load the model and facilitate the experiment's training process. If there is an out-of-memory error, reduce arguments such as batch_size or seq_length.

[warning] Beware of costs when you spin up virtual machines. The total cost will depend on the machine type and the up time of the machine. Always remember to monitor your costs in the billing section of GCP and to spin off your virtual machines when you don’t use them.

## Load the Dataset
The quality of a model is directly tied to the quality of the data it is trained on! The best approach is to begin the process with a dataset. Whether it is an open-source dataset or a custom one manually, planning and considering the dataset in advance is essential. In this lesson, we will utilize the dataset released with the LIMA research. It is publicly available with a non-commercial use license.

The powerful feature of [Deep Lake](https://www.deeplake.ai/) format enables seamless streaming of the datasets. There is no need to download and load the dataset into memory. The hub provides diverse datasets, including the LIMA dataset presented in the "LIMA: Less Is More for Alignment" paper.  The Deep Lake Web UI not only aids in dataset exploration but also facilitates dataset visualization using the embeddings field, taking care of clustering the dataset and map it in 3D space. (We used Cohere embedding API to generate in this example) The enlarged image below illustrates one such cluster where data points in Portuguese language related to coding are positioned closely to each other. Note that Deep Lake Visualization Engine offers you the ability to pick the clustering algorithm.

The code below will create a loader object for the training and test sets.

```

import deeplake

# Connect to the training and testing datasets
ds = deeplake.load('hub://genai360/GAIR-lima-train-set')
ds_test = deeplake.load('hub://genai360/GAIR-lima-test-set')

print(ds)
```
```
Dataset(path='hub://genai360/GAIR-lima-train-set', read_only=True, tensors=['answer', 'question', 'source'])
```

We can then utilize the ConstantLengthDataset class to bundle a number of smaller samples together, enhancing the efficiency of the training process. Furthermore, it also handles dataset formatting by accepting a template function and tokenizing the texts.

To begin, we load the pre-trained tokenizer object for the [Open Pre-trained Transformer (OPT)](https://arxiv.org/abs/2205.01068) model using the Transformers library. We will load the model later. We are using OPT for convenience because it’s an open model with a relatively “small” amount of parameters. The same code in this lesson can be run in another model too, for example, using meta-llama/Llama-2-7b-chat-hf for [LLaMa 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)https://huggingface.co/meta-llama/Llama-2-7b-chat-hf.
```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
```

Moreover, we need to define the formatting function called prepare_sample_text, which takes a row of data in Deep Lake format as input and formats it to begin with a question followed by the answer that is separated by two newlines. This formatting aids the model in learning the template and understanding that if a prompt starts with the question keyword, the most likely response would be to complete it with an answer.

```
def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question'].text()}\n\nAnswer: {example['answer'].text()}"

    return text
```

Now, with all the components in place, we can initialize the dataset, which can be fed to the model for fine-tuning. We call the ConstantLengthDataset class using the combination of a tokenizer, deep lake dataset object, and formatting function. The additional arguments, such as infinite=True ensure that the iterator will restart when all data points have been used, but there are still training steps remaining. Alongside seq_length, which determines the maximum sequence length, it must be completed according to the model's configuration. In this scenario, it is possible to raise it to 2048, although we opted for a smaller value to manage memory usage better. Select a higher number if the dataset primarily comprises shorter texts.

```
from trl.trainer import ConstantLengthDataset

train_dataset = ConstantLengthDataset(
    tokenizer,
    ds,
    formatting_func=prepare_sample_text,
    infinite=True,
    seq_length=1024
)

eval_dataset = ConstantLengthDataset(
    tokenizer,
    ds_test,
    formatting_func=prepare_sample_text,
    seq_length=1024
)

# Show one sample from train set
iterator = iter(train_dataset)
sample = next(iterator)
print(sample)
```

Output
```
{'input_ids': tensor([    2, 45641,    35,  ..., 48443,  2517,   742]), 'labels': tensor([    2, 45641,    35,  ..., 48443,  2517,   742])}
```

As evidenced by the output above, the ConstantLengthDataset class takes care of all the necessary steps to prepare our dataset.

[!NOTE] If you use the iterator to print a sample from the dataset, remember to execute the following code to reset the iterator pointer. `train_dataset.start_iteration = 0`

## Initialize the Model and Trainer
As mentioned previously, we will be using the [OPT model](https://huggingface.co/facebook/opt-1.3b) with 1.3 billion parameters in this lesson, which has the facebook/opt-1.3b model id on the Hugging Face Hub.

The LoRA approach is employed for fine-tuning, which involves introducing new parameters to the network while keeping the base model unchanged during the tuning process. This approach has proven to be highly efficient, enabling fine-tuning of the model by training less than 1% of the total parameters. (For more details, refer to the following post.)

With the TRL library, we can seamlessly add additional parameters to the model by defining a number of configurations. The variable r represents the dimension of matrices, where lower values lead to fewer trainable parameters. lora_alpha serves as the scaling factor, while bias determines which bias parameters the model should train, with options of none, all, and lora_only. The remaining parameters are self-explanatory.

