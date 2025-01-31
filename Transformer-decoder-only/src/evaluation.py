import torch
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
import json
import evaluate

from model import Model
from generation import TextGenerator

with open('config.json', 'r') as f:
    config = json.load(f)
class ModelConfig:
    """Configuration for the model"""
    CONTEXT_LENGTH: int = config['model']['context_length']     # Number of context frames (default: 16)
    D_MODEL: int = config['model']['d_model']                   # Dimension of the model (must be divisible by NUM_HEADS)
    D_FF: int = config['model']['d_ff']                         # Dimension of the feed forward network (must be equal to D_MODEL*4)
    NUM_BLOCKS: int = config['model']['num_blocks']             # Number of transformer blocks in the model (default: 8)
    NUM_HEADS: int = config['model']['num_heads']               # Number of heads in multi-head attention (must divide D_MODEL)
    DROP_OUT: float = config['model']['dropout']                # Drop out rate for regularization (default: 0.1)
    MODEL_PATH: str = config['train']['model_path']             # Path to save the model
    DATASET_PATH: str = config['train']['dataset_path']         # Path to the dataset  
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data(dataset_path: str):
    """
    Load the dataset

    Args:
        dataset_path (str): Path to the dataset
        split (str): Split of the dataset

    Returns:
        prompts (list): List of prompts
    """
    dataset = load_dataset(dataset_path)
    return dataset

def evaluate_texts(generated_texts, reference_texts):
    """
    Evaluate the generated texts
    
    Args:
        generated_texts (list): List of generated texts
        reference_texts (list): List of reference texts

    Returns:
        bleu_score (dict): BLEU score
        rouge_score (dict): ROGUE score
        perplexity_score (dict): Perplexity score
    """

    # BLEU score
    bleu = evaluate.load('bleu')
    bleu_score = bleu.compute(predictions=generated_texts, references=reference_texts)

    # ROGUE score
    rouge = evaluate.load('rouge')
    rouge_score = rouge.compute(predictions=generated_texts, references=reference_texts)

    return {
        'bleu': bleu_score, 
        'rouge': rouge_score, 
    }

def main():

    # test dataset: vesteinn/babylm    
    model = Model().to(ModelConfig.DEVICE)
    model.load_state_dict(torch.load(ModelConfig.MODEL_PATH))
    model.eval()

    # load the dataset
    datasets = load_data(ModelConfig.DATASET_PATH)
    test_datasets = []
    for i in range(0, 10):
        test_datasets.append(datasets['train'][i]['text'])

    # print(f"Loaded {test_datasets} test datasets")

    # load text generator
    generator = TextGenerator(model_path=ModelConfig.MODEL_PATH)

    # prepare the texts=pairs(generated_texts, reference_texts)
    generated_texts = []
    reference_texts = []
    for i in range(0, 10):
        prompt = test_datasets[i]
        generated_text = generator.generate_text(prompt, max_tokens=ModelConfig.CONTEXT_LENGTH, temperature=0.7, top_k=50)
        generated_texts.append(generated_text)
        reference_texts.append(test_datasets[i])

    # print all pairs
    for i in range(len(generated_texts)):
        print(f"Pair {i+1}")
        print(f"Generated text: {generated_texts[i]}")
        print(f"Reference text: {reference_texts[i]}")

    # evaluate the model
    scores = evaluate_texts(generated_texts, reference_texts)
    print(f"BLEU score: {scores['bleu']}")
    print(f"ROUGE score: {scores['rouge']}")

if __name__ == "__main__":
    main()