import time
import random
import matplotlib.pyplot as plt
import pandas as pd
from vllm import LLM, SamplingParams
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field 
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import JsonOutputParser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
OLLAMA_MODEL = "llama3.1:8b-instruct-q8_0"
VLLM_MODEL_PATH = "/mnt/DATA7/MODEL/vllm_model/gguf/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"  # Your vLLM model path
NUM_PROMPTS = 20  # Number of prompts to generate for testing
MAX_TOKENS = 512  # Maximum number of tokens for generation
BATCH_SIZES = [1, 2, 4, 8, 16]  # Batch sizes to test for vLLM

# --- Prompt Generation ---
def generate_math_prompts(num_prompts=NUM_PROMPTS):
    """Generates a set of math-related prompts."""
    prompts = []
    for _ in range(num_prompts):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(["+", "-", "*"])
        prompts.append(f"What is {a} {op} {b}?")
    return prompts

# --- Ollama Benchmarking ---
def benchmark_ollama(prompts, model_name=OLLAMA_MODEL, max_tokens=MAX_TOKENS):
    """Benchmarks Ollama's response time for a set of prompts."""
    model = ChatOllama(
        model=model_name,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0
    )
    system_prompt = "You are a helpful math assistant."
    human_prompt = "{input}"
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    chain = prompt_template | model

    times = []
    results = []
    for prompt in prompts:
        start_time = time.time()
        t = chain.invoke({"input": prompt})
        print(t.content)
        results.append(t)
        end_time = time.time()
        times.append(end_time - start_time)
    # print(results)
    return times

# --- Single vLLM Benchmarking ---
def benchmark_vllm_single(prompts, model, max_tokens=MAX_TOKENS):
    """Benchmarks single vLLM inference (non-batched)."""
    
    llm = model
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_tokens)

    times = []
    result = []
    conversation = [{"role": "system", "content": "You are a helpful math assistant"}]
    for prompt in prompts:
        conversation.append({"role": "user", "content": prompt})
        start_time = time.time()
        t = llm.chat(conversation, sampling_params)
        result.append(t)
        end_time = time.time()
        times.append(end_time - start_time)
        conversation.pop()  # Remove the user prompt to maintain a single-turn conversation
    return times

# --- Batched vLLM Benchmarking ---
def benchmark_vllm_batched(prompts, batch_sizes, model, max_tokens=MAX_TOKENS):
    """Benchmarks vLLM with different batch sizes."""
    llm = model
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_tokens)
    cache_results = []
    results = {}
    for batch_size in batch_sizes:
        logging.info(f"Benchmarking vLLM with batch size: {batch_size}")
        times = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            # Prepare conversations for each prompt in the batch
            conversations = [[{"role": "system", "content": "You are a helpful math assistant"},
                              {"role": "user", "content": p}] for p in batch]

            start_time = time.time()
            temp = llm.chat(conversations, sampling_params)  # Use chat_batch for batched inference
            # Print outputs
            end_time = time.time()
            for i, o in zip(prompts, temp):
                print(f"Prompt: {i}")
                print(f"Output: {o.outputs[0].text}")
                print("-"*30)
                cache_results.append(o.outputs[0].text)


                
                

            times.extend([(end_time - start_time) / len(batch)] * len(batch))
        results[batch_size] = times
    return results

# --- Main Execution and Plotting ---
def main():
    """Runs benchmarks and generates plots."""
    prompts = generate_math_prompts()
    print(prompts)

    # Run benchmarks
    logging.info("Running Ollama benchmark...")
    ollama_times = benchmark_ollama(prompts)

    con = input("Continue? (y/n): ")
    llm = LLM(model=VLLM_MODEL_PATH,max_model_len=30000,gpu_memory_utilization=1.0)


    logging.info("Running single vLLM benchmark...")
    vllm_single_times = benchmark_vllm_single(prompts,llm)
    con = input("Continue? (y/n): ")

    logging.info("Running batched vLLM benchmark...")
    vllm_batched_times = benchmark_vllm_batched(prompts, BATCH_SIZES,llm)
    con = input("Continue? (y/n): ")

    # Create a DataFrame for results
    df = pd.DataFrame({
        "Prompt": prompts,
        "Ollama": ollama_times,
        "vLLM Single": vllm_single_times,
    })
    for batch_size, times in vllm_batched_times.items():
        df[f"vLLM Batch {batch_size}"] = times

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df["Prompt"], df["Ollama"], label="Ollama")
    plt.plot(df["Prompt"], df["vLLM Single"], label="vLLM Single")
    for batch_size in BATCH_SIZES:
        plt.plot(df["Prompt"], df[f"vLLM Batch {batch_size}"], label=f"vLLM Batch {batch_size}")

    plt.xlabel("Prompt")
    plt.ylabel("Completion Time (seconds)")
    plt.title("LLM Benchmark: Ollama vs. vLLM (Single & Batched)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("llm_benchmark.png")
    plt.show()

    print(df)

if __name__ == "__main__":
    main()