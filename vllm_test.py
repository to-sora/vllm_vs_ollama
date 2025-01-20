from vllm import LLM, SamplingParams

# Initialize the conversation
conversation = [
    {"role": "system", "content": "You are a helpful assistant"}
]
# Create a sampling params object
sampling_params = SamplingParams(temperature=0, top_p=0.95,max_tokens =1000)
# Define the model path
m_p = "/mnt/DATA7/MODEL/vllm_model/gguf/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
m_n = "/mnt/DATA7/MODEL/GGUF_model/llama-13b.Q4_K_M.gguf"
# Create an LLM
llm = LLM(model=m_n,
         # max_model_len=30000,
         # gpu_memory_utilization=1.0,
      )

print("Interactive Chat Initialized. Type 'exit' to quit.\n")

# Start interactive chat loop
while True:
    # Get user input
   user_input = input("You: ")
   conversation.append({"role": "user", "content": user_input})
   outputs = llm.chat(conversation, sampling_params)
   print(type(outputs))
   print(f"Assistant: {outputs[0].outputs[0].text}\n")
   conversation.append({"role": "assistant", "content": outputs[0].outputs[0].text})
