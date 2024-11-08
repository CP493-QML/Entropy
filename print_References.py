# Tested with Ubuntu 22.04 WSL2
# pip install openai
# Usage: python print_References.py

# https://platform.openai.com/docs/api-reference/chat/create?lang=python
# seed and system_fingerprint:  https://cookbook.openai.com/examples/reproducible_outputs_with_the_seed_parameter


import os
import time
import math

from openai import OpenAI

# Get the API key from the environment variable
key = os.getenv('OPENAI_API_KEY')

# or provide the  API key string explicitly, which is a not a safe practice
# key = 'sk-ehe-bla-bla-bla'

client = OpenAI(api_key= key)

# If the key is not found, raise an error
if key is None:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key= key)

deployment_model= "gpt-4o"


system_prompt= """
You are a university professor specializing in Quantum Information Theory.  You have written a journal paper entitled "Quantum Data Compression". Now create a References section which includes 10 journal publications suitable for the paper.
"""

# user prompt is not used in this example. Typically, user prompt would be used for some input data if any. 
user_prompt= ""


messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]



def chat_with_gpt(conversation):
    while True:
        try:
            # Make the GPT-4 API call
            print('Initiating API request. Processing may take some time, please wait...')
            completion = client.chat.completions.create(
                model=deployment_model,
                messages=conversation,
                seed=12345,
                temperature=0, # zero for deterministic output
                logprobs=True,  # Request logprobs for each token in the output text
                max_tokens=2000
            )
         
            print('API request Successful')
            
            time.sleep(1)  # Optional delay before continuing
            return completion
            
        except Exception as e:
            print(f'Error occurred: {e}')
            print('Retrying API request in 5 seconds...')
            time.sleep(5)  # Wait before retrying



# Get the assistant's response
resp = chat_with_gpt(messages)
response = resp.choices[0]

assistant_message = response.message.content.strip()

# Print the assistant's response
print(f"{assistant_message}")


# Extract logprobs if available
logprobs_data = response.logprobs.content if response.logprobs else None

# Initialize total entropy
total_entropy = 0
logprob_sum =0

log2_constant = math.log(2)  # Constant to convert nats to bits



# these output tokens are specifically excluded  from the calculation  of the entropy
excluded_tokens = ['```', 'python', ' ', '', '\n', 'latex', 'json', 'tag','**', 'References']


# Exclude tokens that are in excluded_tokens or consist entirely of whitespace
logprobs_data = [token_logprob_data for token_logprob_data in logprobs_data 
                if token_logprob_data.token not in excluded_tokens 
                and token_logprob_data.token.strip() != '']

num_tokens = len(logprobs_data)


token_print_limit = 3 

if token_print_limit > num_tokens:
    token_print_limit = num_tokens


# Get the first three  and last three tokens 
first_tokens = logprobs_data[:token_print_limit]
last_tokens = logprobs_data[-token_print_limit:] if len(logprobs_data) > token_print_limit else []


# Calculate Shannon entropy in bits based on the logprobs of each token
for token_logprob_data in logprobs_data:
    logprob = token_logprob_data.logprob  # Extract logprob of the token
    prob = math.exp(logprob)  # Convert logprob to probability
    entropy_in_nats = -prob * logprob  # Shannon entropy in nats
    entropy_in_bits = entropy_in_nats / log2_constant  # Convert nats to bits
    total_entropy += entropy_in_bits  # Accumulate total entropy in bits
    logprob_sum += logprob
    

    # Now print probabilities for the first 3 tokens
print(f"\nProbabilities of the first {token_print_limit} tokens (printed for debugging and testing only):")
for token_logprob_data in first_tokens:
    logprob = token_logprob_data.logprob  # Extract logprob of the token
    prob = math.exp(logprob)  # Convert logprob to probability
    print(f"'{token_logprob_data.token}'  {prob:.6f}")

# Only print last 3 tokens if they are not the same as the first 3
if len(last_tokens) > 0:
    print(f"\nProbabilities of the last {token_print_limit} tokens (printed for debugging and testing only):")
    for token_logprob_data in last_tokens:
        logprob = token_logprob_data.logprob  # Extract logprob of the token
        prob = math.exp(logprob)  # Convert logprob to probability
        print(f"'{token_logprob_data.token}' {prob:.6f}")


# Normalize entropy by the number of tokens to get average entropy per token
average_entropy = total_entropy / num_tokens if num_tokens > 0 else 0

# Output the total and normalized entropy in bits
print(f"\nNumber of output tokens: {num_tokens}; Total Entropy: {total_entropy:.2f} bits; Normalized Entropy: {average_entropy:.3f} bits; Logprob_sum={logprob_sum:.6f};  System_fingerprint= {resp.system_fingerprint}")