from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True
)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, TextStreamer , TextIteratorStreamer

model = AutoModelForCausalLM.from_pretrained("Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1", torch_dtype=torch.bfloat16 , device_map ="auto")
tokenizer = AutoTokenizer.from_pretrained("Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1", trust_remote_code=True)

import os
import json

# Define the directory path
data_dir = "20calls_hindi_translated_transcripts"
json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

import json

data = []
for filename in json_files:
  # Create the filepath
  filepath = os.path.join(data_dir, filename)
  with open(filepath, "r") as file:
    try:
      # Read the JSON data
      file_data = json.load(file)

      # Add filename and data to the list
      data.append({"filename": filename, "data": file_data})
    except json.JSONDecodeError as e:
      print(f"Error reading '{filename}': {e}")

# Now 'data' contains a list of dictionaries, each with 'filename' and 'data' keys

one_shot_example = """Question 1.) 
# What are the customer's issues and concerns? List them in order of priority.\n
# Answer 1.): ग्राहक की समस्याएं और चिंताएं हैं:\n1. प्रस्तुत किए गए दावे के लिए भुगतान न मिलना।\n2. दावा स्वीकृति को लेकर अनिश्चितता।
"""
# Existing messages list
import json
from tqdm import tqdm
plot_org_tokens = []
all_summary = {}
summary_data = []
for items in tqdm(data):
    key = items['filename']
    transc = items['data']['transcript']
    original_prompt = transc
    results = compressor.compress_prompt_llmlingua2(
        original_prompt,
        rate=0.6,
        force_tokens=['\n', '.', '!', '?', ','],
        chunk_end_tokens=['.', '\n'],
        return_word_label=True,
        drop_consecutive=True
    )# results['compressed_prompt']
    org_tokens = results['origin_tokens']
    comp_tokens = results['compressed_tokens']
    compressed_transc = results['compressed_prompt']
    compressed_transc = transc
    messages = [
        {"role": "system", "content": f"You are Gaja, an AI assistant created by Cognitivelab and trained on top of Llama 3 Large language model (LLM), proficient in English and Hindi. You must response in Hindi Language only using given transcript {compressed_transc} "},
        {"role": "user", "content": f"Answers the below Questions in Hindi with detailed information only using transcript given above . if you dont know the answer dont give false information #Questions : {question} . Format of question answer should be like this #Example: {one_shot_example} "}
    ]
 
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True, 
        return_tensors="pt"
    ).to("cuda")
    
    import time
    # st = time.time()
    outputs = model.generate(
        input_ids,
        max_new_tokens=3192,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        
        do_sample=True,
        temperature=0.3,
        top_p=0.4,
    )
    response = outputs[0][input_ids.shape[-1]:]

    
    all_summary[key] = {"output " : tokenizer.decode(response, skip_special_tokens=True),"original_tokens" : org_tokens , "compressed_tokens" :comp_tokens}
    
    

with open("test_wo_PComp1.json", "w", encoding="utf-8") as f:#test2
    json.dump(all_summary, f, indent=2, ensure_ascii=False)
