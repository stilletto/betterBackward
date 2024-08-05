from mmfreelm.models import HGRNBitConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# Define the source Mistral 7B model to be converted and the name for the new model
source_model_name = "mistral-7B"
target_model_name = "matmul-free-mistral-7B"

# Load the source Mistral 7B model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained(source_model_name)
print("tokenizer loaded")
model = AutoModelForCausalLM.from_pretrained(source_model_name)
print("model loaded")
# Define the configuration for the MatMul-Free model
config = HGRNBitConfig(
    vocab_size=32000,  # Use the same vocabulary size
    hidden_size=2048,  # Define the hidden size
    num_hidden_layers=24,  # Number of layers in the model
    num_heads=8,  # Number of attention heads
    attn_mode='fused_recurrent'  # Using fused recurrent attention
)
print("config")
# Initialize the MatMul-Free model
matmul_free_model = HGRNBitForCausalLM(config)
print("init matmul free model")

# Copy weights from the source model to the MatMul-Free model
def copy_weights(src_model, tgt_model):
    tgt_model.model.embeddings.weight.data = src_model.model.embeddings.weight.data.clone()

    for src_layer, tgt_layer in zip(src_model.model.layers, tgt_model.model.layers):
        tgt_layer.attn.i_proj.weight.data = src_layer.attention.self.query.key.weight.data.clone()
        tgt_layer.attn.f_proj.weight.data = src_layer.attention.self.query.value.weight.data.clone()
        tgt_layer.attn.g_proj.weight.data = src_layer.attention.self.query.query.weight.data.clone()
        tgt_layer.attn.o_proj.weight.data = src_layer.attention.output.dense.weight.data.clone()

        tgt_layer.mlp.gate_proj.weight.data = src_layer.intermediate.dense.weight.data.clone()
        tgt_layer.mlp.down_proj.weight.data = src_layer.output.dense.weight.data.clone()

        tgt_layer.attn_norm.weight.data = src_layer.attention.self.query.layernorm.weight.data.clone()
        tgt_layer.mlp_norm.weight.data = src_layer.pre_ln.weight.data.clone()

    tgt_model.lm_head.weight.data = src_model.lm_head.weight.data.clone()


# Copy the weights
copy_weights(model, matmul_free_model)

# Save the new model
matmul_free_model.save_pretrained(target_model_name)
tokenizer.save_pretrained(target_model_name)




import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the target MatMul-Free model and its tokenizer
model_name = 'matmul-free-mistral-7B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda().half()

# Define an input prompt
input_prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "

# Tokenize the input text
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()

# Generate text using the model
outputs = model.generate(input_ids, max_length=32, do_sample=True, top_p=0.4, temperature=0.6)

# Decode and print the output text
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(generated_text)