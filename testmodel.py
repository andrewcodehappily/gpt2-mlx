"""get checkpoint and test inference speed and output quality"""
import mlx.core as mx
from transformers import AutoTokenizer
from gpt2 import GPT, GPTConfig
from inference import generate_text, VOCAB_SIZE
import time, glob, os

model = GPT(GPTConfig(vocab_size=VOCAB_SIZE))
model.apply(lambda x: x.astype(mx.bfloat16))

import re
latest = max(glob.glob("checkpoints/checkpoint_step_*/"), key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
weights_path = os.path.join(latest, "model_weights.safetensors")
model.load_weights(weights_path)
mx.eval(model.parameters())
print(f"read checkpoint：{os.path.basename(latest.rstrip('/'))}")

enc = AutoTokenizer.from_pretrained("gpt2")

tests = [
    ("Hello", [0, 0.5, 1.0]),
    ("The", [0, 0.5]),
    ("Once upon a time", [0, 0.7]),
    ("The theory of evolution by natural selection", [0, 0.5, 1.0]),
    ("The Industrial Revolution began", [0, 0.5]),
    ("Photosynthesis is the process by which", [0, 0.7]),
    ("The main difference between mitosis and meiosis is", [0, 0.5, 1.0]),
    ("In machine learning, overfitting occurs when", [0, 0.5]),
]

for prompt_text, temps in tests:
    for temp in temps:
        tokens = enc.encode(prompt_text)
        tokens = mx.array(tokens, dtype=mx.int32)
        prompt_arr = mx.expand_dims(tokens, axis=0)
        t0 = time.time()
        output = generate_text(model, prompt=prompt_arr, max_new_tokens=60, temperature=temp, top_k=40)
        dt = time.time() - t0
        print(f"\n{'='*60}")
        print(f"Prompt：{prompt_text}  |  temp={temp}  |  {dt:.1f}s")
        print(f"{'='*60}")
        print(output)
