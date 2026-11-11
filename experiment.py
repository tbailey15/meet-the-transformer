# Meet the Transformer: tokenization and generation experiments with DistilGPT2.
# Run: python experiment.py > results/results.txt
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "distilbert/distilgpt2"
MAX_NEW = 40  # new tokens per generation

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()  # inference only, no training


def title(text):
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70)


def show_tokens(text):
    ids = tokenizer(text)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"\nText:    {text!r}")
    print(f"Tokens:  {tokens}  ({len(ids)} tokens)")
    print(f"IDs:     {ids}")
    print(f"Decoded: {tokenizer.decode(ids)!r}")


def generate(prompt, seed=42, **settings):
    torch.manual_seed(seed)  # same seed + same settings = same output
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW,
                             pad_token_id=tokenizer.eos_token_id, **settings)
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def run(label, prompt, **kwargs):
    print(f"\n[{label}]\n{prompt}|{generate(prompt, **kwargs)!r}")


title("SETUP")
print("Model:", MODEL_ID)
print("torch:", torch.__version__, "| transformers:", transformers.__version__)
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Vocabulary size:", tokenizer.vocab_size)

title("1. TOKEN INSPECTION")
for text in ["Hello world", "hello world", "The cat sat on the mat.",
             "unbelievably", "Albuquerque", "tokenization", "2026", "COVID-19"]:
    show_tokens(text)

title("2. REPEATABILITY (same prompt, controlled settings)")
prompt = "The best way to learn programming is"
for i in range(3):
    run(f"Greedy, run {i + 1}", prompt, do_sample=False)
for i in range(3):
    run(f"Sampling temp=0.8, seed=42, run {i + 1}", prompt, do_sample=True, temperature=0.8)
for seed in [1, 2, 3]:
    run(f"Sampling temp=0.8, seed={seed}", prompt, do_sample=True, temperature=0.8, seed=seed)

title("3. CHANGING TEMPERATURE (seed=42, top_k=0, only temperature changes)")
prompt = "Our new coffee shop is"
for temp in [0.2, 0.7, 1.0, 1.5]:
    run(f"temperature={temp}", prompt, do_sample=True, temperature=temp, top_k=0)

title("4. CHANGING PROMPT WORDING (same task, greedy so only wording changes)")
for prompt in ["Coffee mug",
               "Product description: This ceramic coffee mug",
               "Q: Can you describe a ceramic coffee mug?\nA:",
               "Our new ceramic coffee mug keeps your drink hot and"]:
    run("Write about a coffee mug", prompt, do_sample=False)

title("5. FAILURES AND LIMITATIONS (greedy)")
run("Fact", "The capital of New Mexico is", do_sample=False)
run("Math", "Q: What is 12 + 7?\nA:", do_sample=False)
run("Instruction", "Translate to Spanish: Good morning, how are you?", do_sample=False)
run("Customer question", "Customer: Is your product safe for children?\nSupport agent:", do_sample=False)
