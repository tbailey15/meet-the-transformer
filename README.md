# Meet the Transformer (DistilGPT2)

A small experiment to see how a pretrained transformer tokenizes text and how
prompts and generation settings change its output. No training, only inference.

- **Model:** `distilbert/distilgpt2` (81,912,576 parameters, Apache 2.0 License)
- **Versions:** Python 3.14.7, torch 2.15.0, transformers 5.22.2 (all in `requirements.txt`)
- **Code:** `experiment.py` | **Full output:** `results/results.txt`
- Every run makes 40 new tokens. Seeds are set with `torch.manual_seed`.

## How to run it

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
python -m pip install torch transformers
```

```bash
python experiment.py > results/results.txt
```

The first run downloads the model into the Hugging Face cache. Runs on CPU.

## Key ideas

**Training vs. inference.** Training changes the model's weights by showing it
lots of text and correcting its next-token guesses. Inference uses the finished
weights as they are to make predictions. This project only does inference
(`model.eval()` and `torch.no_grad()`), so the model never learns anything new.

**Tokens.** The model can't read letters or words. The tokenizer splits text into
tokens (whole words, pieces of words, or symbols) and maps each one to an ID from a
50,257-token vocabulary. The model reads and writes these IDs. `Ġ` in a token means
it starts with a space.

**Context.** The model predicts the next token from all the tokens before it (the
prompt plus what it has written so far). Changing the prompt changes what it thinks
should come next, so the wording, format, and details of the prompt shape the output.

## 1. Token inspection

| Text | Tokens | IDs |
|---|---|---|
| `Hello world` | `Hello`, `Ġworld` | 15496, 995 |
| `hello world` | `hello`, `Ġworld` | 31373, 995 |
| `unbelievably` | `un`, `bel`, `iev`, `ably` | 403, 6667, 11203, 1346 |
| `Albuquerque` | `Al`, `buquerque` | 2348, 36461 |
| `tokenization` | `token`, `ization` | 30001, 1634 |
| `2026` | `20`, `26` | 1238, 2075 |
| `COVID-19` | `CO`, `VID`, `-`, `19` | 8220, 11008, 12, 1129 |

Decoding the IDs gives back the exact original text every time.

**What I saw:** common words are one token, but rare or long words get split into
pieces (`unbelievably` = 4 tokens). Numbers are split too (`2026` is `20` + `26`),
so the model doesn't see a number as a value, which helps explain why it is bad at
math. Capital letters matter: `Hello` and `hello` have different IDs.

## 2. Repeatability

Prompt: `The best way to learn programming is`

| Setting | 3 runs |
|---|---|
| Greedy (`do_sample=False`) | Identical: "to learn from the experience of the programmer." |
| Sampling, temp 0.8, seed 42 | Identical: "to follow the same rule as in your own programming:" |
| Sampling, temp 0.8, seeds 1, 2, 3 | All different |

Greedy always picks the most likely token, so it never changes. Sampling picks
randomly, but with the same seed it makes the same picks, so results can be
reproduced. A new seed gives a new answer. Seed 2 got stuck repeating one sentence.

## 3. Changing temperature

Prompt: `Our new coffee shop is` (seed 42, `top_k=0`, only temperature changes)

| Temp | Output (start) |
|---|---|
| 0.2 | "in the works." then only blank lines |
| 0.7 | "gearing up for a new year in Toronto." |
| 1.0 | "gearing up for a $100 grand operating income, ending its Vice Coast Lines takeover..." |
| 1.5 | "...sustainably and once again would we say it farewell for us. For this It AC Jobs Thomas Colombat..." |

Low temperature is safe but boring and quickly runs out of things to say. Around
0.7 it is the most readable. At 1.0 it makes up facts, and at 1.5 it turns to
nonsense. Higher temperature = more random.

## 4. Changing prompt wording

Same task (write about a coffee mug), greedy so only the wording changes:

| Prompt | Output |
|---|---|
| `Coffee mug` | "." then blank lines |
| `Product description: This ceramic coffee mug` | "is a perfect gift for a family of friends." repeated 3+ times |
| `Q: Can you describe a ceramic coffee mug?\nA:` | "I can't say." repeated 8 times |
| `Our new ceramic coffee mug keeps your drink hot and` | "fresh." then blank lines |

Too little context gives nothing useful. A label like "Product description:" pushes
it toward ad-style text. A question format makes it act like an answer, but it
didn't answer. The best results came from starting the sentence for it.

## 5. Failures and risks

| Prompt | Output | Problem |
|---|---|---|
| `The capital of New Mexico is` | "home to the largest number of marijuana dispensaries in the country." | Wrong and made up (it's Santa Fe) |
| `Q: What is 12 + 7?\nA:` | "The answer is that the answer is that..." | Can't do math, loops |
| `Translate to Spanish: Good morning...` | Only blank lines | Can't follow instructions |
| `Customer: Is your product safe for children?\nSupport agent:` | Only blank lines | No answer at all |

Other problems: it often repeats itself or fills the rest with blank lines, and
one sample printed a hidden Unicode character (`‪`).

**Risks for a customer-facing product:**
- It states false things with confidence, which could mislead customers or create
  legal trouble (like a wrong answer about child safety).
- It doesn't follow instructions, so it can't be trusted to answer questions.
- Repeated or empty replies look broken.
- Higher temperatures can make random, off-topic, or embarrassing text.

## What I learned

The same seed and settings always give the same output, so results can be checked.
Temperature trades safety for variety: low is repetitive, high is nonsense.
Prompt wording matters a lot, and DistilGPT2 works best when you start a sentence
for it to continue. It is a text continuer, not an assistant.

**Recommendation:** DistilGPT2 is fine for learning but not for a customer feature.
Further prototyping is only worth it with a larger instruction-tuned model, plus
a repetition penalty, length limits, and human review of the output.

## References

Géron, A. (2025). *Hands-On Machine Learning with Scikit-Learn and PyTorch.* O'Reilly Media.

Grootendorst, M., and Alammar, J. (2026). *An Illustrated Guide to AI Agents.* O'Reilly Media.

Hugging Face. DistilGPT2 model card: https://huggingface.co/distilbert/distilgpt2
