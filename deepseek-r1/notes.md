* deepseek-r1-zero and deepseek-r1 trained using rl instead of sft
* deepseek-r1-zero has "intriguing reasoning behaviours" but struggles with poor readability and language mixing.
* *DeepSeek-R1-Zero encounters challenges such as poor readability, and language mixing* - my thought is that deepseek-r1-zero might be the most important result of the paper. language mixing and poor readability
are from a human pov, for the model it might be a mode of communication.

* to get deepseek-r1, they take deepseek-v3 put it through the following:
    1. finetune with some cold start data to fix language mixing and readability issues 
    2. reasoning oriented RL
    3. sft using data gathered through rejection sampling and deepseek-v3
    4. another round of rl for a broad range of prompts

* *We directly apply RL to the base model without relying on supervised fine-tuning (SFT) as a preliminary step. This approach allows the model to explore chain-of-thought (CoT) for solving complex problems, resulting in the development of DeepSeek-R1-Zero. DeepSeekR1-Zero demonstrates capabilities such as self-verification, reflection, and generating long CoTs, marking a significant milestone for the research community. Notably, it is the first open research to validate that reasoning capabilities of LLMs can be incentivized purely through RL, without the need for SFT. This breakthrough paves the way for future advancements in this area.*

* *We do not apply the outcome or process neural reward model in developing DeepSeek-R1-Zero, because we find that the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process, and retraining the reward model needs additional training resources and it complicates the whole training pipeline.*

* *thinking time of DeepSeek-R1-Zero shows consistent improvement throughout the training process*