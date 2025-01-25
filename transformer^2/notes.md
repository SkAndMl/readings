* concept of self-adaptive llms is interesting and could become really important

* Transformer^2 -> a novel self-adaptation framework that adapts LLMs for unseen tasks in real-time by selectively adjusting only the singular components of their weight matrices

* uses 2 pass 
    1. first pass for identifying the task
    2. second pass for using the trained vectors for the particular task

* since it trains vectors instead of matrices like in LORA, it trains much fewer params while performing better than/equal to LORA

* Singular Value Fine-tuning (SVF) works by only extracting and finetuning the singular values of the weight matrices