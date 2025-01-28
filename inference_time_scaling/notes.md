* main concepts of this paper are:
    - compare two different test-time scaling strategy: 
        1. using a verifier to either sequentially or parallely generate the answer. 
        2. using a PRM+search to generate answer

* to identify the best strategy during testing, they use the base model to predict the 
difficulty bin of the prompt/question and based on the results from the validation set, they use the strategy corresponding to the particular difficulty bin

* *Humans tend to think for longer on difficult problems to reliably improve their decisions. Can we instill a similar capability into today’s large language models (LLMs)?*

* "proposal distribution" just means the language model. why use a fancy name?

* *With both approaches, we find that the efficacy of a particular test-time compute strategy depends critically on both the nature of the specific problem at hand and the base LLM used.*

* *We find that on easy and intermediate questions, and even hard questions (depending on the specific conditions on the pretraining and inference workload), additional test-time compute is often preferable to scaling pretraining. This finding suggests that rather than focusing purely on scaling pretraining, in some settings it is be more effective to pretrain smaller models with less compute, and then apply test-time compute to improve model outputs*

* *this hints at a future where fewer FLOPs are spent during pretraining and more FLOPs are spent at inference.*

* *we define the **test-time compute-optimal scaling strategy** as the strategy that chooses hyperparameters corresponding to a given test-time strategy for maximal performance benefits on a given prompt at test time*