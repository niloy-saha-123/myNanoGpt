# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out'
eval_interval = 50 # Number of training iterations between evaluating accuracy
eval_iters = 20 # Number of iterations to perform when evaluating accuracy
log_interval = 1 # Number of iterations between printing results

# Should we save the checkpoint on every evaluation or only when accuracy improves?
always_save_checkpoint = False

wandb_log = True # Should we log results to wandb.ai?
wandb_project = 'comp560-onboarding' # A wandb project is a collection of related experiments
wandb_run_name = 'my-experiment-quick-test'

dataset = 'shakespeare_char' # The name of the directory containing Prepared data including meta.pkl, train.bin, val.bin
gradient_accumulation_steps = 1 # The nanoGPT training algorithm distinguishes between a full batch and a micro batch. The number of training instances in a full batch is gradient_accumulation_steps * batch_size. 
batch_size = 12 # The number of training instances in a micro batch.
block_size = 64 # The number of tokens in the context window of the GPT

# baby GPT model :)
n_layer = 4 # The number of layers in the GPT
n_head = 4 # The number of attention heads in the GPT
n_embd = 128 # The embedding size of the GPT, also often called the model dimension. It is the length of each vector processed by each layer. Tokens are converted into vectors of length n_emdb before any other processing takes place. The vector is split when processed by attention heads -- its values are partitioned over each head, so we require that n_emdb is a multiple of n_emdb 
dropout = 0.0 # The fraction of network connections that are randomly disabled during training, to encourage robustness.

learning_rate = 1e-3 # This determines how much the neural network parameters change at each time step while training. This parameter actually specifies the initial learning rate which will probably be decayed over time to improve accuracy. Guideline: 1e-4 may be suitable for small models with 100 million parameters. For very tiny models (<10M params), 1e-3 may be suitable.
max_iters = 50 # The maximum number of training iterations to perform.
lr_decay_iters = 2000 # The number of iterations over which the learning rate will decay. Make equal to max_iters, usually.
min_lr = 1e-4 # The final learning rate after it has been decayed -- learning_rate / 10 usually
beta2 = 0.95 # This relates to how much momentum the changes in parameters are given, and therefore how quickly the network adapts. An AI assistant suggested 0.95 is widely used. Karpathy suggests 0.99 for tiny models, stating: "make a bit bigger because number of tokens per iter is small".

warmup_iters = 100 # There is another training phase called warm up in which the learning rate starts very close to 0 and increases. This is the number of iterations for warm up. When the number of iterations is large, 1% warm up would be typical. For tiny models, Karpathy states: "not super necessary potentially"

device = 'mps'  # running on macbook's gpu
compile = False # do not torch compile the model

