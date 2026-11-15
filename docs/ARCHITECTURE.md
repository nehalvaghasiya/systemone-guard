# Architecture and reliability design

The selected network is small (21,306 parameters) because it is optimized for a narrow, typed decision space. Raw text fields are word-tokenized and padded to 96 tokens. The network embeds words, performs two parallel depthwise convolutions with 3- and 5-token windows, concatenates the original embeddings with convolution responses, applies a shared projection and attention pooling, then concatenates five categorical state embeddings. Four fixed-schema heads return logits simultaneously: three-way decision and three binary auxiliary tasks.

**Why not autoregression?** With fixed answer spaces there is no reason to generate JSON characters or decode a label token by token. Direct logits also support softmax probabilities and validation-only temperature scaling. This alone does not guarantee honest or robust confidence; calibration must be tested separately under distribution shift.

**Why not proprietary Jev?** The public launch only characterizes Jev by its single-pass typed outputs, parallel inference and RLCD. Reimplementation of undisclosed components would be guesswork. This project is from-scratch PyTorch supervised multitask training; it does not use RLCD or claim to reproduce Jev.

**Policy input:** approval state and data sensitivity are supplied as trusted structured fields. Untrusted context and natural-language request are model features, not authority. The simulated gold policy is public in `src/systemone_guard/data.py`; it should be treated as a policy oracle, not as ground-truth evidence that an ML classifier is safe.

**Never wire the classifier directly to destructive tools.** A production design uses a deterministic identity/RBAC/ABAC/tenant authorization boundary, tool allowlists, sensitive-data controls, explicit human approvals, independent logging and rate limiting, with model output as one advisory signal.
