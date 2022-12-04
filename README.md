# AfriSenti-SemEval 2023 Shared Task 12

In this repo we currently have focussed on understanding following
* Do we always need more data (pre-trained models) to perform well on low-resource languages.

* Are there any similarity among African languages. This helps in countering any shortage of data for some African languages.

* How do fine-tuned model compare to pre-trained African models in zero-shot performance.

Methodology
* The analysis is done using mBert and afriBerta for models to fine-tune and one pre-trained on African languages respectively.
* To compare subset of languages to all-languages performance, f1 scores of monolingual models are considered and ones crossing 50% mark are chosen.
* Pytorch framework is used and Adam optimizer, Cycle LR scheduler is used for training. Mixed Precision Training and Gradient Accumulation is used for memory and time optimization. Due to resource contraints, the models were fine-tuned on V100 GPU for only 5 epochs.
