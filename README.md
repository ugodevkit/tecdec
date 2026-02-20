# TecDec
The TecDec model is an experiment that applyes the text classification CNN approach to a macro level social analysis (power dynamics). It is designed to be a tool for various multidisciplinary venues, but mainly for CCDA (Computational Critical Discourse Analysis). The project is in beta, but I hope to develop it fast.

In this repository you will find a Python Trial Workbench application, that helps you download the model from Hugging Face (SpaCy best model, or a wheel), the weights, and even the raw JSONL data, and play a bit with them. There is a test-tab and a benchmark-tab. It is still in a development phase, so there can be errors.

The main program for the extraction elaboration, labeling, testing, benchmarking, and playgrounds is not yet in a reprository on GitHub, but it will be soon realeased (I hope '^^). It will be a full fledged Java/Python hybrid application, that has the ambition of becoming a generic text-classifier-training suite... but that's the future, for now.

## To run the workbench
Simply run this:
```bash
uv run --with-requirements requirements.txt tecdectrial.py 
```
It should do the uv magic...

## Resources
The Hugging Face model and dataset are available here:
- ugo86/ja_tecdec_labeler
- ugo86/ja_tecdec_labels

## License
Although simple, this collection of scripts is released under GPL-3.0

## Special Thanks
A special thank is due to the LLM-jp dataset creators and mantainers (and all the projects connected).
You can find their work at [llm-jp-corpus-v4](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4).
Note: This project is an independent derivative and is not endorsed by llm-jp-corpus-v4 creators.
