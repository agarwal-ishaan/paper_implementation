# Paper Implementations

Working through a personal reading list of ML papers, one at a time — implementing each from
scratch and writing up what I learned, rather than just reading and moving on. Each subfolder
holds the paper PDF plus (once tackled) an implementation, tests, and a notebook walkthrough.

## Workflow

Each paper goes through the same loop:

1. **Design** — a short spec (`design.md`) covering the paper's core mechanism, what's being
   simplified relative to the paper's original compute budget, and what the demo needs to show.
2. **Plan** — a task-by-task implementation plan (`plan.md`), TDD-first: tests before
   implementation for anything with non-obvious logic.
3. **Implement** — the core mechanism as tested, importable Python (`model.py`, `train.py`, ...),
   not notebook-only code.
4. **Walkthrough** — a Jupyter notebook tying it together: paper explanation, the implementation,
   a real training run, and visualizations of what the mechanism is actually doing.

## Papers

| Paper | Folder | Status |
|---|---|---|
| Deep Networks with Stochastic Depth | [`stochastic_depth/`](stochastic_depth/) | ✅ implemented — [notebook](stochastic_depth/stochastic_depth.ipynb), [writeup](stochastic_depth/README.md) |
| LoRA | [`lora/`](lora/) | ✅ implemented — [notebook](lora/lora.ipynb), [writeup](lora/README.md) |
| A Watermark for Large Language Models | [`llm_watermark/`](llm_watermark/) | not yet |
| Amortized Planning with Large-Scale Transformers (Chess) | [`amortized_planning_chess/`](amortized_planning_chess/) | not yet |
| CLASP | [`clasp/`](clasp/) | not yet |
| Cold Diffusion | [`cold_diffusion/`](cold_diffusion/) | not yet |
| Contrastive Decoding | [`contrastive_decoding/`](contrastive_decoding/) | not yet |
| Training data-efficient image transformers (DeiT) | [`deit/`](deit/) | not yet |
| DistillBERT | [`DistillBERT/`](DistillBERT/) | 🚧 design + plan written, not yet implemented |
| DoRA | [`dora/`](dora/) | not yet |
| EfficientNet | [`efficientnet/`](efficientnet/) | not yet |
| How Attentive are Graph Attention Networks (GATv2) | [`gatv2/`](gatv2/) | not yet |
| Graph Attention Networks (GAT) | [`graph_attention_networks/`](graph_attention_networks/) | not yet |
| Image2StyleGAN++ | [`image2stylegan_plus_plus/`](image2stylegan_plus_plus/) | not yet |
| Neural ODEs | [`neural_odes/`](neural_odes/) | not yet |
| PatchTST (A Time Series is Worth 64 Words) | [`patchtst_time_series/`](patchtst_time_series/) | not yet |
| PerSAM (Personalize Segment Anything with One Shot) | [`persam/`](persam/) | not yet |
| Prefix-Tuning | [`prefix_tuning/`](prefix_tuning/) | not yet |
| Retrieval-Augmented Generation (RAG) | [`rag/`](rag/) | not yet |
| RePaint | [`repaint/`](repaint/) | not yet |
| RePlug | [`replug/`](replug/) | not yet |
| 3D Self-Supervised Methods for Medical Imaging | [`self_supervised_medical_imaging/`](self_supervised_medical_imaging/) | not yet |
| Simplifying Graph Convolutional Networks (SGC) | [`sgc/`](sgc/) | not yet |
| Show, Attend and Tell | [`show_attend_tell/`](show_attend_tell/) | not yet |
| SimpleShot | [`simpleshot/`](simpleshot/) | not yet |
| A Close Look at Deep Learning with Small Data | [`small_data_deep_learning/`](small_data_deep_learning/) | not yet |
| SteganoGAN | [`steganogan/`](steganogan/) | not yet |
| TimeGANs | [`timegan/`](timegan/) | not yet |
| Investigating the Limitations of Transformers with Simple Arithmetic Tasks | [`transformer_arithmetic_limitations/`](transformer_arithmetic_limitations/) | not yet |
| Tree of Thoughts | [`tree_of_thoughts/`](tree_of_thoughts/) | not yet |
| Generating Natural Questions About an Image | [`visual_question_generation/`](visual_question_generation/) | not yet |
