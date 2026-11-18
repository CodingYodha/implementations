# implementations

From-scratch PyTorch implementations of vision transformer papers. Each notebook builds the model out of basic layers (patching, attention, encoder and decoder blocks) instead of loading a ready-made model.

## What's inside

| Notebook | Paper | Dataset | Status |
|---|---|---|---|
| [ViT.ipynb](ViT.ipynb) | Vision Transformer (*An Image is Worth 16x16 Words*) | MNIST | Trained for 5 epochs |
| [MAE.ipynb](MAE.ipynb) | Masked Autoencoders (*Masked Autoencoders Are Scalable Vision Learners*) | CIFAR-10 | Trained for 5 epochs |
| [VideoMAE.ipynb](VideoMAE.ipynb) | VideoMAE (masked autoencoders for video) | Video frames loaded with OpenCV | Model code only, not trained yet |

## How it works

- **ViT:** the image is cut into 4×4 patches. Each patch becomes an embedding with position information. The embeddings go through transformer encoder blocks (multi-head self-attention + MLP), then a classifier head predicts the digit.
- **MAE:** the image is cut into 2×2 patches and 75% of them are hidden at random. The encoder only sees the visible patches. A small decoder then rebuilds the hidden pixels.
- **VideoMAE:** the same masking idea applied to a video clip, stored as frames of shape (T, C, H, W).

The attention, MLP, encoder and decoder blocks are all written by hand in `torch.nn`.

## Results

Numbers below are printed in the notebooks.

| Model | Dataset | Result |
|---|---|---|
| ViT | MNIST | 97.39% accuracy after 5 epochs |
| MAE | CIFAR-10 | Validation reconstruction loss 0.0382 after 5 epochs |
| VideoMAE | – | [TODO: add result] |

The MAE was trained with smaller dimensions, fewer encoder/decoder layers and fewer epochs so it fits on a single consumer GPU. Larger settings should give better reconstructions.

## How to run

```bash
pip install torch torchvision torchinfo matplotlib opencv-python
jupyter notebook ViT.ipynb
```

Open `MAE.ipynb` or `VideoMAE.ipynb` the same way. MNIST and CIFAR-10 are downloaded by `torchvision` the first time you run the notebooks. VideoMAE needs a video file of your own.
