## :zap: DoodleFormer: Creative Sketch Drawing with Transformers (ECCV'22)
<!-- 
[arXiv](https://arxiv.org/abs/2112.03258) | [paper](https://arxiv.org/pdf/2112.03258) | [demo](https://colab.research.google.com/github/ankanbhunia/DoodleFormer/blob/main/demo.ipynb) | [BibTeX](#bibtex)
  -->
 <p align='center'>
  <b>
    <a href="https://arxiv.org/abs/2112.03258">ArXiv</a>
    | 
    <a href="https://arxiv.org/pdf/2112.03258">Paper</a>
    | 
    <a href="https://colab.research.google.com/github/ankanbhunia/DoodleFormer/blob/main/demo.ipynb">Demo</a>
  </b>
</p> 

 
 <p align="center">
<img src=figures/qual.jpg />
</p>

<!-- 
<img src="Figures/Result.gif" width="800"/>
 -->
 
[Ankan Kumar Bhunia](https://scholar.google.com/citations?user=2leAc3AAAAAJ&hl=en),
[Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en),
[Hisham Cholakkal](https://scholar.google.com/citations?user=bZ3YBRcAAAAJ&hl=en), 
[Rao Muhammad Anwer](https://scholar.google.fi/citations?user=_KlvMVoAAAAJ&hl=en),
[Fahad Shahbaz Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao),
[Jorma Laaksonen](https://scholar.google.com/citations?user=qQP6WXIAAAAJ&hl=en) &
[Michael Felsberg](https://scholar.google.com/citations?user=lkWfR08AAAAJ&hl=en)



> **Abstract:** 
>*Creative sketching or doodling is an expressive activity, where imaginative and previously unseen depictions of everyday visual objects are drawn. Creative sketch image generation is a challenging vision problem, where the task is to generate diverse, yet realistic creative sketches possessing the unseen composition of the visual-world objects. Here, we propose a novel coarse-to-fine two-stage framework, DoodleFormer, that decomposes the creative sketch generation problem into the creation of coarse sketch composition followed by the incorporation of fine-details in the sketch. We introduce graph-aware transformer encoders that effectively capture global dynamic as well as local static structural relations among different body parts. To ensure diversity of the generated creative sketches, we introduce a probabilistic coarse sketch decoder that explicitly models the variations of each sketch body part to be drawn. Experiments are performed on two creative sketch datasets: Creative Birds and Creative Creatures. Our qualitative, quantitative and human-based evaluations show that DoodleFormer outperforms the state-of-the-art on both datasets, yielding realistic and diverse creative sketches. On Creative Creatures, DoodleFormer achieves an absolute gain of 25 in terms of Fr\`echet inception distance (FID) over the state-of-the-art. We also demonstrate the effectiveness of DoodleFormer for related applications of text to creative sketch generation and sketch completion..* 

## Citation

If you use the code for your research, please cite our paper:

```
@article{bhunia2021doodleformer,
  title={Doodleformer: Creative sketch drawing with transformers},
  author={Bhunia, Ankan Kumar and Khan, Salman and Cholakkal, Hisham and Anwer, Rao Muhammad and Khan, Fahad Shahbaz and Laaksonen, Jorma and Felsberg, Michael},
  journal={ECCV},
  year={2022}
}
```

## Software environment

- Python 3.7
- PyTorch >=1.4

## Setup & Training

Please see ```INSTALL.md``` for installing required libraries. First, create the enviroment with Anaconda. Install Pytorch and the other packages listed in requirements.txt. The code is tested with PyTorch 1.3.1 and CUDA 10.0:

```bash
  git clone https://github.com/ankanbhunia/doodleformer
  conda create -n doodler python=3.7
  conda activate doodleformer
  conda install pytorch==1.4 -c pytorch
  pip install -r requirements.txt
```

Next, download the processed Creative Birds and Creative Creatures datasets from the GoogleDrive: https://drive.google.com/drive/folders/14ZywlSE-khagmSz23KKFbLCQLoMOxPzl?usp=sharing and unzip the folders under the directory `../data/`.

To process the raw data from the scratch, check the scripts in `data_process.py`.



### [Stage-1] PL-Net Training

The first stage, PL-Net, takes the initial stroke points as the conditional input and learns to return the bounding boxes corresponding to each body part (coarse structure of the sketch) to be drawn. To train the PL-Net, run the following command:

```
python PL-Net.py --dataset='sketch-bird' \
                 --data_path='data/doodledata.npy' \
                 --exp_name='ztolayout' \
                 --batch_size=32 \
                 --beta=1 \
                 --model_dir='models' \
                 --save_per_epoch=20 \
                 --vis_per_step=200 \
                 --learning_rate=0.0001
                 
```


### [Stage-2] PS-Net Training

The second stage, PS-Net, takes the predicted box locations along with C as inputs and generates the final sketch image. To train the PS-Net, run the following command:

```
python PS-Net.py --dataset='sketch-bird' \
                 --data_path='data/doodledata.npy' \
                 --exp_name='layouttosketch' \
                 --batch_size=32 \
                 --beta=1 \
                 --model_dir='models' \
                 --save_per_epoch=20 \
                 --vis_per_step=200 \
                 --learning_rate=0.0001
                 
```

If you want to use ```wandb``` please install it and change your auth_key in the ```train.py``` file (ln:4). 

You can change different parameters in the ```params.py``` file.

### Inference

TBD

