## :zap: DoodleFormer: Creative Sketch Drawing with Transformers
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

Please see ```INSTALL.md``` for installing required libraries. You can change the content in the file ```mytext.txt``` to visualize generated handwriting while training.   


Download Dataset files and models from https://drive.google.com/file/d/16g9zgysQnWk7-353_tMig92KsZsrcM6k/view?usp=sharing and unzip inside ```files``` folder. In short, run following lines in a bash terminal. 

```bash
git clone https://github.com/ankanbhunia/Handwriting-Transformers
cd Handwriting-Transformers
pip install --upgrade --no-cache-dir gdown
gdown --id 16g9zgysQnWk7-353_tMig92KsZsrcM6k && unzip files.zip && rm files.zip
```

To start training the model: run

```
python train.py
```

If you want to use ```wandb``` please install it and change your auth_key in the ```train.py``` file (ln:4). 

You can change different parameters in the ```params.py``` file.

You can train the model in any custom dataset other than IAM and CVL. The process involves creating a ```dataset_name.pickle``` file and placing it inside ```files``` folder. The structure of ```dataset_name.pickle``` is a simple python dictionary. 

```python
{
'train': [{writer_1:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]}, {writer_2:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...], 
'test': [{writer_3:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]}, {writer_4:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...], 
}
```

## Handwriting synthesis results

Please check the ```results``` folder in the repository to see more qualitative analysis. Also, please check out colab demo to try with your own custom text and writing style [![Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ankanbhunia/Handwriting-Transformers/blob/main/demo.ipynb)

 <p align="center">
<img src=Figures/paperresult.jpg width="1000"/>
</p>

 <p align="center">
<img src=Figures/qualresult.jpg width="1000"/>
</p>



## Handwriting reconstruction results
 Reconstruction results using the proposed HWT in comparison to GANwriting and Davis et al. We use
the same text as in the style examples to generate handwritten images.

 <p align="center">
<img src=Figures/recons2.jpg width="600"/>
</p>

<!-- 
<img src="Figures/result.jpg" >

<img src="Figures/recons2.jpg" >
 -->

