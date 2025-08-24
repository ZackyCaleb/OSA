
# Object-Semantic Alignment Across Domains: Enhancing Domain Invariance of Facial Expression Features for Unsupervised Domain Adaptation
This repository is the official implementation of “Object-Semantic Alignment Across Domains: Enhancing Domain Invariance of Facial Expression Features for Unsupervised Domain Adaptation”. 
## Data Preparation
   * Downloading the original images after obtaining official authorization for the mentioned datasets: [RAF-DB](http://whdeng.cn/RAF/model1.html), [KDEF](http://www.emotionlab.se/kdef/), [MMI](https://www.mmifacedb.eu/search/#), [SFEW](https://users.cecs.anu.edu.au/~few_group/AFEW.html), [Affectnet](http://mohammadmahoor.com/affectnet/), [FER2013](https://github.com/gitshanks/fer2013), [ExpW](https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html).
   * Allocating training and testing datasets.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python main.py --train_list0 <path_to_source_train_data> --test_list6 <path_to_target_train_data> --test_list10 <path_to_target_test_data> --pretrained <path_to_initial_weights> --learn_rate 1e-4 --lambda_1 1.0 --lambda_2 1.0
```

## Evaluation

To evaluate my model, run:

```eval
python test.py --model-file OSA.pth 
```

## Trained Models

You can download pretrained models here:

- [OSA_R2S](https://drive.google.com/file/d/1-1vIODhdZzdTOMXSgq-0LArVblmcF5or/view?usp=drive_link) trained on SFEW using RAF-DB as source domain.


## Contributing
We propose an Object-Semantic Alignment (OSA) model that incorporates dynamic coordinate attention, joint mutual information, and marginal entropy loss to enhance the domain invariance of facial representations and mitigate domain shifts.
## Acknowledgment
1. [Li Y, Zhang Z, Chen B, et al. Deep Margin-Sensitive Representation Learning for Cross-Domain Facial Expression Recognition[J]. IEEE Transactions on Multimedia, 2023, 25: 1359-1373.](https://ieeexplore.ieee.org/abstract/document/9676449)
2. [Zhang S, Zhang Y, Zhang Y, et al. A dual-direction attention mixed feature network for facial expression recognition[J]. Electronics, 2023, 12(17): 3595.](https://github.com/SainingZhang/DDAMFN/tree/main)
