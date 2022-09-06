# 2022 SNU FastMRI Challenge

### 0. Baseline Codes
- LISTatSNU (2022) FastMRI_challenge [[Source Code](https://github.com/LISTatSNU/FastMRI_challenge)]
- facebookresearch (2022) fastMRI [[Source Code](https://github.com/facebookresearch/fastMRI)]
- XiaowenK (2020) UNet_Family [[Source Code](https://github.com/XiaowenK/UNet_Family)]
- JonOttesen (2022) DIRCN [[Source Code](https://github.com/JonOttesen/DIRCN)]

### 1. Environment Setup
- 저희는 기본적으로 Google 에서 제공하는 Colab Pro+ 를 이용해 모델 학습 및 검증을 진행하였습니다.
- Train 과정에서는 주어진 데이터만을 사용하였으며, 1 ~ 280 번까지를 train 용도로, 281 ~ 407 번 까지를 validation 용도로 사용하였습니다 (train : validation = 7 : 3).
- Evaluate 과정에서도 마찬가지로 주어진 leaderboard 데이터만을 사용하였습니다.


### 2. How to Train
- 총 4개의 모델을 최종적으로 ensemble하여 evaluate 과정에 사용하였습니다. 각 모델을 Train시키는 코드에 대한 설명은 아래와 같습니다.
- 모델 1 (E2E Varnet에 fc layer 추가, cascade=6)
    - Code > `train_varnet.py` 실행
- 모델 2 (DIRCN의 ResXUnet을 기본 Unet으로 대체, cascade=6)
    - Code > `train_dircn_no_resxunet.py` 실행
- 모델 3 (DIRCN의 ResXUnet을 Attention Unet으로 대체, cascade=5)
    - Code > `train_dircn_no_resxunet_attn_unet.py` 실행
- 모델 4 (데이터의 image와 grappa 라벨을 2 channel 입력으로 받는 Attention Unet)
    - Code > `train_unet_attn.py` 실행


### 3. How to evaluate
- 코드 실행에 앞서, Train 과정을 거쳐 나온 best_model 들은 각각 이하의 위치에 저장됩니다.
    - 모델 1: model > test_varnet > checkpoints
    - 모델 2: model > test_dircn > checkpoints
    - 모델 3: model > test_dircn_attn > checkpoints
    - 모델 4: model > test_Unet_attn > checkpoints
- 각 위치에 저장된 모델의 파일명을 각각 이하의 파일명으로 변경합니다. 
    - 모델 1: `best_model_varnet.pt`
    - 모델 2: `best_model_dircn_with_unet.pt`
    - 모델 3: `best_model_dircn_with_attnunet.pt`
    - 모델 4: `best_model_attnunet.tar`
- Code > `evaluate.py` 코드를 실행합니다.
- `evaluate.py` 코드 실행이 완료되면, 이어서 Code > `leaderboard_eval.py` 코드를 실행합니다. 최종적인 SSIM 이 출력됩니다.
- `evaluate.py` 실행 결과 reconstruction 된 파일들은 model > reconstructions_forward 내에 저장됩니다.


### 4. Extra Information
- 실행 시 Code, utils, model, readme.txt 외의 폴더 및 파일들은 실행에 영향을 주지 않습니다.
- model 폴더는 Train 과정에서 자동으로 생성됩니다.
- 문제를 해결한 방식에 대한 소개는 [Ideas in Models.pdf](https://github.com/frogyunmax/fastMRI/blob/master/Ideas%20in%20Models.pdf) 파일 내에 설명되어 있습니다.
