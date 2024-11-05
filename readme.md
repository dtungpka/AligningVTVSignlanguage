# README

## Introduction

This project is for training sign language recognition models using skeleton and RGB data.

## Requirements

- Python 3.9.13
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Data

- Skeleton and RGB data can be downloaded at [skeleton.zip](https://daihocphenikaa-my.sharepoint.com/:u:/g/personal/21010294_st_phenikaa-uni_edu_vn/EeqGCq4MsZtMm54TiKhx980BAPQSrwMjPX_hpMRaSMD-Uw?download=1)
- Evaluation dataset can be downloaded at [evaluation_pack.pkl](https://daihocphenikaa-my.sharepoint.com/:u:/g/personal/21010294_st_phenikaa-uni_edu_vn/ESIr97SZa6FNtG3P76ddu9EBk_eKCnVG-HHM71rjNVjKpQ?download=1)

## Usage

### Running on Google Colab or Jupyter Notebook

If you are running on Google Colab or Jupyter Notebook, please refer to `notebook.ipynb`

### Running Locally

1. **Select Signs to Train**

   Run `select_signs.py` to create a list of signs to train:

   ```bash
   python select_signs.py
   ```

   This will generate a `pending_training.txt` file.

2. **Train the Model**

   Run `train.py` to start training:
   ```bash
   python train.py
   ```
   The training results will be saved in the `results` folder.

## Augment Viewer

The `augment_viewer.ipynb` or `augment_viewer.py` script can interactively show augmented variations of the skeleton data.

## Model Architecture

The `model.py` file holds the implementation of the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.