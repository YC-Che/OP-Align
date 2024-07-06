# OP-Align: Object-level and Part-level Alignment for Self-supvervised Category-level Articulated Object Pose Estimation
OP-Align is a model designed for category-level articulated object pose esimation task with self-supervised learning.
This repo contains the code of OP-Align model and our real-word dataset.


## Dataset
We provide a novel real-world dataset for category-level Articulated Object Pose Estimation task.
You can download our dataset from ...

After downloading, put files into data dir.

Each *.npz files contains point cloud captured from a single-view RGB-D camera. To visualize the data based on image format, reshape array into (480,640,-1).

- pc (307200, 3) # 480 * 640 * xyz
- color (307200, 3) # 480 * 640 * rgb
- detection (307200,) # 480 * 640, maskRCNN/SAM result
- segmentation (307200,) # 480 * 640, segmentation GT, 0 indicates background
- part (2, 15) # P * (9+3+3), per-part rotation, translation, scale
- joint (1, 6) # J * (3+3), per-joint direction, pivot



## Enviroment
OP-Align uses a similar enviroment with [E2PN](https://github.com/minghanz/E2PN/tree/main) and adds PyTorch3D module.
```
conda env create -f OP_environment.yml
conda activate OP
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
cd vgtk; python setup.py build_ext --inplace; cd ..
mkdir log
ln -s <dataset_location> dataset
```
## Training
Each category has different joint settings


python run_art.py train --num-iterations 20000 experiment --seed 1234 --experiment-id Any_Name_You_like --run-mode train equi_settings --dataset-type Real --shape-type basket_output --nmasks 3 --njoints 2 --partial 0 model --rotation-range 120 --joint-type r --prob-threshold 0.05 --rigid-cd-w 0.5 --color-cd-w 0

python run_art.py train --num-iterations 20000 experiment --seed 1234 --experiment-id Any_Name_You_like --run-mode train equi_settings --dataset-type Real --shape-type drawer_output --nmasks 2 --njoints 1 --partial 0 model --rotation-range 120 --joint-type p --prob-threshold 0.05 --rigid-cd-w 0.5 --color-cd-w 0

python run_art.py train --num-iterations 20000 experiment --seed 1234 --experiment-id Any_Name_You_like --run-mode train equi_settings --dataset-type Real --shape-type laptop_output --nmasks 2 --njoints 1 --partial 0 model --rotation-range 120 --joint-type r --prob-threshold 0.05 --rigid-cd-w 0.5 --color-cd-w 0

python run_art.py train --num-iterations 20000 experiment --seed 1234 --experiment-id Any_Name_You_like --run-mode train equi_settings --dataset-type Real --shape-type suitcase_output --nmasks 2 --njoints 1 --partial 0 model --rotation-range 120 --joint-type r --prob-threshold 0.05 --rigid-cd-w 0.5 --color-cd-w 0

python run_art.py train --num-iterations 20000 experiment --seed 1234 --experiment-id Any_Name_You_like --run-mode train equi_settings --dataset-type Real --shape-type scissor_output --nmasks 2 --njoints 1 --partial 0 model --rotation-range 120 --joint-type r --prob-threshold 0.05 --rigid-cd-w 0.5 --color-cd-w 0

## Testing
python run_art.py train --num-iterations 20000 --resume-path <The_Path_of_PTH_File> experiment --seed 1234 --experiment-id Any_Name_You_like --run-mode test equi_settings --dataset-type Real --shape-type basket_output --nmasks 3 --njoints 2 --partial 0 model --rotation-range 120 --joint-type r --prob-threshold 0.05 --rigid-cd-w 0.5 --color-cd-w 0

python run_art.py train --num-iterations 20000 --resume-path <The_Path_of_PTH_File> experiment --seed 1234 --experiment-id Any_Name_You_like --run-mode test equi_settings --dataset-type Real --shape-type drawer_output --nmasks 2 --njoints 1 --partial 0 model --rotation-range 120 --joint-type p --prob-threshold 0.05 --rigid-cd-w 0.5 --color-cd-w 0

python run_art.py train --num-iterations 20000 --resume-path <The_Path_of_PTH_File> experiment --seed 1234 --experiment-id Any_Name_You_like --run-mode test equi_settings --dataset-type Real --shape-type laptop_output --nmasks 2 --njoints 1 --partial 0 model --rotation-range 120 --joint-type r --prob-threshold 0.05 --rigid-cd-w 0.5 --color-cd-w 0

python run_art.py train --num-iterations 20000 --resume-path <The_Path_of_PTH_File> experiment --seed 1234 --experiment-id Any_Name_You_like --run-mode test equi_settings --dataset-type Real --shape-type suitcase_output --nmasks 2 --njoints 1 --partial 0 model --rotation-range 120 --joint-type r --prob-threshold 0.05 --rigid-cd-w 0.5 --color-cd-w 0

python run_art.py train --num-iterations 20000 --resume-path <The_Path_of_PTH_File> experiment --seed 1234 --experiment-id Any_Name_You_like --run-mode test equi_settings --dataset-type Real --shape-type scissor_output --nmasks 2 --njoints 1 --partial 0 model --rotation-range 120 --joint-type r --prob-threshold 0.05 --rigid-cd-w 0.5 --color-cd-w 0