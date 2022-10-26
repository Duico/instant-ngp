# How to run the script?

1. Install detectron2: `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`, more info here https://detectron2.readthedocs.io/en/latest/tutorials/install.html

2. Install nuscenes-devkit: `pip install nuscenes-devkit`

3. Install the progress bar: `pip install tqdm`

4. Run the script by passing the nuScenes root directory and the scene token of your interest. The script will create a `masks/` folder with the masks for every single photo of the driving sequence (including all cameras).