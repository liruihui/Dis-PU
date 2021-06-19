# Point Cloud Upsampling via Disentangled Refinement

This repository contains a Tensorflow implementation of the paper:

[Point Cloud Upsampling via Disentangled Refinement](http://arxiv.org/abs/2106.04779). 
<br>
[Ruihui Li](https://liruihui.github.io/), 
[Xianzhi Li](https://nini-lxz.github.io/),
[Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/), 
[Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/).
<br>
<br>
CVPR 2021

## Getting Started

1. Clone the repository:

   ```shell
   https://github.com/liruihui/Dis-PU.git
   cd Dis-PU
   ```
   Installation instructions for Ubuntu 16.04:
   * Make sure <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> are installed. Only this configurations has been tested: 
     - Python 3.6.9, TensorFlow 1.11.1
   * Follow <a href="https://www.tensorflow.org/install/pip">Tensorflow installation procedure</a>.
     
2. Compile the customized TF operators by `sh complile_op.sh`. 
   Follow the information from [here](https://github.com/yanx27/PointASNL) to compile the TF operators. 
   
3. Train the model:
    First, you need to download the training patches in HDF5 format from [GoogleDrive](https://drive.google.com/open?id=13ZFDffOod_neuF3sOM0YiqNbIJEeSKdZ) and put it in folder `data`.
    Then run:
   ```shell
   cd code
   python dis-pu.py --phase train
   ```

4. Evaluate the model:
    First, you need to download the pretrained model from [GoogleDrive](https://drive.google.com/file/d/1SL1kcqex6rRrpjRp4fH-6XrVHyy1bYas/view?usp=sharing), extract it and put it in folder 'model'.
    Then run:
   ```shell
   cd code
   python dis-pu.py --phase test
   ```
   You will see the input and output results in the folder `data/test/output`.
   
5. The training and testing mesh files can be downloaded from [GoogleDrive](https://drive.google.com/open?id=1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC).

### Evaluation code
We provide the code to calculate the uniform metric in the evaluation code folder. In order to use it, you need to install the CGAL library. Please refer [this link](https://www.cgal.org/download/linux.html) and  [PU-Net](https://github.com/yulequan/PU-Net) to install this library.
Then:
   ```shell
   cd evaluation_code
   cmake .
   make
   ./evaluation Icosahedron.off Icosahedron.xyz
   ```
The second argument is the mesh, and the third one is the predicted points.

## Citation

If Dis-PU is useful for your research, please consider citing:

    @inproceedings{li2021dispu,
         title={Point Cloud Upsampling via Disentangled Refinement},
         author={Li, Ruihui and Li, Xianzhi and Heng, Pheng-Ann and Fu, Chi-Wing},
         booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
         year = {2021}
     }


### Questions

Please contact 'lirh@cse.cuhk.edu.hk'

