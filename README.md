# Learning Modified Indicator Functions for Surface Reconstruction

In this work, we propose a learning-based approach for implicit surface reconstruction from raw point 
clouds without normals. Inspired by Gauss Lemma in potential energy
theory, we design a novel deep neural network to perform surface integral and learn the modified indicator
functions from un-oriented and noisy point clouds. Our method generates smooth surfaces with high normal consistency. Our implementation is based on [Points2Surf](https://github.com/ErlerPhilipp/points2surf). 


## Dependencies
Our work requires Python>=3.7, Pytorch>=1.6 and CUDA>=10.2. To build all the dependencies, execute the following command:
``` bash
pip install -r requirements.txt
``` 
## Start and Test
To generate Fig. 1 to Fig. 12 in our work, execute the following command:
``` bash
sh run_grsi.sh
```
The results will be placed in ./results/{model_name}/{dataset_name}/rec/mesh after the execution is completed. 
It takes hundreds of seconds for generating a shape on average, depending on your environments (about 200s with test batchsize 500 on Tesla V100 GPUs).

To generate Fig. 13, execute the following command:
``` bash
sh run_sparse.sh
```
This procedure of this example is long because we need large query threshold for sparse samplings.

## Models and Datasets
You can download all the models and datasets of our work from [here](https://cloud.tsinghua.edu.cn/d/4cbf3ace009442149090/). 
To conduct different experiments, you need 
to match the prefixes and modelpostfixes of .sh files in ./experiments. We also put some examples in this folder.
The prefix 'lmi' is used for the experiments in Section 5.2 and 5.4. 
The Prefixes 'lmi_ablation' and 'lmi_no_sef' are used for Section 5.3. 
The Prefixes 'lmi_holes' and 'lmi_sparse' are used for Section 5.5. 


## Train
Since the training set is large, we seperate it into four volumes named ABC.zip, ABC.z01, ABC.z02 and ABC.z03.
You need to download all of them and merge them with the following command in Linux (or directly unzip ABC.zip in Windows).
``` bash
zip ABC.zip ABC.z01 ABC.z02 ABC.z03 -s=0 --out ABC_train.zip
```
Then you can unzip the merged file and put them into ./datasets.
``` bash
unzip ABC_train.zip
```
Execute the following command to train.
``` bash
sh train.sh
```
You can choose an appropriate batchsize for training according to your environment. For example, you can set it to 600 for 4 RTX 2080Ti GPUs.


## Citation
```
```
