# [FSE'25] Software Fairness Dilemma: Is Bias Mitigation a Zero-Sum Game?

Welcome to the homepage of our FSE'25 paper "Software Fairness Dilemma: Is Bias Mitigation a Zero-Sum Game?". The homepage contains data, scrips, and intermediate results used in this paper.

## Experimental environment

We use Python 3.7 for our experiments. We use the IBM AI Fairness 360 (AIF360) toolkit to implement bias mitigation methods and compute fairness metrics. 

Installation instructions for Python 3.7 and AIF360 can be found at https://github.com/Trusted-AI/AIF360. That page provides several ways for the installation. We recommend creating a virtual environment for it (as shown below), because AIF360 requires specific versions of many Python packages which may conflict with other projects on your system. If you want to try other installation ways or encounter any errors during the installation process, please refer to the page (https://github.com/Trusted-AI/AIF360) for help.

#### Conda

Conda is recommended for all configurations. [Miniconda](https://conda.io/miniconda.html)
is sufficient (see [the difference between Anaconda and
Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda)
if you are curious) if you do not already have conda installed.

Then, to create a new Python 3.7 environment, run:

```bash
conda create --name aif360 python=3.7
conda activate aif360
```

The shell should now look like `(aif360) $`. To deactivate the environment, run:

```bash
(aif360)$ conda deactivate
```

The prompt will return to `$ `.

Note: Older versions of conda may use `source activate aif360` and `source
deactivate` (`activate aif360` and `deactivate` on Windows).

### Install with `pip`

To install the latest stable version from PyPI, run:

```bash
pip install 'aif360'
```

[comment]: <> (This toolkit can be installed as follows:)

[comment]: <> (```)

[comment]: <> (pip install aif360)

[comment]: <> (```)

[comment]: <> (More information on installing AIF360 can be found on https://github.com/Trusted-AI/AIF360.)

In addition, we require the following Python packages. Note that TensorFlow is only required for implementing the existing fairness improvement method named ADV. If you do not want to implement this method, you can skip the installation of TensorFlow (the last line of the following commands).
```
pip install sklearn
pip install numpy
pip install shapely
pip install matplotlib
pip install "tensorflow >= 1.13.1, < 2"
pip install --upgrade protobuf==3.20.0
pip install fairlearn
```

## Dataset

You can refer to https://github.com/Trusted-AI/AIF360/tree/master/aif360/data to download the raw data files.


## Scripts and results

You can reproduce all the results based on the intermediate results provided by us.

* ```Analysis_code/``` contains the scripts for producing the results for all RQs. You can reproduce all the results based on the intermediate results provided by us by running ```rq1.py```, ```rq2.py```, ```rq3.py```, ```rq4_1.py```, and ```rq4_2.py```.

* ```Results_Single/``` contains the raw results of applying each bias mitigation method to single-attribute tasks. Each file in this folder has 22 columns, with the first column indicating the metric, the next 20 columns the metric values of 20 runs, and the last column indicating the mean value of the 20 runs.

*  ```Results_Multiple/``` contains the raw results of applying each bias mitigation method to multiple-attribute tasks. Each file in this folder has 22 columns, with the first column indicating the metric, the next 20 columns the metric values of 20 runs, and the last column indicating the mean value of the 20 runs.e metric, and the next 20 columns the metric values of 20 runs.

You can also replicate the results from scratch using the datasets provided above and the following code of each bias mitigation method.

* ```FairMethod_Single/``` contains the scripts for implementing each bias mitigation method for single-attribute tasks.

* ```FairMethod_Multiple/``` contains the scripts for implementing each bias mitigation method for multiple-attribute tasks.

* ```Discussion/``` contains the scripts and data to replicate the results in the Discussion section.


* ```aif360.zip``` contains the scripts (provided by [Chen et al.](https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness) at ICSE 2024) of adapting REW, ADV, and EOP to make them applicable to multiple protected attributes.

## Citation
Please consider citing the following paper when using our code or data.
```
@inproceedings{zpFSE2025,
  title={Software Fairness Dilemma: Is Bias Mitigation a Zero-Sum Game?},
  author={Zhenpeng Chen and Xinyue Li and Jie M. Zhang and Weisong Sun and Ying Xiao and Tianlin Li and Yiling Lou and Yang Liu},
  booktitle={Proceedings of the ACM International Conference on the Foundations of Software Engineering, FSE 2025},
  year={2025}
}
```
