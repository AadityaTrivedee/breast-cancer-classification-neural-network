# breast-cancer-classification-neural-network

## Key facts

- **Breast cancer caused 670,000 deaths globally in 2022.**
- **Roughly half of all breast cancers occur in women with no specific risk factors other than sex and age.**
- **Breast cancer was the most common cancer in women in 157 countries out of 185 in 2022.**
- **Breast cancer occurs in every country in the world.**
- **Approximately 0.5–1% of breast cancers occur in men.**

---

## Overview

Breast cancer is a disease in which abnormal breast cells grow out of control and form tumours. If left unchecked, the tumours can spread throughout the body and become fatal.

Breast cancer cells begin inside the milk ducts and/or the milk-producing lobules of the breast. The earliest form (in situ) is not life-threatening and can be detected in early stages. Cancer cells can spread into nearby breast tissue (invasion). This creates tumours that cause lumps or thickening.

Invasive cancers can spread to nearby lymph nodes or other organs (metastasize). Metastasis can be life-threatening and fatal.

## Scope of the problem

In 2022, there were 2.3 million women diagnosed with breast cancer and 670,000 deaths globally. Breast cancer occurs in every country of the world in women at any age after puberty but with increasing rates in later life.

Global estimates reveal striking inequities in the breast cancer burden according to human development. For instance, in countries with a very high Human Development Index (HDI), 1 in 12 women will be diagnosed with breast cancer in their lifetime and 1 in 71 women die of it.

In contrast, in countries with a low HDI; while only 1 in 27 women is diagnosed with breast cancer in their lifetime, 1 in 48 women will die from it.

## Who is at risk?

Female gender is the strongest breast cancer risk factor. Approximately 99% of breast cancers occur in women and 0.5–1% of breast cancers occur in men.

## The Model

The key challenge against its detection are how to classify tumours into malignant (cancerous) or benign(non-cancerous). This project utilises a single hidden layer neural network to predict the probability of malignant breast cancer cells in women. The classification works on 0s and 1s, 0 being Begingn and 1 being malignant. The data used in this set is extracted from Kaggle ( Wisconsin Breast Cancer). There are 9 features to predict the dataset:

| Clump Thickness: Assessment of the thickness of tumour cell clusters (1 - 10).        |
| ------------------------------------------------------------------------------------ |
| Uniformity of Cell Size: Uniformity in the size of tumour cells (1 - 10).             |
| Uniformity of Cell Shape: Uniformity in the shape of tumour cells (1 - 10).           |
| Marginal Adhesion: Degree of adhesion of tumour cells to surrounding tissue (1 - 10). |
| Single Epithelial Cell Size: Size of individual tumour cells (1 - 10).                |
| Bare Nuclei: Presence of nuclei without surrounding cytoplasm (1 - 10).              |
| Bland Chromatin: Assessment of chromatin structure in tumour cells (1 - 10).          |
| Normal Nucleoli: Presence of normal-looking nucleoli in tumour cells (1 - 10).        |
| Mitoses: Frequency of mitotic cell divisions (1 - 10).                               |
| Class: Classification of tumour type (0 for benign, 1 for malignant).                 |

The model utilises a hidden layer with 6 neurons featured by the activation function of tanh for better prediction of the model. It then predicts the hypothesis through the sigmoid function. The model is traditionally made with all formulas and procedures self-written. It includes graphing the data too. The model has a raw and rounded-off training dataset accuracy of around 99% each and a testing set of 98%. The model is optimised for a learning rate of 0.000086 and a number of iterations of 20000. The model also allows you to predict your own dataset along with its own raw accuracy data for confirmation.

### Architecture

![architecture.png](https://res.craft.do/user/full/f816ce7c-43c2-6c9f-cde1-0f4c21033df4/doc/f2b2eca4-1499-4a11-9aca-499fb486b013/75294a29-61f3-41f3-a2c1-04ae8e99ca23)

## Sources:

[Breast cancer](https://www.who.int/news-room/fact-sheets/detail/breast-cancer)


