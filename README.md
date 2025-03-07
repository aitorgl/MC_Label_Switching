# Multiclass Classification with ECOC and Label Switching Correction (LSEnsemble)

This repository implements a robust multiclass classification framework using Error-Correcting Output Codes (ECOC) with a specialized focus on mitigating the impact of label switching problems. It features the LSEnsemble algorithm, designed to improve classification accuracy in the presence of label noise or ambiguity, particularly relevant in complex datasets.

## Project Overview

Utilizing Error-Correcting Output Codes (ECOC), this project decomposes multiclass problems into binary classification sub-problems, allowing for the effective modeling of complex decision boundaries. The LSEnsemble algorithm introduces label switching as a method to generate diverse base learners and, crucially, to rebalance the binary sub-problems. This rebalancing is designed to make the learning task easier for the base classifiers, especially in scenarios with high class imbalance. The key insight is that in the original imbalanced space, the estimation of the likelihood ratio, and consequently the underlying a posteriori probabilities, becomes challenging for base learners. By transforming the problem through label switching, LSEnsemble generates a new, more balanced representation where the likelihood ratio can be more accurately computed. This improved estimation facilitates better decision-making by the base classifiers, ultimately leading to enhanced performance in the original classification task. Furthermore, the optimal Bayes thresholds of the original binary dichotomous problems can be transformed to the rebalanced problem, as detailed in our work on class-switching ensembles:

Optimum Bayesian thresholds for rebalanced classification problems using class-switching ensembles,
Pattern Recognition, 2022. https://doi.org/10.1016/j.patcog.2022.109158

## Key Features

* **ECOC Decomposition:**
    * Implementation of ECOC for effective multiclass to binary transformation.
    * Flexible ECOC encoding options configurable via `config.yaml`.
* **LSEnsemble Algorithm:**
    * Specialized algorithm for label switching correction, improving model robustness.
    * Configurable optimization parameters for LSEnsemble via `config.yaml`.
* **Comprehensive Evaluation:**
    * Evaluation suite using a range of multiclass metrics: balanced accuracy, Cohen's kappa, geometric mean, sensitivity.
    * Detailed logging and output for performance analysis.
* **Configuration-Driven Design:**
    * `config.yaml` allows for easy customization of datasets, models, and evaluation settings.
    * Streamlined model selection logic based on peak or average performance, configurable via `config.yaml`.
* **Model Persistence:**
    * Trained models and their configurations are saved using pickle for efficient testing and deployment.
    * Naming convention for the saved models that contains the parameters used for LSEnsemble.
* **Testing Script (`test.py`):**
    * Enables evaluation of pre-trained models without retraining, facilitating efficient testing.
    * Read the config.yaml file to load the parameters, and generate the model.
* **Dataset Flexibility:**
    * Designed to handle diverse datasets stored in a designated data folder.

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/franjgs/MC_Label_Switching.git](https://github.com/franjgs/MC_Label_Switching.git)
    cd MC_Label_Switching
    ```

2.  **Configure `config.yaml`:**

    * Adjust settings for datasets, models, and evaluation parameters.

3.  **Run the training script:**

    ```bash
    python train.py
    ```

4.  **Run the testing script:**

    ```bash
    python test.py
    ```

## Dependencies

* Python 3.x
* scikit-learn
* NumPy
* Pandas
* PyYAML

## Datasets

* Place your datasets in the `datasets` folder (or specify the path in `config.yaml`).

## Output

* Trained models and evaluation metrics are saved in the `results` folder.

## Contributing

* Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

* UC3M
