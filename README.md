# Multiclass Classification with ECOC and Label Switching Correction (LSEnsemble)

This repository implements a robust multiclass classification framework using Error-Correcting Output Codes (ECOC) with a specialized focus on mitigating the impact of label switching problems. It features the LSEnsemble algorithm, designed to improve classification accuracy in the presence of label noise or ambiguity, particularly relevant in complex datasets.

## Project Overview

Error-Correcting Output Codes (ECOC) decompose multiclass problems into multiple binary classification sub-problems, which often suffer from severe class imbalance.

LSEnsemble addresses this issue by introducing label switching, a technique that generates diverse base learners and rebalances the binary sub-problems. The rebalancing factor is selected to simplify learning for base classifiers, particularly in highly imbalanced scenarios.

In an imbalanced space, accurately estimating the likelihood ratio and the underlying a posteriori probabilities is challenging for base learners. LSEnsemble mitigates this by transforming the problem through label switching, which can be combined with other neutral rebalancing strategies (e.g., cost-sensitive learning or population adjustments). This transformation produces a more balanced representation, improving the estimation of the transformed likelihood ratio.

Additionally, the optimal Bayes thresholds for the rebalanced problem can be computed, leading to improved classification performance in the original problem space.

For further details on likelihood ratio estimation and Bayes threshold transformations, see:
"Optimum Bayesian thresholds for rebalanced classification problems using class-switching ensembles," Pattern Recognition, 2022. [https://doi.org/10.1016/j.patcog.2022.109158](https://doi.org/10.1016/j.patcog.2022.109158)

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
