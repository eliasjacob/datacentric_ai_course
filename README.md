# Datacentric AI - Learning with Limited Labels using Weak Supervision and Uncertainty-Aware Training

## [Dr. Elias Jacob de Menezes Neto](https://docente.ufrn.br/elias.jacob)

This repository contains the code and resources for the course **"Learning with Limited Labels using Weak Supervision and Uncertainty-Aware Training"**. The course goes deep into advanced techniques for training machine learning models effectively when labeled data is scarce or noisy, focusing on data-centric AI approaches, weak supervision methodologies, semi-supervised learning strategies, and annotation error detection mechanisms.

## Course Content

The course is structured into several modules, each focusing on specific aspects of learning with limited labels. Below is an overview of the topics covered, along with keypoints and takeaways from each module.

### Theoretical Introduction to Data-Centric AI and Weakly Supervised Learning
- Data-Centric AI paradigm
- Principles of Data-Centric AI
- Weak supervision techniques
- Types of weak supervision
- Aggregation of multiple labeling sources

### Semi-supervised and Positive Unlabeled Learning
- Semi-supervised learning approaches
- Self-training, co-training, and multi-view learning
- Label propagation
- Positive Unlabeled (PU) learning
- Elkan and Noto approach to PU learning

### Weak Supervision Pipeline
- Labeling Functions (LFs)
- Label Model
- Integration with Semi-Supervised Learning
- Snorkel Framework
- Evaluation metrics and comparison with fully supervised learning

### Advanced Topics in Weak Supervision - Name Entity Recognition
- Named Entity Recognition (NER) using weak supervision
- Skweak Framework
- Document-Level Labeling
- Transfer Learning in NER tasks
- Iterative refinement of labeling functions

### Annotation Error Detection
- Types of label noise
- Label noise transition matrix
- Retagging techniques
- Confident Learning for identifying mislabeled instances

### Confident Learning and Cleanlab
- Confident Learning methodology
- Cleanlab library
- Application to various data modalities
- Extension to multi-label classification
- Handling model miscalibration

### Advanced Label Models
- Snorkel MeTaL
- Generative Model
- Flying Squid
- Dawid-Skene
- Hyper Label Model
- CrowdLab

### Influence Functions 
- Influence functions for model interpretation
- Source-Aware Influence Functions

### Active Learning
- Active Learning strategies
- Uncertainty sampling
- Query by Committee
- Diversity sampling


## Prerequisites

To get started with the course, ensure you have the following:

- **Access to a Machine with a GPU**: Recommended for computationally intensive tasks; alternatively, use Google Colab.
- **Installation of Poetry**: For managing Python dependencies. Install it [here](https://python-poetry.org/docs/). (`pip install poetry`)
- **Weights & Biases Account**: For experiment tracking and visualization. Sign up [here](https://wandb.ai/).

## Installation

Follow these steps to set up the environment and dependencies:

1. **Clone the Repository**:

    ```shell
    git clone https://github.com/eliasjacob/datacentric_ai_course.git
    cd datacentric_ai_course
    ```

2. **Install Dependencies**:

    ```shell
    poetry install
    poetry shell
    ```

3. **Authenticate Weights & Biases**:

    ```shell
    wandb login
    ```

## Getting Started

Once the environment is set up, you can start exploring the course materials, running code examples, and working on the practical exercises.

### Notes

- Some parts of the code may require a GPU for efficient execution. If you don't have access to a GPU, consider using Google Colab.

## Teaching Approach

The course employs a **top-down** teaching method, starting with high-level overviews and practical applications before delving into underlying details. This approach helps maintain motivation and provides a clearer picture of how different components fit together.

### Learning Methods

1. **Hands-On Coding**: Engage actively in coding exercises and projects.
2. **Explaining Concepts**: Articulate your understanding by writing about what you've learned or helping peers.

You'll be encouraged to follow along with coding exercises and explain your learning to others. Summarizing key points as the course progresses will also be part of the learning process.

## Final Project

Your final project will be evaluated based on several criteria:

- **Technical Quality**: How well you implement the project.
- **Creativity**: The originality of your approach.
- **Usefulness**: The practical value of your project.
- **Presentation**: How effectively you present your project.
- **Report**: The clarity and thoroughness of your report.

### Project Guidelines

- **Individual Work**: The project must be done individually.
- **Submission**: Submit a link to a GitHub repository or shared folder with your code, data, and report. Use virtual environments and `requirements.txt` to facilitate running your code.
- **Deadline**: The project will be due 15 days after the end of the course.
- **Submission Platform**: Submit your project using the designated platform (e.g., SIGAA).

## Contributing

Contributions to the course repository are welcome! Follow these steps to contribute:

1. **Fork the Repository**: Click on the "Fork" button at the top right of the repository page.
2. **Create a New Branch**:

    ```shell
    git checkout -b feature/YourFeature
    ```

3. **Make Your Changes**: Implement your feature or fix.
4. **Commit Your Changes**:

    ```shell
    git commit -m 'Add some feature'
    ```

5. **Push to the Branch**:

    ```shell
    git push origin feature/YourFeature
    ```

6. **Create a Pull Request**: Go to your fork on GitHub and click the "New pull request" button.

## Contact

For any questions or feedback regarding the course materials or repository, you can [contact me](mailto:elias.jacob@ufrn.br).
