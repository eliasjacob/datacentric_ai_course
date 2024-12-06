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
 - For GPU support:
    ```shell
    poetry install --sync -E cuda --with cuda
    poetry shell
    ```
    
- For CPU-only support:
    ```shell
    poetry install --sync -E cpu
    poetry shell
    ```
3. **Authenticate Weights & Biases**:

    ```shell
    wandb login
    ```

## Using VS Code Dev Containers

This repository is configured to work with Visual Studio Code Dev Containers, providing a consistent and isolated development environment. To use this feature:

1. Install [Visual Studio Code](https://code.visualstudio.com/) and the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.

2. Clone this repository to your local machine (if you haven't already):

3. Open the cloned repository in VS Code.

4. When prompted, click "Reopen in Container" or use the command palette (F1) and select "Remote-Containers: Reopen in Container".

5. VS Code will build the Docker container and set up the development environment. This may take a few minutes the first time.

6. Once the container is built, you'll have a fully configured environment with all the necessary dependencies installed.

Using Dev Containers ensures that all course participants have the same development environment, regardless of their local setup. It also makes it easier to manage dependencies and avoid conflicts with other projects.


## Teaching Approach

The course will use a **top-down** teaching method, which is different from the traditional **bottom-up** approach. 

- **Top-Down Method**: We'll start with a high-level overview and practical application, then delve into the underlying details as needed. This approach helps maintain motivation and provides a clearer picture of how different components fit together.
- **Bottom-Up Method**: Typically involves learning individual components in isolation before combining them into more complex structures, which can sometimes lead to a fragmented understanding.

### Example: Learning Baseball
Harvard Professor David Perkins, in his book [Making Learning Whole](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719), compares learning to playing baseball. Kids don't start by memorizing all the rules and technical details; they begin by playing the game and gradually learn the intricacies. Similarly, in this course, you'll start with practical applications and slowly uncover the theoretical aspects.

> **Important**: Don't worry if you don't understand everything initially. Focus on what things do, not what they are. 

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
