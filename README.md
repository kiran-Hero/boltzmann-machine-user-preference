# Boltzmann Machine for User Preference Analysis

## 1. Project Title

**Developing a Boltzmann Machine Using Binary User-Preference Data**

---

## 2. Problem Statement

Develop a Boltzmann Machine using binary user-preference data.
Analyze hidden-unit activations for different user patterns and
interpret the latent features learned by the model.

---

## 3. Objective

The main objectives of this project are:

- To develop a Restricted Boltzmann Machine (RBM) for binary data.
- To represent user preferences using binary values.
- To train the model to learn hidden patterns in user preferences.
- To analyze hidden-unit activations for different users.
- To identify and interpret latent features learned by the model.
- To visualize the training process and learned representations.

---

## 4. Dataset

The dataset contains binary preferences of users for different
categories.

The following preference features are used:

- Movie
- Music
- Sports
- Gaming
- Travel
- Food
- Shopping
- Technology

The values are represented as:

```text
1 = Preference exists
0 = Preference does not exist

The dataset is stored in:
dataset/user_preferences.csv

5. Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git
- GitHub
6. Model Architecture
This project uses a Restricted Boltzmann Machine (RBM), which is
a simplified form of a Boltzmann Machine.
The model contains:
8 Visible Units
       |
       v
Restricted Boltzmann Machine
       |
       v
4 Hidden Units

Visible Layer
The visible layer represents the user's actual preferences:
Movie
Music
Sports
Gaming
Travel
Food
Shopping
Technology

Hidden Layer
The hidden layer learns underlying patterns from combinations
of the visible user preferences.
The hidden units are not manually assigned labels. Their meanings
are interpreted after training by analyzing their activations and
learned weights.
7. Methodology
The project follows these steps:
1. Create binary user-preference data.
2. Load the dataset using Pandas.
3. Convert the preference data into numerical values.
4. Initialize the RBM weights and biases.
5. Calculate hidden-unit probabilities.
6. Sample the hidden units.
7. Reconstruct the visible layer.
8. Update weights and biases using Contrastive Divergence.
9. Repeat the training process for multiple epochs.
10. Calculate hidden-unit activations for each user.
11. Analyze the learned weights.
12. Interpret the latent features.
13. Visualize the results.
8. RBM Training
The model is trained using Contrastive Divergence.
During training, the RBM performs the following process:
User Preference Data
        |
        v
Visible Layer
        |
        v
Hidden Unit Probabilities
        |
        v
Hidden Unit Sampling
        |
        v
Reconstruct Visible Layer
        |
        v
Calculate Reconstruction Error
        |
        v
Update Weights and Biases

This process is repeated for multiple epochs so that the model
can learn useful hidden representations.
9. Hidden Unit Activation Analysis
After training, hidden-unit probabilities are calculated for
each user.
For example:
User    H1     H2     H3     H4
U01    0.82   0.21   0.76   0.15
U02    0.18   0.87   0.29   0.72

A higher value means that the corresponding hidden unit is
more strongly activated for that user's preference pattern.
The hidden activations are used to compare different users
and identify similar preference patterns.
10. Latent Feature Analysis
The learned weights between the visible and hidden layers are
analyzed to identify the preferences most strongly associated
with each hidden unit.
For example, if a hidden unit has strong positive weights for:
Movie
Music
Gaming

it may represent an entertainment-related latent feature.
Similarly, if another hidden unit has strong associations with:
Sports
Travel

it may represent an outdoor or travel-related latent feature.
These interpretations are obtained from the learned model
rather than being manually assigned before training.
11. Results
11.1 Training Loss
The model records reconstruction error during training.
The training-loss graph is stored at:
results/training_loss.png

A decreasing reconstruction error indicates that the model is
learning to reconstruct the input preference patterns.
11.2 Hidden Unit Activations
The hidden-unit activation heatmap is stored at:
results/hidden_activations.png

This visualization shows the activation strength of H1, H2,
H3 and H4 for each user.
It helps identify which hidden patterns are strongly activated
for different users.
11.3 Latent Features
The learned latent features are visualized in:
results/latent_features.png

This heatmap shows the relationship between user-preference
features and the hidden units.
The strongest learned weights are used to interpret the
possible meaning of each hidden unit.
11.4 Hidden Activation Values
The numerical hidden-unit activation values are saved in:
results/hidden_activation_values.csv

This file contains the activation values of H1, H2, H3 and H4
for every user.
12. Project Structure
boltzmann-machine-user-preference/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── dataset/
│   └── user_preferences.csv
│
├── src/
│   └── boltzmann_machine.py
│
├── notebooks/
│   └── BM_Analysis.ipynb
│
├── results/
│   ├── training_loss.png
│   ├── hidden_activations.png
│   ├── latent_features.png
│   └── hidden_activation_values.csv
│
└── screenshots/
    ├── training_output.png
    ├── hidden_activation_output.png
    └── latent_feature_output.png

13. How to Run the Project
Step 1: Clone the repository
git clone https://github.com/kiran-Hero/boltzmann-machine-user-preference.git

Step 2: Open the project
cd boltzmann-machine-user-preference

Step 3: Create a virtual environment
python3 -m venv .venv

Step 4: Activate the virtual environment
For macOS/Linux:
source .venv/bin/activate

Step 5: Install the required libraries
pip install -r requirements.txt

Step 6: Run the Boltzmann Machine
python src/boltzmann_machine.py

14. Output Files
After successful execution, the following files are generated:
results/training_loss.png
results/hidden_activations.png
results/latent_features.png
results/hidden_activation_values.csv

These files contain the training results, hidden-unit
activations and learned latent features.
15. Conclusion
The Restricted Boltzmann Machine successfully learns hidden
representations from binary user-preference data.
The hidden-unit activations provide information about different
user patterns, while the learned weights help identify latent
features associated with combinations of user preferences.
The project demonstrates how a Boltzmann Machine can be used
to discover hidden structures in binary preference data.
16. Limitations
- The dataset used in this project is relatively small.
- User preferences are represented using binary values.
- The interpretation of hidden units depends on the learned
  weights and activations.
- More users and preference categories could improve the
  quality of the learned representations.
17. Future Enhancements
The project can be extended by:
- Using a larger real-world user-preference dataset.
- Increasing the number of hidden units.
- Comparing different learning rates.
- Comparing different numbers of training epochs.
- Adding recommendation functionality.
- Using real movie, music or product preference datasets.
- Developing a web interface for user preference analysis.
18. Author
Kiran Saravanan
B.Tech Artificial Intelligence and Machine Learning
Veltech
19. References
1. Hinton, G. E. – A Practical Guide to Training Restricted
   Boltzmann Machines.
2. NumPy Documentation.
3. Pandas Documentation.
4. Matplotlib Documentation.
5. Seaborn Documentation.
6. Scikit-learn Documentation.
