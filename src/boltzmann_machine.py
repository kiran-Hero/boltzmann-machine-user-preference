import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/user_preferences.csv")

user_ids = data["User"].values
preferences = data.drop("User", axis=1)

feature_names = preferences.columns.tolist()
X = preferences.values.astype(float)

print("Dataset loaded successfully!")
print("Number of users:", X.shape[0])
print("Number of preferences:", X.shape[1])
print("\nFirst 5 users:")
print(data.head())


# ==========================================
# 2. RESTRICTED BOLTZMANN MACHINE
# ==========================================

class RBM:

    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        # Weight matrix
        self.weights = np.random.normal(
            0,
            0.01,
            (n_visible, n_hidden)
        )

        # Visible and hidden biases
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def hidden_probability(self, visible):
        """
        Calculate probability of hidden units being active.
        """
        activation = np.dot(visible, self.weights) + self.hidden_bias
        return self.sigmoid(activation)

    def visible_probability(self, hidden):
        """
        Calculate probability of visible units being active.
        """
        activation = np.dot(hidden, self.weights.T) + self.visible_bias
        return self.sigmoid(activation)

    def sample_hidden(self, visible):
        probability = self.hidden_probability(visible)
        return (probability > np.random.rand(*probability.shape)).astype(float)

    def sample_visible(self, hidden):
        probability = self.visible_probability(hidden)
        return (probability > np.random.rand(*probability.shape)).astype(float)

    def train(self, data, epochs=100):

        losses = []

        for epoch in range(epochs):

            # Positive phase
            positive_hidden_prob = self.hidden_probability(data)

            positive_association = np.dot(
                data.T,
                positive_hidden_prob
            )

            # Sample hidden layer
            hidden_sample = self.sample_hidden(data)

            # Negative phase
            negative_visible_prob = self.visible_probability(hidden_sample)

            negative_hidden_prob = self.hidden_probability(
                negative_visible_prob
            )

            negative_association = np.dot(
                negative_visible_prob.T,
                negative_hidden_prob
            )

            # Update weights
            self.weights += self.learning_rate * (
                positive_association - negative_association
            ) / data.shape[0]

            # Update biases
            self.visible_bias += self.learning_rate * np.mean(
                data - negative_visible_prob,
                axis=0
            )

            self.hidden_bias += self.learning_rate * np.mean(
                positive_hidden_prob - negative_hidden_prob,
                axis=0
            )

            # Reconstruction error
            loss = np.mean(
                (data - negative_visible_prob) ** 2
            )

            losses.append(loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- Reconstruction Error: {loss:.4f}"
                )

        return losses


# ==========================================
# 3. CREATE RBM
# ==========================================

np.random.seed(42)

n_visible = X.shape[1]
n_hidden = 4

rbm = RBM(
    n_visible=n_visible,
    n_hidden=n_hidden,
    learning_rate=0.1
)


# ==========================================
# 4. TRAIN MODEL
# ==========================================

print("\nStarting RBM training...\n")

losses = rbm.train(
    X,
    epochs=100
)

print("\nTraining completed!")


# ==========================================
# 5. PLOT TRAINING LOSS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(losses)

plt.title("RBM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Error")

plt.tight_layout()

plt.savefig(
    "results/training_loss.png",
    dpi=300
)

plt.show()


# ==========================================
# 6. CALCULATE HIDDEN ACTIVATIONS
# ==========================================

hidden_activations = rbm.hidden_probability(X)

hidden_df = pd.DataFrame(
    hidden_activations,
    columns=["H1", "H2", "H3", "H4"]
)

hidden_df.insert(0, "User", user_ids)

print("\nHidden Unit Activations:")
print(hidden_df.to_string(index=False))


# ==========================================
# 7. HIDDEN ACTIVATION HEATMAP
# ==========================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    hidden_activations,
    xticklabels=["H1", "H2", "H3", "H4"],
    yticklabels=user_ids,
    annot=True,
    fmt=".2f"
)

plt.title("Hidden Unit Activations for Users")
plt.xlabel("Hidden Units")
plt.ylabel("Users")

plt.tight_layout()

plt.savefig(
    "results/hidden_activations.png",
    dpi=300
)

plt.show()


# ==========================================
# 8. ANALYZE LATENT FEATURES
# ==========================================

weights_df = pd.DataFrame(
    rbm.weights,
    index=feature_names,
    columns=["H1", "H2", "H3", "H4"]
)

print("\nLearned Feature Weights:")
print(weights_df)


# ==========================================
# 9. LATENT FEATURE HEATMAP
# ==========================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    weights_df,
    annot=True,
    fmt=".2f",
    center=0
)

plt.title("Learned Latent Features")
plt.xlabel("Hidden Units")
plt.ylabel("User Preferences")

plt.tight_layout()

plt.savefig(
    "results/latent_features.png",
    dpi=300
)

plt.show()


# ==========================================
# 10. PRINT STRONGEST FEATURES
# ==========================================

print("\nLatent Feature Interpretation:")

for hidden_unit in weights_df.columns:

    strongest_features = (
        weights_df[hidden_unit]
        .abs()
        .sort_values(ascending=False)
        .head(3)
    )

    print(
        f"\n{hidden_unit} is strongly associated with:"
    )

    for feature in strongest_features.index:
        weight = weights_df.loc[feature, hidden_unit]

        print(
            f"  {feature}: {weight:.3f}"
        )


# ==========================================
# 11. SAVE HIDDEN ACTIVATIONS
# ==========================================

hidden_df.to_csv(
    "results/hidden_activation_values.csv",
    index=False
)

print("\nResults saved successfully!")

print("\nGenerated files:")
print("results/training_loss.png")
print("results/hidden_activations.png")
print("results/latent_features.png")
print("results/hidden_activation_values.csv")