"""Demonstrate Naive Bayes classification."""
import random
from typing import List, Tuple, Mapping, Optional, Sequence
import numpy as np
from UnigramModel import UnigramModel
from ProcessEmail import process_email_body_simple
import pandas as pd


class NaiveBayesEmailClassifier:
    def __init__(self, vocabulary_map):
        self.vocabulary_map = vocabulary_map
        self.h0_language_model = None
        self.h1_language_model = None
        self.ph0 = None
        self.ph1 = None

    def onehot(self, token: Optional[str]) -> np.ndarray:
        """Generate the one-hot encoding for the provided token."""
        embedding = np.zeros((len(self.vocabulary_map), 1))
        idx = self.vocabulary_map.get(token, len(self.vocabulary_map) - 1)
        embedding[idx, 0] = 1
        return embedding

    def encode_document(self, tokens: Sequence[Optional[str]]) -> List[np.ndarray]:
        """Apply one-hot encoding to each document."""
        return [self.onehot(token) for token in tokens]

    def assemble_data(
        self, sample_spam: List[str], sample_nonspam: List[str], test_percent: int = 10
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Assemble the training and testing data from spam and non-spam samples."""
        h0_observations = [
            (self.encode_document(sentence), 0) for sentence in sample_spam
        ]
        h1_observations = [
            (self.encode_document(sentence), 1) for sentence in sample_nonspam
        ]

        all_data = h0_observations + h1_observations
        random.shuffle(all_data)

        break_idx = round(len(all_data) * test_percent / 100)
        return all_data[break_idx:], all_data[:break_idx]

    def train_naive_bayes(self, training_data, UnigramModel):
        """Train the Naive Bayes model using the training data."""
        h0_documents = [
            observation[0] for observation in training_data if observation[1] == 0
        ]
        h1_documents = [
            observation[0] for observation in training_data if observation[1] == 1
        ]

        self.h0_language_model = UnigramModel(len(self.vocabulary_map))
        self.h0_language_model.train(
            [token for document in h0_documents for token in document]
        )

        self.h1_language_model = UnigramModel(len(self.vocabulary_map))
        self.h1_language_model.train(
            [token for document in h1_documents for token in document]
        )

        self.ph0 = len(h0_documents) / len(training_data)
        self.ph1 = len(h1_documents) / len(training_data)

    def predict(self, document):
        """Predict the label of a given document."""
        h0_logp_unnormalized = self.h0_language_model.apply(document) + np.log(self.ph0)
        h1_logp_unnormalized = self.h1_language_model.apply(document) + np.log(self.ph1)

        logp_data = np.logaddexp(h0_logp_unnormalized, h1_logp_unnormalized)
        h0_logp = h0_logp_unnormalized - logp_data
        h1_logp = h1_logp_unnormalized - logp_data

        pc0 = np.exp(h0_logp)
        pc1 = np.exp(h1_logp)
        return 1 if pc1 > pc0 else 0


completeSpam = pd.read_csv("/workspaces/NLP_finalProject/data/completeSpamAssassin.csv")
# delete space
df = pd.DataFrame(completeSpam)

# Splitting the dataframe into two based on the label
spam_df = df[df["Label"] == 1]
non_spam_df = df[df["Label"] == 0]

# Applying the simplified processing function to each group
spam_emails = spam_df["Body"].apply(process_email_body_simple)
non_spam_emails = non_spam_df["Body"].apply(process_email_body_simple)

# Compile the processed emails into separate lists for spam and non-spam
spam_list = [line for email in spam_emails for line in email]
non_spam_list = [line for email in non_spam_emails for line in email]

# Display the first few elements of the compiled simple list to verify the structure
sample_spam = spam_list[:800]  # 300
sample_nonspam = non_spam_list[:1000]  # 500
# print(sample_spam)
vocabulary = sorted(
    set(token for sentence in sample_spam + sample_nonspam for token in sentence)
) + [None]
vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}


def main():
    # Initialize the classifier with the vocabulary map
    classifier = NaiveBayesEmailClassifier(vocabulary_map)

    # Assume UnigramModel is defined elsewhere
    # Train the classifier
    training_data, testing_data = classifier.assemble_data(sample_spam, sample_nonspam)
    classifier.train_naive_bayes(training_data, UnigramModel)

    # Test the classifier
    num_correct = 0
    for document, label in testing_data:
        guess = classifier.predict(document)
        print(guess, label)  # Optional: Print prediction and actual label

        if guess == label:
            num_correct += 1

    # Print the accuracy
    print(f"Accuracy: {num_correct / len(testing_data)}")


if __name__ == "__main__":
    main()
