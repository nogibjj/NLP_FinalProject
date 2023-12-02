"""Demonstrate Naive Bayes classification."""
from UnigramModel import UnigramModel
from ProcessEmail import process_email_body_simple
from NaiveBayesEmailClassifier import NaiveBayesEmailClassifier
import pandas as pd


# define a function that returns vocabulary_map
def get_vocabulary_map(vocabulary):
    vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}

    return vocabulary_map


def main():
    # Load the data
    df = pd.read_csv("/workspaces/NLP_finalProject/data/completeSpamAssassin.csv")

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

    vocabulary = sorted(
        set(token for sentence in sample_spam + sample_nonspam for token in sentence)
    ) + [None]

    # define vocabulary_map
    vocabulary_map = get_vocabulary_map(vocabulary)

    # Initialize the classifier with the vocabulary map
    classifier = NaiveBayesEmailClassifier(vocabulary_map)

    # Assume UnigramModel is defined elsewhere
    # Train the classifier
    training_data, testing_data = classifier.split_data(sample_spam, sample_nonspam)
    classifier.train_naive_bayes(training_data, UnigramModel)

    # Test the classifier
    num_correct = 0
    for document, label in testing_data:
        guess = classifier.predict(document)
        # print(guess, label)  # Optional: Print prediction and actual label

        if guess == label:
            num_correct += 1

    # Print the accuracy
    print(f"Accuracy: {num_correct / len(testing_data)}")


if __name__ == "__main__":
    main()
