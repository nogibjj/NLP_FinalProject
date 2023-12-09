# this code is to generate sythetic email data
import sys
import os
import pandas as pd

# import packages from Generative_Probabilistic_Model
sys.path.append(os.path.abspath(os.path.join("..")))
from Generative_Probabilistic_Model.ProcessEmail import process_email_body_simple

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
# print(sample_spam)
vocabulary = sorted(
    set(token for sentence in sample_spam + sample_nonspam for token in sentence)
) + [None]


# define a function that create dictionaries for spam and non-spam that records the frequency of each word
def create_dict(vocabulary, emails):
    dict = {}
    for word in vocabulary:
        dict[word] = 0
    for lines in emails:
        for word in lines:
            dict[word] += 1
    return dict


# apply the function to spam and non-spam emails
spam_dict = create_dict(vocabulary, sample_spam)
non_spam_dict = create_dict(vocabulary, sample_nonspam)
print(non_spam_dict)
