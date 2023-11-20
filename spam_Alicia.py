"""Demonstrate Naive Bayes classification."""
import random
from typing import List, Mapping, Optional, Sequence
import nltk
import numpy as np
from numpy.typing import NDArray
import math

FloatArray = NDArray[np.float64]

# define model
class UnigramModel:
    """The unigram language model."""

    def __init__(self, size: int) -> None:
        """Initialize."""
        self.size = size
        self.p: Optional[FloatArray] = None

    def train(self, encodings: List[FloatArray]) -> "UnigramModel":
        """Train the model on data."""
        counts = np.ones((self.size, 1))
        for encoding in encodings:
            counts += encoding
        self.p = counts / counts.sum()
        return self

    def apply(self, encodings: List[FloatArray]) -> float:
        """Compute the log probability of a document."""
        if self.p is None:
            raise ValueError("This model is untrained")
        return (
            np.hstack(encodings).sum(axis=1, keepdims=True).T @ np.log(self.p)
        ).item()

completeSpam = pd.read_csv("completeSpamAssassin.csv")
# 删除空行
df = pd.DataFrame(completeSpam)
df.dropna(subset=["Body"], inplace=True)
## 之后删除
df = df.loc[0:50]
# 拆分为以每一行为单位的嵌套list，内里由逗号分开
df["lines"] = df["Body"].apply(lambda email: email.split("\n"))
spam = df[df["Label"] == 1]["lines"]
ham = df[df["Label"] == 0]["lines"]

vocabulary = sorted(
    set(token for sentence in spam + ham for token in sentence)
) + [None]
vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}


def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary_map), 1))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx, 0] = 1
    return embedding


def encode_document(tokens: Sequence[Optional[str]]) -> List[FloatArray]:
    """Apply one-hot encoding to each document."""
    encodings = [onehot(vocabulary_map, token) for token in tokens]
    return encodings


# assemble training and testing data
h0_observations = [(encode_document(sentence), 0) for sentence in spam]
h1_observations = [(encode_document(sentence), 1) for sentence in ham]
all_data = h0_observations + h1_observations
random.shuffle(all_data)
test_percent = 10
break_idx = round(test_percent / 100 * len(all_data))
training_data = all_data[break_idx:]
testing_data = all_data[:break_idx]

# train Naive Bayes
h0_documents = [observation[0] for observation in training_data if observation[1] == 0]
h1_documents = [observation[0] for observation in training_data if observation[1] == 1]
h0_language_model = UnigramModel(len(vocabulary_map))
h0_language_model.train([token for document in h0_documents for token in document])
h1_language_model = UnigramModel(len(vocabulary_map))
h1_language_model.train([token for document in h1_documents for token in document])
ph0 = len(h0_documents) / len(training_data)
ph1 = len(h1_documents) / len(training_data)

num_correct = 0
for document, label in testing_data:
    # apply model for each class
    h0_logp_unnormalized = h0_language_model.apply(document) + np.log(ph0)
    h1_logp_unnormalized = h1_language_model.apply(document) + np.log(ph1)

    # normalize
    logp_data = np.logaddexp(h0_logp_unnormalized, h1_logp_unnormalized)
    h0_logp = h0_logp_unnormalized - logp_data
    h1_logp = h1_logp_unnormalized - logp_data

    # make guess
    pc0 = np.exp(h0_logp)
    pc1 = np.exp(h1_logp)
    guess = 1 if pc1 > pc0 else 0
    print(pc0, pc1, guess, label)

    if guess == label:
        num_correct += 1

print(num_correct / len(testing_data))
