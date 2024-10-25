import numpy as np
import pandas as pd
import spacy
import nltk

# Test numpy
data = np.array([1, 2, 3])
print("Numpy array:", data)

# Test pandas
df = pd.DataFrame(data, columns=["Numbers"])
print("Pandas DataFrame:\n", df)

# Test NLTK
nltk.download('punkt')
text = "Hello, world!"
tokens = nltk.word_tokenize(text)
print("NLTK Tokens:", tokens)

# Test SpaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is an email sorting agent test.")
for ent in doc.ents:
    print(ent.text, ent.label_)
