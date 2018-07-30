# text_cat

A proof-of-concept exploration project to assess the viability of the 'spaCy' Python library for natural language processing.
Scripts have been modified and won't run out of the box (for now).

## Summary
The objective is to take advantage of a data set containing longform, prosaic text documents to bring value to a hypothetical client. 

The proposal was to first implement a text classifier to sift out sets of relevant documents, categorised into self-defined topics. A named-entity recogniser is then trained to allow for the ability to identify specific keywords, and to adapt the language model to the context of the document (in this case to the Singaporean context). From there, some information extraction can be performed to deliver only the most relevant snippets of the documents to the client.

The project was primarily exploratory in nature, with the goal being to identify an appropriate language, library and framework to perform the aforementioned tasks. Hence, progres on the project was optimised for speed rather than comprehensiveness. Multiple languages were explored: R, Python and Java; their respective libraries were also tested: OpenNLP, scikit-learn, spaCy, StanfordNLP, etc. 

Ultimately, Python was the language of choice for its balance between ease of transitioning to more general programming, and the suite of analytics-related libraries. spaCy became the preferred library as the production-level nature of the library was advantageous.