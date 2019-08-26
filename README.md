# RAKE_keywords

This is an implementation of the RAKE algorithm for automatic keyword extraction from documents. The Rake class operates at a document level and was heavily influenced by the rake-nltk package (https://github.com/csurfer/rake-nltk). The RakeSummary class extends this to a document level.

Provided a list of documents, RakeSummary will perform RAKE to determine candidate and extracted keywords and will then for each keyword will calculate its 'essentiality' and 'exclusivity' to the the entire corpus. These metrics are described in the Rose et al. (2010) paper.