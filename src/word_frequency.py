"""Word frequency analysis for media bias detection."""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from .db import Database, load_config


class WordFrequencyAnalyzer:
    """Compute word frequencies and TF-IDF scores for articles."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration.

        Args:
            config: Word frequency configuration with stopwords, min_word_length, etc.
        """
        self.config = config
        self.min_word_length = config.get("min_word_length", 3)

        # Combine English stopwords with custom stopwords
        custom_stopwords = config.get("custom_stopwords", [])
        self.stopwords = set(ENGLISH_STOP_WORDS) | set(custom_stopwords)

        self.ranking_method = config.get("ranking_method", "frequency")
        self.tfidf_scope = config.get("tfidf_scope", "per_source")
        self.top_n = config.get("top_n_words", 50)

    def preprocess_text(self, text: str) -> str:
        """Preprocess text: lowercase, remove special chars, etc.

        Args:
            text: Raw text to preprocess

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def compute_frequencies_by_source(
        self,
        articles_by_source: Dict[str, List[Dict]]
    ) -> Dict[str, List[Tuple[str, int]]]:
        """Compute word frequencies per source.

        Args:
            articles_by_source: Dict mapping source_id to list of articles

        Returns:
            Dict mapping source_id to list of (word, frequency) tuples
        """
        results = {}

        for source_id, articles in articles_by_source.items():
            # Combine all text for this source
            texts = [
                self.preprocess_text(f"{article.get('title', '')} {article.get('content', '')}")
                for article in articles
            ]

            # Use CountVectorizer for frequency counting
            vectorizer = CountVectorizer(
                stop_words=list(self.stopwords),
                min_df=1,
                token_pattern=r'(?u)\b\w{' + str(self.min_word_length) + r',}\b'
            )

            try:
                # Fit and transform
                counts = vectorizer.fit_transform(texts)

                # Sum across all documents
                word_counts = np.asarray(counts.sum(axis=0)).flatten()

                # Get feature names (words)
                feature_names = vectorizer.get_feature_names_out()

                # Create word-frequency pairs and sort
                word_freq_pairs = list(zip(feature_names, word_counts))
                word_freq_pairs.sort(key=lambda x: x[1], reverse=True)

                # Take top N
                results[source_id] = word_freq_pairs[:self.top_n]

            except ValueError as e:
                # Handle case where no words pass the filters
                print(f"Warning: Could not extract words for source {source_id}: {e}")
                results[source_id] = []

        return results

    def compute_tfidf_per_source(
        self,
        articles_by_source: Dict[str, List[Dict]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Compute TF-IDF scores treating each source separately.

        Args:
            articles_by_source: Dict mapping source_id to list of articles

        Returns:
            Dict mapping source_id to list of (word, tfidf_score) tuples
        """
        results = {}

        for source_id, articles in articles_by_source.items():
            # Combine all text for this source
            texts = [
                self.preprocess_text(f"{article.get('title', '')} {article.get('content', '')}")
                for article in articles
            ]

            # Use TfidfVectorizer
            vectorizer = TfidfVectorizer(
                stop_words=list(self.stopwords),
                min_df=1,
                token_pattern=r'(?u)\b\w{' + str(self.min_word_length) + r',}\b'
            )

            try:
                # Fit and transform
                tfidf_matrix = vectorizer.fit_transform(texts)

                # Average TF-IDF scores across all documents
                avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

                # Get feature names (words)
                feature_names = vectorizer.get_feature_names_out()

                # Create word-score pairs and sort
                word_score_pairs = list(zip(feature_names, avg_tfidf))
                word_score_pairs.sort(key=lambda x: x[1], reverse=True)

                # Take top N
                results[source_id] = word_score_pairs[:self.top_n]

            except ValueError as e:
                # Handle case where no words pass the filters
                print(f"Warning: Could not extract words for source {source_id}: {e}")
                results[source_id] = []

        return results

    def compute_tfidf_cross_source(
        self,
        articles_by_source: Dict[str, List[Dict]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Compute TF-IDF scores across all sources.

        This treats all articles as a single corpus, which allows comparing
        which words are distinctive for each source.

        Args:
            articles_by_source: Dict mapping source_id to list of articles

        Returns:
            Dict mapping source_id to list of (word, tfidf_score) tuples
        """
        # Prepare data: one "document" per source (all articles concatenated)
        sources = []
        source_texts = []

        for source_id, articles in articles_by_source.items():
            sources.append(source_id)
            combined_text = " ".join([
                self.preprocess_text(f"{article.get('title', '')} {article.get('content', '')}")
                for article in articles
            ])
            source_texts.append(combined_text)

        # Use TfidfVectorizer on the corpus
        vectorizer = TfidfVectorizer(
            stop_words=list(self.stopwords),
            min_df=1,
            token_pattern=r'(?u)\b\w{' + str(self.min_word_length) + r',}\b'
        )

        try:
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(source_texts)

            # Get feature names (words)
            feature_names = vectorizer.get_feature_names_out()

            # Extract top words for each source
            results = {}
            for idx, source_id in enumerate(sources):
                # Get TF-IDF scores for this source
                source_scores = tfidf_matrix[idx].toarray().flatten()

                # Create word-score pairs and sort
                word_score_pairs = list(zip(feature_names, source_scores))
                word_score_pairs.sort(key=lambda x: x[1], reverse=True)

                # Take top N
                results[source_id] = word_score_pairs[:self.top_n]

            return results

        except ValueError as e:
            # Handle case where no words pass the filters
            print(f"Warning: Could not extract words across sources: {e}")
            return {source_id: [] for source_id in sources}


def compute_word_frequencies(result_version_id: str, wf_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to compute and store word frequencies.

    Args:
        result_version_id: Version UUID
        wf_config: Word frequency configuration

    Returns:
        Summary statistics
    """
    print(f"Starting word frequency analysis for version {result_version_id}")
    print(f"Configuration: {wf_config}")

    # Initialize analyzer
    analyzer = WordFrequencyAnalyzer(wf_config)

    # Load all articles grouped by source
    print("Loading articles from database...")
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT id, title, content, source_id
                FROM {schema}.news_articles
                WHERE content IS NOT NULL AND content != ''
                ORDER BY source_id, date_posted
            """)
            articles = cur.fetchall()

    # Group by source
    articles_by_source = {}
    for article in articles:
        source_id = article['source_id']
        if source_id not in articles_by_source:
            articles_by_source[source_id] = []
        articles_by_source[source_id].append(article)

    print(f"Loaded {len(articles)} articles from {len(articles_by_source)} sources")
    for source_id, source_articles in articles_by_source.items():
        print(f"  {source_id}: {len(source_articles)} articles")

    # Compute word frequencies based on ranking method
    ranking_method = wf_config.get("ranking_method", "frequency")
    tfidf_scope = wf_config.get("tfidf_scope", "per_source")

    print(f"\nComputing word {ranking_method}...")

    if ranking_method == "frequency":
        word_results = analyzer.compute_frequencies_by_source(articles_by_source)
    elif ranking_method == "tfidf":
        if tfidf_scope == "per_source":
            word_results = analyzer.compute_tfidf_per_source(articles_by_source)
        else:  # cross_source
            word_results = analyzer.compute_tfidf_cross_source(articles_by_source)
    else:
        raise ValueError(f"Invalid ranking_method: {ranking_method}")

    # Prepare data for database storage
    print("\nPreparing data for storage...")
    all_frequencies = []

    for source_id, word_list in word_results.items():
        for rank, (word, score) in enumerate(word_list, 1):
            freq_data = {
                "source_id": source_id,
                "word": word,
                "rank": rank
            }

            if ranking_method == "frequency":
                freq_data["frequency"] = int(score)
                freq_data["tfidf_score"] = None
            else:  # tfidf
                freq_data["frequency"] = 0  # Not available in TF-IDF mode
                freq_data["tfidf_score"] = float(score)

            all_frequencies.append(freq_data)

    # Store in database
    print(f"Storing {len(all_frequencies)} word frequency records...")
    with Database() as db:
        db.store_word_frequencies(all_frequencies, result_version_id)

    # Update pipeline status
    from .versions import update_pipeline_status
    update_pipeline_status(result_version_id, "word_frequency", True)

    # Prepare summary
    summary = {
        "total_articles": len(articles),
        "sources": len(articles_by_source),
        "total_word_records": len(all_frequencies),
        "ranking_method": ranking_method,
        "tfidf_scope": tfidf_scope if ranking_method == "tfidf" else None,
        "top_words_per_source": {
            source_id: [word for word, _ in word_list[:10]]
            for source_id, word_list in word_results.items()
        }
    }

    print("\n=== Word Frequency Analysis Complete ===")
    print(f"Total articles analyzed: {summary['total_articles']}")
    print(f"Sources: {summary['sources']}")
    print(f"Word records stored: {summary['total_word_records']}")
    print(f"\nTop 10 words per source:")
    for source_id, top_words in summary['top_words_per_source'].items():
        print(f"  {source_id}: {', '.join(top_words)}")

    return summary
