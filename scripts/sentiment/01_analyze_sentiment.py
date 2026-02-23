#!/usr/bin/env python3
"""
Sentiment Analysis Pipeline - Multi-Model

Analyzes sentiment of news articles using multiple models:
- RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)
- DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- FinBERT (ProsusAI/finbert) - optimized for financial news
- VADER (vaderSentiment) - lexicon-based, very fast
- TextBlob - pattern-based, very fast

Usage:
    python scripts/sentiment/01_analyze_sentiment.py [--limit LIMIT] [--models MODEL1 MODEL2 ...]

Options:
    --limit LIMIT          Limit number of articles to process (default: all)
    --models MODELS        Specific models to run (default: all enabled models)
                          Available: roberta, distilbert, finbert, vader, textblob
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path (parent.parent because we're in scripts/sentiment/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db import Database, ditwah_filters
from src.sentiment import get_sentiment_analyzer, get_sentiment_stats
from src.config import load_config


def display_status(db: Database, models: list = None):
    """Display current sentiment analysis status for all models."""
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS STATUS - MULTI-MODEL")
    print("="*70)

    # Total articles
    total = db.get_article_count(filters=ditwah_filters())
    print(f"\nTotal articles in database: {total:,}")

    # Get enabled models from config
    config = load_config()
    enabled_models = models or config['sentiment']['enabled_models']

    print(f"\nEnabled models: {', '.join(enabled_models)}")
    print(f"\n{'Model':<15} {'Analyzed':<12} {'Progress':<12} {'Avg Sentiment':<15} {'Avg Confidence'}")
    print("-" * 70)

    for model in enabled_models:
        stats = db.get_sentiment_stats(model)
        if stats and stats.get("total_analyzed"):
            analyzed = stats["total_analyzed"]
            pct = (analyzed / total) * 100
            avg_sent = stats.get('avg_sentiment', 0)
            avg_conf = stats.get('avg_confidence', 0)

            print(f"{model:<15} {analyzed:>8,} {pct:>10.1f}% {avg_sent:>14.2f} {avg_conf:>14.2f}")
        else:
            print(f"{model:<15} {0:>8} {0:>10.1f}% {0:>14.2f} {0:>14.2f}")

    print("="*70 + "\n")


def run_analysis(limit: int = None, models: list = None):
    """Run sentiment analysis with all enabled models."""
    config = load_config()
    enabled_models = models or config['sentiment']['enabled_models']

    print(f"\nStarting multi-model sentiment analysis...")
    print(f"Enabled models: {', '.join(enabled_models)}")
    print(f"Total models: {len(enabled_models)}\n")

    for model_type in enabled_models:
        print(f"\n{'='*70}")
        print(f"Processing with {model_type.upper()} model")
        print(f"{'='*70}\n")

        # Get articles not yet analyzed by this model
        with Database() as db:
            articles = db.get_articles_without_sentiment(model_type, limit=limit)

        if not articles:
            print(f"✓ All articles already analyzed with {model_type}")
            continue

        print(f"Found {len(articles):,} articles to analyze")

        # Get analyzer for this model
        try:
            analyzer = get_sentiment_analyzer(model_type, config)
        except Exception as e:
            print(f"❌ Error loading {model_type} analyzer: {e}")
            continue

        # Analyze articles
        start_time = time.time()
        try:
            results = analyzer.analyze_batch(articles, show_progress=True)
        except Exception as e:
            print(f"❌ Error during {model_type} analysis: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - start_time

        # Store results in database
        print("\nStoring results in database...")
        with Database() as db:
            result_dicts = [r.to_dict() for r in results]
            db.store_sentiment_analyses(result_dicts)

        # Print statistics
        stats = get_sentiment_stats(results)
        print(f"\n✓ Completed {model_type} analysis:")
        print(f"  - Articles analyzed: {len(results):,}")
        print(f"  - Time elapsed: {elapsed:.1f}s")
        print(f"  - Articles/second: {len(results)/elapsed:.1f}")
        print(f"  - Avg sentiment: {stats['avg_sentiment']:.2f}")
        print(f"  - Avg confidence: {stats['avg_confidence']:.2f}")
        print(f"  - Sentiment distribution:")
        print(f"    • Negative (<-0.5): {stats['negative_count']:,} ({stats['negative_count']/stats['total']*100:.1f}%)")
        print(f"    • Neutral (-0.5 to 0.5): {stats['neutral_count']:,} ({stats['neutral_count']/stats['total']*100:.1f}%)")
        print(f"    • Positive (>0.5): {stats['positive_count']:,} ({stats['positive_count']/stats['total']*100:.1f}%)")

    # Refresh materialized view
    print(f"\n{'='*70}")
    print("Refreshing sentiment summary materialized view...")
    with Database() as db:
        db.refresh_sentiment_summary()
    print("✓ Materialized view refreshed")
    print("\n✓ All sentiment analysis complete!\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze sentiment of news articles using multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of articles to process"
    )
    parser.add_argument(
        "--models",
        nargs='+',
        choices=['roberta', 'distilbert', 'finbert', 'vader', 'textblob'],
        help="Specific models to run (default: all enabled models)"
    )

    args = parser.parse_args()

    # Display current status
    with Database() as db:
        display_status(db, args.models)

    # Run analysis
    try:
        run_analysis(args.limit, args.models)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Display final status
    with Database() as db:
        display_status(db, args.models)


if __name__ == "__main__":
    main()
