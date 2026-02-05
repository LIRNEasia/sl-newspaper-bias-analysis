#!/usr/bin/env python3
"""
CLI tool for managing result versions.

Usage:
  python3 scripts/manage_versions.py list [--type topics|clustering|summarization|word_frequency]
  python3 scripts/manage_versions.py delete <version_id>
  python3 scripts/manage_versions.py stats <version_id>
"""

import sys
import os
import argparse
from typing import Optional
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.versions import (
    list_versions,
    get_version,
    get_version_statistics,
    delete_version_interactive
)


def list_versions_cmd(analysis_type: Optional[str] = None):
    """List all versions with their details."""
    versions = list_versions(analysis_type=analysis_type)

    if not versions:
        print("No versions found.")
        return

    print("\n" + "="*100)
    print(f"RESULT VERSIONS{f' ({analysis_type})' if analysis_type else ''}")
    print("="*100)
    print(f"{'ID':<38} {'Name':<20} {'Type':<15} {'Complete':<10} {'Created':<20}")
    print("-"*100)

    for v in versions:
        complete = "✓ Yes" if v['is_complete'] else "✗ No"
        created = v['created_at'].strftime('%Y-%m-%d %H:%M') if hasattr(v['created_at'], 'strftime') else str(v['created_at'])
        print(f"{v['id']:<38} {v['name']:<20} {v['analysis_type']:<15} {complete:<10} {created:<20}")

    print("="*100)
    print(f"\nTotal: {len(versions)} version(s)")

    # Show pipeline status for incomplete versions
    incomplete = [v for v in versions if not v['is_complete']]
    if incomplete:
        print(f"\nIncomplete versions ({len(incomplete)}):")
        for v in incomplete:
            status = v.get('pipeline_status', {})
            steps = []
            for step, complete in status.items():
                icon = "✓" if complete else "✗"
                steps.append(f"{step}:{icon}")
            print(f"  {v['name']}: {' | '.join(steps)}")


def show_stats_cmd(version_id: str):
    """Show detailed statistics for a version."""
    version = get_version(version_id)
    if not version:
        print(f"❌ Version not found: {version_id}")
        return

    stats = get_version_statistics(version_id)

    print("\n" + "="*60)
    print("VERSION DETAILS")
    print("="*60)
    print(f"ID: {version['id']}")
    print(f"Name: {version['name']}")
    print(f"Analysis Type: {version['analysis_type']}")
    print(f"Description: {version['description'] or '(none)'}")
    print(f"Complete: {'Yes' if version['is_complete'] else 'No'}")
    print(f"Created: {version['created_at']}")
    print(f"Updated: {version['updated_at']}")

    print(f"\n{'='*60}")
    print("STATISTICS")
    print("="*60)
    print(f"  Embeddings: {stats['embeddings']:,}")
    print(f"  Topics: {stats['topics']:,}")
    print(f"  Article Analyses: {stats['article_analysis']:,}")
    print(f"  Event Clusters: {stats['event_clusters']:,}")
    print(f"  Article-Cluster Mappings: {stats['article_clusters']:,}")
    print(f"  Article Summaries: {stats['article_summaries']:,}")
    print(f"\n  TOTAL RECORDS: {sum(stats.values()):,}")
    print("="*60)

    # Show pipeline status
    if not version['is_complete']:
        print("\nPIPELINE STATUS:")
        status = version.get('pipeline_status', {})
        for step, complete in status.items():
            icon = "✓" if complete else "✗"
            print(f"  {icon} {step}")


def delete_version_cmd(version_id: str):
    """Delete a version with interactive confirmation."""
    delete_version_interactive(version_id)


def main():
    parser = argparse.ArgumentParser(
        description="Manage result versions for media bias analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all versions
  python3 scripts/manage_versions.py list

  # List only topic versions
  python3 scripts/manage_versions.py list --type topics

  # Show detailed stats for a version
  python3 scripts/manage_versions.py stats <version-id>

  # Delete a version (with confirmation)
  python3 scripts/manage_versions.py delete <version-id>
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List command
    list_parser = subparsers.add_parser('list', help='List all versions')
    list_parser.add_argument(
        '--type',
        choices=['topics', 'clustering', 'summarization', 'word_frequency', 'combined'],
        help='Filter by analysis type'
    )

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show version statistics')
    stats_parser.add_argument('version_id', help='Version ID (UUID)')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a version')
    delete_parser.add_argument('version_id', help='Version ID (UUID)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'list':
        list_versions_cmd(analysis_type=args.type)
    elif args.command == 'stats':
        show_stats_cmd(args.version_id)
    elif args.command == 'delete':
        delete_version_cmd(args.version_id)


if __name__ == '__main__':
    main()
