#!/bin/bash
# Reset Database Script
# Drops and recreates all analysis result tables while preserving source data

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Database Reset Script ===${NC}"
echo "This script will drop and recreate all analysis result tables."
echo "Source data (news_articles) will be preserved."
echo ""

# Read database configuration from config.yaml
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
CONFIG_FILE="$PROJECT_DIR/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: config.yaml not found at $CONFIG_FILE${NC}"
    exit 1
fi

# Parse database configuration from YAML using Python
DB_CONFIG=$(python3 -c "
import yaml
import sys

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

db = config['database']
print(f\"{db['host']}|{db['port']}|{db['name']}|{db['schema']}|{db['user']}\")
")

IFS='|' read -r DB_HOST DB_PORT DB_NAME DB_SCHEMA DB_USER <<< "$DB_CONFIG"

echo "Database: $DB_NAME"
echo "Schema: $DB_SCHEMA"
echo "Host: $DB_HOST:$DB_PORT"
echo "User: $DB_USER"
echo ""

# Confirmation prompt
read -p "Are you sure you want to drop all result tables? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo -e "${YELLOW}Aborted.${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}Dropping result tables...${NC}"

# Drop tables in correct order (respecting foreign key constraints)
PGPASSWORD=$(python3 -c "import yaml; config = yaml.safe_load(open('$CONFIG_FILE')); print(config['database']['password'])") \
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS $DB_SCHEMA.article_clusters CASCADE;
DROP TABLE IF EXISTS $DB_SCHEMA.event_clusters CASCADE;
DROP TABLE IF EXISTS $DB_SCHEMA.article_analysis CASCADE;
DROP TABLE IF EXISTS $DB_SCHEMA.embeddings CASCADE;
DROP TABLE IF EXISTS $DB_SCHEMA.topics CASCADE;
DROP TABLE IF EXISTS $DB_SCHEMA.result_versions CASCADE;
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Tables dropped successfully${NC}"
else
    echo -e "${RED}✗ Failed to drop tables${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Creating tables from schema...${NC}"

# Apply schema.sql
PGPASSWORD=$(python3 -c "import yaml; config = yaml.safe_load(open('$CONFIG_FILE')); print(config['database']['password'])") \
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$PROJECT_DIR/schema.sql"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Tables created successfully${NC}"
else
    echo -e "${RED}✗ Failed to create tables${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Database reset complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Create a result version:"
echo "   python3 -c \"from src.versions import create_version; print(create_version('baseline', 'Default configuration'))\""
echo ""
echo "2. Run the analysis pipelines:"
echo "   # For topic analysis:"
echo "   python3 scripts/topics/01_generate_embeddings.py --version-id <uuid>"
echo "   python3 scripts/topics/02_discover_topics.py --version-id <uuid>"
echo ""
echo "   # For clustering analysis:"
echo "   python3 scripts/clustering/01_generate_embeddings.py --version-id <uuid>"
echo "   python3 scripts/clustering/02_cluster_events.py --version-id <uuid>"
echo ""
