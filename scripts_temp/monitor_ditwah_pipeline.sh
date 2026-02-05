#!/bin/bash
# Monitor Ditwah Claims Pipeline Progress

OUTPUT_FILE="/tmp/claude-1014/-home-ranushi-Taf-claude-sl-newspaper-bias-analysis/tasks/b798aac.output"

echo "🔍 Monitoring Ditwah Claims Pipeline..."
echo "========================================"
echo ""

# Check if file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "❌ Output file not found: $OUTPUT_FILE"
    exit 1
fi

# Show last 30 lines
echo "📊 Recent Progress:"
echo "-------------------"
tail -30 "$OUTPUT_FILE"

echo ""
echo "========================================"

# Count API calls
API_CALLS=$(grep -c "HTTP/1.1 200 OK" "$OUTPUT_FILE")
echo "✅ Successful API calls: $API_CALLS"

# Count processing messages
PROCESSING=$(grep -c "Processing article" "$OUTPUT_FILE")
echo "📝 Articles processed: $PROCESSING / 1657"

# Calculate percentage
if [ $PROCESSING -gt 0 ]; then
    PERCENT=$((PROCESSING * 100 / 1657))
    echo "📈 Progress: $PERCENT%"
fi

# Check for errors
ERRORS=$(grep -c "ERROR" "$OUTPUT_FILE")
if [ $ERRORS -gt 0 ]; then
    echo "⚠️  Errors found: $ERRORS"
    echo ""
    echo "Recent errors:"
    grep "ERROR" "$OUTPUT_FILE" | tail -5
fi

echo ""
echo "To watch live: tail -f $OUTPUT_FILE"
