#!/bin/bash
# Monitor Batched Ditwah Claims Pipeline

OUTPUT_FILE="/tmp/claude-1014/-home-ranushi-Taf-claude-sl-newspaper-bias-analysis/tasks/ba51779.output"

echo "🔍 Monitoring Batched Ditwah Claims Pipeline..."
echo "================================================"
echo ""

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "❌ Output file not found"
    exit 1
fi

# Show last 20 lines
echo "📊 Recent Progress:"
echo "-------------------"
tail -20 "$OUTPUT_FILE"

echo ""
echo "================================================"

# Count batches processed
BATCHES=$(grep -c "Batch.*Extracted.*claims" "$OUTPUT_FILE")
echo "✅ Batches completed: $BATCHES / 34"

# Calculate percentage
if [ $BATCHES -gt 0 ]; then
    PERCENT=$((BATCHES * 100 / 34))
    echo "📈 Progress: $PERCENT%"
fi

# Count total claims extracted
TOTAL_CLAIMS=$(grep "Batch.*Extracted.*claims" "$OUTPUT_FILE" | grep -oP '\d+(?= claims)' | awk '{s+=$1} END {print s}')
if [ ! -z "$TOTAL_CLAIMS" ]; then
    echo "📝 Total claims extracted: $TOTAL_CLAIMS"
fi

# Check for completion
if grep -q "Pipeline complete" "$OUTPUT_FILE"; then
    echo "🎉 PIPELINE COMPLETED!"
    echo ""
    grep -A 10 "PIPELINE SUMMARY" "$OUTPUT_FILE"
fi

# Check for errors
ERRORS=$(grep -c "ERROR" "$OUTPUT_FILE")
if [ $ERRORS -gt 0 ]; then
    echo "⚠️  Errors found: $ERRORS"
fi

echo ""
echo "To watch live: tail -f $OUTPUT_FILE"
