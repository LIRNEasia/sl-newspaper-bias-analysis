# Database Migrations

## Model Storage Migration

### What Changed

BERTopic models are now stored in the PostgreSQL database for team collaboration, rather than only on local filesystems.

**Benefits:**
- ✅ Team members can view visualizations from any machine
- ✅ Models automatically shared via the database
- ✅ No manual file copying needed
- ✅ Backward compatible with existing filesystem models

### Running the Migration

**Step 1: Update the database schema**

```bash
# Replace with your actual credentials
psql -h localhost -U your_db_user -d your_database -f migrations/add_model_storage.sql
```

**Step 2: Verify the column was added**

```bash
psql -h localhost -U your_db_user -d your_database -c "\d media_bias.result_versions"
```

You should see a `model_data` column of type `bytea`.

**Step 3: Test with a new pipeline run**

```bash
# Create a test version
python3 -c "
from src.versions import create_version, get_default_topic_config
version_id = create_version('test-db-storage', 'Testing database model storage', get_default_topic_config(), analysis_type='topics')
print(f'Version ID: {version_id}')
"

# Run the pipeline (will save to both filesystem AND database)
python3 scripts/topics/01_generate_embeddings.py --version-id <version-id>
python3 scripts/topics/02_discover_topics.py --version-id <version-id>
```

**Step 4: Verify model saved to database**

```sql
SELECT id, name,
       CASE WHEN model_data IS NULL THEN 'No model in DB'
            ELSE pg_size_pretty(length(model_data)::bigint)
       END as model_size
FROM media_bias.result_versions
WHERE name = 'test-db-storage';
```

You should see something like "7.5 MB" in the model_size column.

**Step 5: Test dashboard on a different machine**

1. On a different machine (or delete local `models/` directory)
2. Start the dashboard: `streamlit run dashboard/app.py`
3. Go to Topics tab → Select the test version
4. Scroll to "Topic Model Visualizations"
5. Visualizations should load successfully (from database)

### How It Works

#### Pipeline Behavior

When you run `scripts/topics/02_discover_topics.py`:

1. BERTopic model is trained
2. Model saved to `models/bertopic_model_{version_id[:8]}/` (filesystem)
3. Model also compressed as tar.gz and saved to database
4. Both saves happen automatically - no flags needed

#### Dashboard Behavior

When you view the Topics tab:

1. Dashboard tries to load model from database first
2. If not in database, falls back to filesystem
3. If neither location has the model, shows a message
4. Streamlit caches the loaded model in memory

#### Backward Compatibility

- **Old versions** (created before migration): Work as before, load from filesystem
- **New versions** (created after migration): Automatically stored in database
- **No breaking changes**: Everything still works if you don't run the migration

### Troubleshooting

**Q: Migration fails with "column already exists"**

A: The migration script uses `IF NOT EXISTS`, so it's safe to run multiple times. If it still fails, the column might already exist from a previous attempt.

**Q: Model saves to filesystem but not database**

A: Check the pipeline output for warnings. Common causes:
- Database connection issues
- Permissions issues on the `result_versions` table
- Model directory doesn't exist when save_model_to_version() is called

**Q: Dashboard shows "model not found" even after pipeline runs**

A: Check if the model was actually saved:
```sql
SELECT name, model_data IS NOT NULL as has_model
FROM media_bias.result_versions
WHERE id = '<version-id>';
```

If `has_model` is false, the database save failed. Check pipeline logs.

**Q: Database getting too large**

A: Each model is ~6-8 MB compressed. With 20 versions, that's ~140 MB. To clean up:

```sql
-- Delete old versions (this cascades to model_data)
DELETE FROM media_bias.result_versions
WHERE created_at < NOW() - INTERVAL '30 days'
AND name NOT IN ('baseline', 'production');  -- Keep important versions

-- Or just clear model_data but keep the version
UPDATE media_bias.result_versions
SET model_data = NULL
WHERE created_at < NOW() - INTERVAL '30 days';
```

### Migration Rollback

If you need to rollback:

```sql
ALTER TABLE media_bias.result_versions
DROP COLUMN IF EXISTS model_data;
```

Dashboard will automatically fall back to filesystem-only loading.
