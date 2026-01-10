# Community Dataset Groups

This directory contains the community-contributed dataset groups page for Minari documentation.

## Files

- `community.yaml` - Source data file containing all community dataset groups

## How to Add a Dataset Group

To add your dataset group to the community page:

1. Open a PR editing `docs/datasets/community/community.yaml`
2. Add a new entry with the following structure:

```yaml
- dataset_group: hf://your-username/your-group-name  # Full path to dataset group
  display_name: Your Dataset Group Name              # Human-readable name
```

3. Run the generator script to preview your changes (optional):
   ```bash
   python docs/_scripts/gen_community.py
   ```

## Required Fields

- `dataset_group`: Full path to the dataset group (e.g., `hf://username/group-name`)
- `display_name`: Human-readable name for the dataset group

## Notes

- All datasets within the specified group will be automatically listed on the group's page
- Each dataset in your group should have proper metadata (description, env_spec, etc.)
