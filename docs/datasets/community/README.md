# Community Datasets

This directory contains the community-contributed datasets page for Minari documentation.

## Files

- `community.yaml` - Source data file containing all community datasets

## How to Add a Dataset

To add your dataset to the community page:

1. Open a PR editing `docs/datasets/community/community.yaml`
2. Add a new entry with the following structure:

```yaml
- dataset_id: your-username/your-dataset-name  # HuggingFace dataset ID
  display_name: Your Dataset Display Name      # Human-readable name
```

4. Run the generator script to preview your changes (optional):
   ```bash
   python docs/_scripts/gen_community.py
   ```

## Required Fields

- `dataset_id`: HuggingFace dataset ID in format `username/dataset-name`
- `display_name`: Human-readable name for the dataset
