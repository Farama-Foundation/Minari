# Community Datasets

This directory contains the community-contributed datasets page for Minari documentation.

## Files

- `community.yaml` - Source data file containing all community datasets
- `index.rst` - Generated RST page (auto-generated, do not edit manually)

## How to Add a Dataset

To add your dataset to the community page:

1. Open a PR editing `docs/datasets/community/community.yaml`
2. Add a new entry with the following structure:

```yaml
- dataset_id: your-username/your-dataset-name  # HuggingFace dataset ID
  display_name: Your Dataset Display Name      # Human-readable name
  description: Brief description of your dataset
  author: Your Name or Organization            # Optional
  tags: [tag1, tag2, tag3]                     # Optional
```

4. Run the generator script to preview your changes:
   ```bash
   python docs/_scripts/gen_community.py
   ```

## Required Fields

- `dataset_id`: HuggingFace dataset ID in format `username/dataset-name`
- `display_name`: Human-readable name for the dataset

## Optional Fields

- `description`: Brief description of what the dataset contains
- `author`: Author name or organization
- `tags`: List of tags for categorization (e.g., robotics, atari, continuous-control)

## Generating the Page

The `index.rst` file is auto-generated from `community.yaml` using:

```bash
python docs/_scripts/gen_community.py
```

This script is part of the documentation build process and creates:
- A thumbnail gallery grid (similar to the tutorials page)
- Automatic links to HuggingFace dataset pages

## Development

The generator script is located at `docs/_scripts/gen_community.py`.

Dependencies:
- PyYAML (listed in `docs/requirements.txt`)
