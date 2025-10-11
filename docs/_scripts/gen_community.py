import os
import pathlib

import yaml


DATASET_FOLDER = pathlib.Path(__file__).parent.parent.joinpath("datasets")


def _rst_escape(text: str) -> str:
    """Escape text for RST."""
    return text.replace("*", r"\*").replace("_", r"\_").replace("`", r"\`")


# -- Generate community page -------------------------------------------------
def generate_community_page(
    yaml_path=DATASET_FOLDER.joinpath("community", "community.yaml"),
    out_rst=DATASET_FOLDER.joinpath("community", "index.rst"),
):
    if not os.path.exists(yaml_path):
        print(f"YAML file not found: {yaml_path}")
        return

    with open(yaml_path) as f:
        community_data = yaml.safe_load(f) or []

    content = "Community Datasets\n"
    content += "==================\n\n"
    content += "Below is a list of datasets contributed by the community. "
    content += "To add yours, open a PR editing ``docs/datasets/community/community.yaml``.\n\n"

    content += ".. raw:: html\n\n"
    content += '    <div class="sphx-glr-thumbnails">\n\n'

    for dataset_entry in community_data:
        dataset_id = dataset_entry.get(
            "dataset_id", ""
        )  # e.g., "username/dataset-name"
        display_name = _rst_escape(dataset_entry.get("display_name", dataset_id))
        description = _rst_escape(dataset_entry.get("description", ""))

        # Build HuggingFace URL
        hf_url = f"https://huggingface.co/datasets/{dataset_id}" if dataset_id else ""

        # Thumb card - wrap entire card in clickable link
        content += ".. raw:: html\n\n"
        tooltip = description if description else display_name
        content += f'    <div class="sphx-glr-thumbcontainer" tooltip="{tooltip}">\n'
        content += f'      <a class="sphx-glr-thumbcontainer-link" href="{hf_url}">\n\n'

        content += ".. only:: html\n\n"
        # Default to minari-text.png
        img_src = "/_static/img/minari-text.png"
        content += f"  .. image:: {img_src}\n"
        content += f"    :alt: {display_name}\n\n"

        content += ".. raw:: html\n\n"
        content += f'      <div class="sphx-glr-thumbnail-title">{display_name}</div>\n'
        content += "      </a>\n"
        content += "    </div>\n\n"

    content += ".. raw:: html\n\n"
    content += "    </div>\n\n"

    os.makedirs(os.path.dirname(out_rst), exist_ok=True)
    with open(out_rst, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    generate_community_page()


if __name__ == "__main__":
    main()
