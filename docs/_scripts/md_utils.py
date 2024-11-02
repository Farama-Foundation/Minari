from typing import Dict


def dict_to_table(table_dict: Dict[str, str]) -> str:
    markdown = "|    |    |\n |----|----|"
    for key, value in table_dict.items():
        markdown += f"\n| {key} | {value} |"
    return markdown
