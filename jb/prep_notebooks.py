import nbformat as nbf
from glob import glob

# Collect a list of all notebooks in the content folder
notebooks = glob("*.ipynb")

# Text to look for in adding tags
text_search_dict = {
    "# Solution": "hide-cell",  # Hide the cell with a button to show
}

# Search through each notebook and look for the text, add a tag if necessary
for ipath in notebooks:
    ntbk = nbf.read(ipath, nbf.NO_CONVERT)

    for cell in ntbk.cells:
        cell_tags = cell.get('metadata', {}).get('tags', [])
        cell_tags = []
        for key, val in text_search_dict.items():
            if key in cell['source']:
                if val not in cell_tags:
                    cell_tags.append(val)
        if len(cell_tags) > 0:
            cell['metadata']['tags'] = cell_tags

    nbf.write(ntbk, ipath)
