#!/bin/bash

folder_path="./extras"

if [ -d "$folder_path" ]; then
    # Folder exists, delete it
    echo "Deleting folder: $folder_path"
    rm -rf "$folder_path"
    echo "Folder deleted successfully."
fi

conda env remove --name three-gen-validation

