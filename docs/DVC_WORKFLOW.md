# DVC Dataset Versioning Workflow

## Tools
- Python
- Git
- DVC
- Pandas
- NumPy
- Scikit-learn

## DVC Remote
myremote - Local folder remote

## Dataset Versioning Workflow

1. Generate or obtain dataset.
2. Save dataset inside data/raw/.
3. Track dataset using dvc add.
4. Stage the .dvc file using git add.
5. Commit Version 1.
6. Upload using dvc push.
7. Modify dataset by adding 20 synthetic rows.
8. Track updated dataset using dvc add.
9. Commit Version 2.
10. Upload using dvc push.
11. Compare versions using dvc diff.
12. Restore historical versions using git checkout and dvc checkout.