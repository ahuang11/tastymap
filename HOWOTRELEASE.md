`git tag v0.4.1 --message 'Bump version 0.4.0 -> 0.4.1'`
`git push --tags`
`python3 -m build`
`twine upload dist/*`

Create release on GitHub.
