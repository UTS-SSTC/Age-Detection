# Age Detection

## Setup Environment

**Prerequisite**

- Python 3.11.9
- Anaconda
- VSCode
- Git
- NVIDIA Graphics
- NVIDIA CUDA Toolkit

**Steps**

1. Clone the Age-Detection repository from GitHub to your local machine<br>`git clone https://github.com/UTS-SSTC/Age-Detection.git`
2. Navigate into the cloned repository directory<br>`cd Age-Detection`
3. Create a new Conda environment named "Age-Detection" with Python 3.11.9<br>`conda create -n "Age-Detection" python=3.11.9`
4. Activate the newly created Conda environment<br>`conda activate Age-Detection`
5. Install all the required dependencies specified in the requirements.txt file<br>`pip install -r requirements/requirements-normal.txt`<br>`pip install -r requirements/requirements-pytorch.txt`
6. Open the entire folder in VSCode for development
7. Note: If you need to use TensorFlow's GPU acceleration, you must run the entire project in WSL2 to get support, as TensorFlow's GPU acceleration has been deprecated on Windows.

## Rules

**Commit Messages - Conventional Commits**

The commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

1. **fix:** a commit of the *type* `fix` patches a bug in your codebase (this correlates with [`PATCH`](http://semver.org/#summary) in Semantic Versioning).
2. **feat:** a commit of the *type* `feat` introduces a new feature to the codebase (this correlates with [`MINOR`](http://semver.org/#summary) in Semantic Versioning).
3. **BREAKING CHANGE:** a commit that has a footer `BREAKING CHANGE:`, or appends a `!` after the type/scope, introduces a breaking API change (correlating with [`MAJOR`](http://semver.org/#summary) in Semantic Versioning). A BREAKING CHANGE can be part of commits of any *type*.
4. *types* other than `fix:` and `feat:` are allowed, for example [@commitlint/config-conventional](https://github.com/conventional-changelog/commitlint/tree/master/@commitlint/config-conventional) (based on the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)) recommends `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others.
5. *footers* other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to [git trailer format](https://git-scm.com/docs/git-interpret-trailers).

Additional types are not mandated by the Conventional Commits specification, and have no implicit effect in Semantic Versioning (unless they include a BREAKING CHANGE). A scope may be provided to a commit’s type, to provide additional contextual information and is contained within parenthesis, e.g., `feat(parser): add ability to parse arrays`.