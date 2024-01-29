# Minari Contribution Guidelines

We welcome:

- Bug reports
- Pull requests for bug fixes
- Documentation improvements
- Tutorials and tutorial improvements


## Contributing to the codebase

### Coding

Contributing code is done through standard github methods:

1. Fork this repo
3. Commit your code
4. Submit a pull request. It will be reviewed by maintainers and they'll give feedback or make requests as applicable

### Considerations
- Make sure your new code is properly documented, tested and fully-covered
- Any fixes should include fixes to the appropriate documentation

### Git hooks
The CI will run several checks on the new code pushed to the Minari repository. These checks can also be run locally without waiting for the CI by following the steps below:
1. [install `pre-commit`](https://pre-commit.com/#install),
2. install the Git hooks by running `pre-commit install`.

Once those two steps are done, the Git hooks will be run automatically at every new commit. The Git hooks can also be run manually with `pre-commit run --all-files`, and if needed they can be skipped (not recommended) with `git commit --no-verify`. **Note:** you may have to run `pre-commit run --all-files` manually a couple of times to make it pass when you commit, as each formatting tool will first format the code and fail the first time but should pass the second time.

### Testing Environment
Along with `pre-commit`, CI will be running Unit Tests on the complete code base to ensure all the functionalities are working properly. Unit tests can also be run manually on local machine by following the steps below:

1. To setup the testing environment you need to install optional testing dependencies for minari by running `pip3 install '.[testing]'` from the root directory.
2. After setting up the testing environment, you can run the unit tests using `pytest -v` command. For every new functionality or change in existing functionality, necessary changes are to be made in the respective unit tests.

## Contributing tutorials
Tutorials are a crucial way to help people learn how to use Minari and we greatly appreciate any contributions. However, we have a few guidelines for tutorials:

### Tutorial content
- Tutorials should be written in single .py scripts with clear comments explaining the code and thought process with a naive user in mind.
- Tutorials should be within their own directory within `/tutorials/`, with the directory name being the tutorial theme.
- Tutorials should come with their own **fully versioned** `requirements.txt` file that minimally lists all required dependencies needed to run all .py scripts within the directory.
- The `requirements.txt` file should include `minari`, fixed to the current release (eg. `minari==0.0.1`).

### Adding your tutorials to the docs
- Create the directory `/docs/tutorials/<TUTORIAL_THEME>`.
- You should make a `.md` file for each tutorial within the above directory.
- Each `.md` file should have an "Environment Setup" section and a "Code" section. The title should be of the format `<TUTORIAL_THEME>: <TUTORIAL_TOPIC>`.
- The Environment Setup section should reference the `requirements.txt` file you created using `literalinclude`.
- The Code section should reference the `.py` file you created using `literalinclude`.
- `/docs/index.md` should be modified to include every new tutorial.

### Testing your tutorial
- Tutorial themes should be added to `.github/workflows/linux-tutorials-test.yml`, by adding the theme name to the `tutorial` part of the `matrix`.
- When making a pull request, tests for the newly added tutorials should run. We cannot accept any tutorials that fail any tests, though we can provide assistance in debugging errors - feel free to reach out for advice at any stage!
