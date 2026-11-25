# Job-Generator

The `job-generator` creates all jobs for GitLab CI.
It takes multiple compiler and software versions as input and generates a test matrix by [pairwise combination](https://en.wikipedia.org/wiki/All-pairs_testing).
When you run the `job-generator` process, the output is a valid GitLab CI YAML file for CI.

The `job-generator` uses the [bashi](https://github.com/alpaka-group/bashi/) library, which provides the functions for creating combinations.

`bashi` is designed for use with alpaka-based projects such as alpaka itself, as well as projects that uses alpaka, such as [PIConGPU](https://github.com/ComputationalRadiationPhysics/picongpu/).
Therefore, `bashi` provides a set of rules that define what is technically possible for alpaka-based applications.
For example which nvcc version supports which gcc version.
The `job-generator`, on the other hand, adds its own rules.
Most of these rules reduce the testing effort, e.g., by allowing only a subset of possible backend combinations to reduce the number of test jobs.
In addition to extent the rule set, the `job-generator` implements everything that happens after the combination list is generated.
Most of the functionality involves generating the GitLab CI YAML code from the combination list.

## Naming

The `job-generator` uses `bashi`'s naming convention.
[Here](https://github.com/alpaka-group/bashi/blob/main/docs/naming.md) you can find out what the terms mean.

# Install

The `job-generator` is a Python package and is very easy to install.

```bash
# create a virtual environment with your favorite tool, then
pip3 install .
```

# Usage

Run `job-generator <container_version>` to generate the YAML file for the GitLab CI job.

```bash
job-generator 4.1
```

Use `job-generator --help` to display additional options.

## GPU pinning

Each HIP job exports `ROCR_VISIBLE_DEVICES` as `index % ALPAKA_CI_GPU_COUNT` to pin one GPU per job.
`ALPAKA_CI_GPU_COUNT` defaults to 8 and can be overridden per runner, `ALPAKA_CI_GPU_INDEX` overrides the index for a single job.
On ROCm, do not combine this with `HIP_VISIBLE_DEVICES`.

# alpaka-validate

`alpaka-validate` is a tool that can be used to check whether a combination is supported by alpaka.
It is automatically installed along with `job-generator`.

Run `alpaka-validate --help` to display all the parameters that can be checked.

![alpaka-validate](./docs/alpaka-validate.png)

# Verification

When combining parameters in pairs, the goal is to generate as few combinations as possible while ensuring that every pair of parameter-values appears in at least one combination.
Due to the filtering rules, not every combination of two parameter-values is allowed.
The `job-generator` automatically checks whether all valid pairs appear at least once in a combination and ensures that no invalid pairs occur.

# Development

If you want to modify the `job-generator` code, you can install it in an editable version.
This means that the `job-generator` command will run your modified code.

```bash
# create a virtual environment with your favorite tool, then
pip3 install --editable .
```

## Modify bashi

If you want to modify `bashi`, you must enable the Python environment for `job-generator`, clone the [bashi repository](https://github.com/alpaka-group/bashi), and run the command `pip3 install --editable .` in the `bashi` repository.
After that, `job-generator` will access the source code in the local repository, and you can edit it.

## Add new Software Versions

New software versions can be added to [versions.py](./src/alpaka_bashi/versions.py).
Depending on the software, new combinations are generated without any issues.

In the case of an issue, there are two possibilities.

1. There is already a filter rule that handles this case in principle, and the `verify()` function already removes the invalid parameter-value pairs. In this case, all that is needed is to add a new entry to the corresponding version dependency list. For example, the latest CUDA SDK only supports GCC up to version X.
2. The new version creates an invalid relationship that is not yet covered by a filter rule.

To determine which scenario applies, review the existing filter rules:

- The [bashi filter-chain](https://github.com/alpaka-group/bashi/blob/main/src/bashi/filter_chain.py) shows you which filter stages are called in what order.
- The first filter stages have been implemented in `bashi`. The following filters are used (the links may be outdated):
    - [compiler filter](https://github.com/alpaka-group/bashi/blob/main/src/bashi/filter_compiler.py)
    - [backend filter](https://github.com/alpaka-group/bashi/blob/main/src/bashi/filter_backend.py)
    - [software dependencies filter](https://github.com/alpaka-group/bashi/blob/main/src/bashi/filter_software_dependency.py)
- The [custom filter](./src/alpaka_bashi/alpaka_filter.py) is implemented here in the `alpaka-bashi` project.

**Tip:** Most filter rules do not include a comment explaining how they work.
Read the string in the function call `self.reason()` before the `return False` to get a summary of the filter rule.

If a filter is available, read the [next section](#extend-an-existing-rule) to learn how to expand it.
If no filter is available, please also read the [next section](#extend-an-existing-rule) to understand how filters work in general, and then read the [section that follows](#writing-a-new-rule).

In all cases, [alpaka-validate](#alpaka-validate) is useful for checking whether your changes work.

### Extend an existing Rule

The key object for retrieving version relationships, such as which GCC version is supported by a specific CUDA version, is the [VersionRelation object](https://github.com/alpaka-group/bashi/blob/main/src/bashi/version/relation.py) object.
Filter rules and the validation of generated combinations use this object to retrieve version relationships.
The object’s constructor allows you to override version relationships.

For example, we want to specify that CUDA 13.4 does not support GCC 16.
First, we need to find the filter rule.
The rule is `Rule: c5` in the [bashi compiler filter](https://github.com/alpaka-group/bashi/blob/main/src/bashi/filter_compiler.py).

```python
if row[DEVICE_COMPILER].name == NVCC:
    if row[HOST_COMPILER].name == GCC:
        # Rule: c5
        # related to rule b10
        # remove all unsupported nvcc gcc version combinations
        # define which is the latest supported gcc compiler for a nvcc version

        # if a nvcc version is not supported by bashi, assume that the version supports the
        # latest gcc compiler version
        if row[DEVICE_COMPILER].version <= self.version.get_nvcc_gcc_max_version()[0].nvcc:
            # check the maximum supported gcc version for the given nvcc version
            for nvcc_gcc_comb in self.version.get_nvcc_gcc_max_version():
                if row[DEVICE_COMPILER].version >= nvcc_gcc_comb.nvcc:
                    if row[HOST_COMPILER].version > nvcc_gcc_comb.host:
                        self.reason(
                            f"nvcc {row[DEVICE_COMPILER].version} "
                            f"does not support gcc {row[HOST_COMPILER].version}",
                        )
                        return False
                    break
```

The function call `self.version.get_nvcc_gcc_max_version()` accesses a member variable that stores the `VersionRelation` object.
If we look at the implementation of `VersionRelation`, we see that we can set `nvcc_gcc_max_version` in the object's constructor.

`alpaka-bashi` provides a function to modify the structure of the `VersionRelation` object.
The function is `get_alpaka_version_relation()` in [version.py](./src/alpaka_bashi/versions.py).

The following example shows how to extend the `nvcc_gcc_max_version` list.

```python
from bashi.version.dependencies.nvcc import NvccHostSupport, NVCC_GCC_MAX_VERSION

# ...

def get_alpaka_version_relation() -> bashi.VersionRelation:
    """Returns:
    bashi.VersionRelation: bashi.VersionRelation object with alpaka specific modifications.
    """

    # we use NVCC_GCC_MAX_VERSION as base to avoid redefining relations for other CUDA versions
    # define the maximum supported gcc version for a specific nvcc version
    nvcc_gcc_max_version = NVCC_GCC_MAX_VERSION + [
        NvccHostSupport("13.4", "15"),
    ]

    return bashi.VersionRelation(
        nvcc_gcc_max_version=nvcc_gcc_max_version,
    )
```

The `verify()` functions also use the `VersionRelation` object to retrieve the software relationships.
As a result, the list of invalid parameter-value-tuples is automatically expanded.
In our case, `verify()` checks that no combination contains the pair `device compiler nvcc 13.4 + host compiler gcc 16`.

### Writing a new Rule

To add a new restriction for software dependencies, you need to follow a few steps. First, you must add a new filter rule to the [alpaka_filter.py](./src/alpaka_bashi/alpaka_filter.py) file.
Please review the existing filter rules to understand how they work.
`bashi` also provides a [guide](https://github.com/alpaka-group/bashi/blob/main/docs/rules.md) on implementing filter rules.

Once the filter rule is implemented and does not raise an exception of type `covertable.exceptions.InvalidCondition` (see the next section), the `verify()` function should fail and indicate that an invalid pair was found.
Therefore, the `verify()` function in the [verify.py](./src/alpaka_bashi/verify.py) file must be extended.
Here, too, the [bashi implementation](https://github.com/alpaka-group/bashi/tree/main/src/bashi/result_modules) can serve as a guide.

### Python Exception: covertable.exceptions.InvalidCondition

Sometimes, when running the `job-generator`, an exception of type `covertable.exceptions.InvalidCondition` may be triggered when a new software version is added.
This can happen if a filter rule or a software rule is missing.
`bashi` provides a [best practices guide](https://github.com/alpaka-group/bashi/blob/main/docs/rules.md) for resolving the issue.

**Attention:** This guide refers to the `example.py` file and the `bashi-validate` tool.
If you are applying this guide to the `job-generator` package, you must use the `job-generator` application instead of `example.py` and `alpaka-validate` instead of `bashi-validate.`

- To display which combinations the `job-generator` accepts and which it does not, run `job-generator --debug-print=args`. Green combinations pass and red ones are invalid. Run `job-generator --help` for more information.
- Use `alpaka-validate` instead of `bashi-validate`. `alpaka-validate` uses the `bashi` filters as well as the alpaka-specific filter.

## Update the Verification

In most cases, the verification functions adapt to the new software version, and verification passes automatically. However, sometimes existing verification rules need to be adjusted or new ones added. The verification rules are defined in the file [verify.py](./src/alpaka_bashi/verify.py).

This [guide](https://github.com/alpaka-group/bashi/blob/main/docs/remove-parameter-value-pairs.md) explains how the utils functions work to define which parameter-value pairs are valid and which are invalid.
