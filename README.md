# Physically-based Simulation in Computer Graphics HS2021 - Course Exercises

## Installation

### Git
Before we begin, you must have Git running, a distributed revision control system which you need to handin your assignments as well as keeping track of your code changes. We refer you to the online [Pro Git book](https://git-scm.com/book/en/v2) for more information. There you will also find [instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git]) on how to to install it. On Windows, we suggest using [git for windows](https://git-for-windows.github.io/).



### Cloning the Exercise Repository
Before you are able to clone your private exercise repository, you need to have an active [gitlab@ETH](https://gitlab.ethz.ch/) account. Then you can [fork](https://docs.gitlab.com/ee/gitlab-basics/fork-project.html) this project to create your own private online repository.

In the next step you need to clone it to your local hard drive:
```bash
git clone --recursive https://gitlab.ethz.ch/'Your_Git_Username'/pbs21.git
```
'Your_Git_Username' needs to be replaced accordingly. This can take a moment.

If you already have cloned a repository and now want to load it’s submodules you have to use submodule update.
```bash
git submodule update --init --recursive
```

### Taichi with Conda

We will use [taichi](https://github.com/taichi-dev/taichi) for this course.

Please have a look at https://docs.taichi.graphics/docs/#installation before setting up conda environment. You have to install `libtinfo5` on Ubuntu 19.04++ and [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) on Windows.

```bash
conda env create -f pbs.yaml -n pbs
conda activate pbs
```

If you meet `validation layers requested but not available` issue on Windows, please install [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).


### Update Your Forked Repository

To update your forked repository, check this page: [how-do-i-update-a-github-forked-repository](https://stackoverflow.com/questions/7244321/how-do-i-update-a-github-forked-repository)

Basically, you are required to add our repository as a remote to your own one:
```
git remote add upstream https://gitlab.ethz.ch/cglsim/pbs20.git
```
Then, fetch updates from it:
```
git fetch upstream
```
Lastly, move to your `master` branch and merge updates into yours:
```
git checkout master
git merge upstream/master
```
Note that you need to run the first command *only once* for adding, and the following steps (cmake as well!) should be done again for new updates.