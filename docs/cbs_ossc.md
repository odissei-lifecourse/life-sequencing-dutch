
# Documentation for working on the CBS RA and on the OSSC


## Disk space

This is information from Tom.

- It is cheaper on the OSSC than on the RA
- On RA, the current limit is 100 GB.
- On the OSSC, the default is a couple of TBs. Above some threshold, it costs SBUs, but this cost is lower than the cost on the RA.

## Running slurm scripts on the OSSC

- For running evaluation scripts (`generate_llm_report.sh` and `generate_network_report.sh`),
    - for the network report, 240G memory works and 120G memory is too little.
    - for the LLM report with new embeddings stored in hdf5, evaluating the full dataset takes ~800G memory and 5-8 hours (for evaluating embeddings of 3-4 different models).
        - the `sample` option in `generate_life_course_report` allows to run the report on the first N rows in the embedding data set. For 5Mio persons, this takes ~30minutes and 170GB. For 1Mio persons, it takes ~12minutes and 120GB.
- The work environment is for small jobs, and has a bit less than 40GB memory. In our experience, it's best to use the work environment also through batch jobs.



## Permissions on the OSSC

It goes through [access control lists](https://servicedesk.surf.nl/wiki/pages/viewpage.action?pageId=30660238).
This currently creates constraints on the permissions of directories/files. If user A creates directory `mydir/`, user B in the same project may not automatically have permissions to write/execute code on in `mydir/`. As far as we understand, the user that created `mydir/` needs to use the `setfacl` and `getfacl` to give other project users those permissions.

### Workflow when creating new directories

When a user creates a new directory, they should set the permissions as follows:
```bash
setfacl -R -m g:osscPROJNR:rwx ./mydir # sets permissions recursively on existing files and directories in mydir/
setfacl -R -d --set g:osscPROJNR:rwx ./mydir # sets permissions as a default on directories created inside mydir/
```
where `PROJNR` is the CBS project number, for instance 9424.


### Permissions on older directories

As per ticket SD-74551 on the SURF service desk,
- there should be now a default ACL in which everyone in the group ossc9424 also has read, write, execute permission.
- user ossc9424fhvo is the owner of all files that were previously owned by ossc9424
- the directories `~/Dakota_network`, `~/Tanzir/`, and `~/Life_Course_Evaluation` still are chowned by their respective creators. But `getfacl` returns either `group:rwx` or `group:ossc9424:rwx`, which I understand gives rwx access to the ossc9424 group in both cases.
- there are still some directories for which the group lacks write access, for instance `Network_Embeddings`.




## Working with Git on the OSSC

The folder `git_remotes` contains bare git repositories that act as remote git servers.

Create your own clone of the repository in your personal folder as follows:

```
cd your_folder/
git clone /gpfs/ostor/ossc9424/homedir/git_remotes/life-sequencing-dutch.git
```

You should be able to work with this in the same way as with a remote GitHub repository.


### Setting up a new git remote server

- Make sure that all users have the right permissions. `setfacl -Rdm g:ossc9424:rwx git_remotes/`. This sets recursive `rwx` permissions on all folders created inside `git_remotes`/
- Create the remote
    ```bash
    cd git_remotes
    git init --bare --shared=group myrepo.git
    cd myrepo.git
    pwd
    ```
- Proceed with cloning as above.


## Filesystem

Available partitions and accounting on Snellius https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting. For the OSSC, one can only get full nodes.


## Working on GPU nodes


Note that slurm does not accept the two common GPU specifications
```bash
#SBATCH --gres=gpu:4
#SBATCH --gpus:4
```

Instead, for GPU, Ben suggests to always use
```bash
#SBATCH --exclusive
```

But I'm not sure how this impacts the ability to run multiple jobs in parallel, and how this works on CPU-only nodes.



### Using a single GPU

```bash
#SBATCH xyx


module load abc


export CUDA_VISIBLE_DEVICES=0

python script.py

```


```python
# script.py
import pytorch_lightning as pl

trainer = pl.trainer(strategy="ddp")

```

### Using multiple jobs with 1 GPU each

As above, export the visible device number in the bash script. Each job should see a different device id.

### Using multiple GPUs (1 node with 4 GPUs)

Unclear. We have not managed to make the  `nccl`  communication backend for torch lightning work. But `nccl` itself has worked with other packages (pytorch) on the OSSC. See also [this issue](https://github.com/odissei-lifecourse/life-sequencing-dutch/issues/8).


## slurm utilities
The following are useful to have in the `~/.bashrc` file
```bash
# squeue with more info and full job name
alias sq='squeue -o "%.18i %.30j %.8u %.2t %10M %.6D %5m %11l %11L %R"'

# recentfiles a b: list recent files in location "a" with matching pattern "b"
# useful when there are many log files
recentfiles() {
    ls -alth "$1" | grep "$2" | head
}

```
For more slurm aliases, see [here](https://gist.github.com/pansapiens/1b770fdbafa75f9aacb851d99a2aa9e2)


## Open questions

- [x] ~~I noticed a couple of times that the GPU node crashes when doing `scancel`. It then goes into `down` state and does not recover fast / maybe not even automatically. Maybe this has to do with the virtualization of the OSSC.~~ This has been work fine as of late.
