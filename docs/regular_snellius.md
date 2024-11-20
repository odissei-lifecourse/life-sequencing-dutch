
This document is about the "bare-metal" snellius instance set up for collaborative development and fake data.

There is 1TB space on `/projects/0/prjs1019`.
- It is not backed up
- Permissions are controlled by us, see https://servicedesk.surf.nl/wiki/pages/viewpage.action?pageId=30660238
    - The best way to manage permissions is to set defaults, and do so recursively. This can be done with `setfacl -Rdm ...`. This sets the permissions for all existing files in `...` as well as a default for files created in the future.
    - This is not possible on the root dir. Thus, for each directory in `./projects/0/prjs1019`, one needs to `setfacl`.
    - This has been done for the `data/` directory:
        ```bash
        cd /projects/0/prjs1019
        setfacl -Rdm group:prjs1019:rwx ./data/
        ```
    - We've had some problems with the above command in the past. In the future, for any directories that are shared between users of the group `prjs1019`, do the following for allowing write-access within the group:
      ```bash
      setfacl -R -m g:prjs1019:rwx ./some-directory
      setfacl -R -d --set g:prjs1019:rwx ./some-directory
      ```
