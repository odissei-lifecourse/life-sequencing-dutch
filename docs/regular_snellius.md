
This document is about the "bare-metal" snellius instance set up for collaborative development and fake data.

There is 1TB space on `/projects/0/prjs1019`.
- It is not backed up
- Permissions are controlled by us, see https://servicedesk.surf.nl/wiki/pages/viewpage.action?pageId=30660238
    - The best way to manage permissions is to set defaults, and do so recursively. This can be done with
      ```bash
      setfacl -R -m g:prjs1019:rwx ./some-directory
      setfacl -R -d --set g:prjs1019:rwx ./some-directory
      ```
      This sets the permissions for all existing files in `./some-directory` as well as a default for files created in the future.
    - This needs to be done for each directory inside the root directory `/projects/0/prjs1019`.
