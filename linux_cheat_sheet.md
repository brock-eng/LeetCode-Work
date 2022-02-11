# CLI Commands

###  Basic CMDS
ls		  - list files	\
mkdir 	- make directory	\
ln		  - create a link	\
touch		- create a filie	\
rm		  - remove file	\
rmdir		- remove directory\
rm -r		- remove dir w/ files	\
nohup 	- dictates that a process runs in the background	\
cat     - show contents of a text file

### Memory CMDS
df		    - show file size usage	\
free		  - show physical memory	\
vmstat -a	- virtual memory		\
sar 		  - cpu usage		\
history	  - cmd history		\

### Info CMDS
env 		      - system environment variables	\
set 		      - show all variables		\
set -o posix	- remove the functions from this variable listing	\
pwd		        - display file system total space and available space \

# Concept Questions

### What is Logical Volume?
Hierarchy:
1 - Volume Groups
2 - Physical Volumes
3 - Logical Volumes

### What is the root user?
System administrator - has all the perms for all current software

### How to reduce or shrink the size of LVM (Logical Volume Management)
1. Unmount the filesystem using <unmount>
2. Check for problems using <e2fsck -f /file>
3. Use <resiz32fs /file "3G">
4. Use <lvreduce -L 10G /file>
5. Re-mount the filesystem using <mount>

### What is swap space?
Swap space is an amount of space used by Linux to temporarily hold active programs \
Improves system performance	\
Typical size is twice the amount available of physical memory	\

### What are the kind of permissions available in Linux?
Read-Only
Write
Execute

### How to change permissions in Linux?
chmod		- change permissions of files/dir	\
chown		- change the owner of files/dir	\
chgrp 	- change group owner		\

### What are symbolic links?
Like shortcuts in windows

### What are hard links?
Points directly to a file (not folder)

### Maximum filename length?
255 Chars

### What is redirection in Linux?
Used to pass the output of an operation to another operation	\
Example <w > log>

### How to terminate a process in Linux?
Use <kill "process_name/pid"

### Which daemon tracks events on your system?
syslogd

### What is partial backup?
When you select a portion of the file hierarchy to back up

### What is Inode?
Stores informated about a file (metadata)

### Which CMD is used to set a process to execute in less time?
<nice [option] [cmd [arg]...]	\
Priority ranges from -20 to 19
