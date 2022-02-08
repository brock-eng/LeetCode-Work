# SQL Cheat Sheet


## CMDS
### Basic
SELECT		- grab data		\
UPDATE		- update		\
INSERT		- insert a new entry	\
DELETE		- delete a row

### Most important
INSERT INTO		- inserts new data into a database	\
CREATE DATABASE		\
ALTER DATABASE		\
DROP TABLE		- delete table	\
CREATE INDEX		- creates an index (search key) \
DROP INDEX		- delete an index

## Questions

### Difference between DELETE / TRUNCATE
Truncate is non-reversible operation

### What are joins in SQL?
Four types <Inner Join / Full Join / Left Join / Right Join>
Used to combine rows from two or more tables

### Diff between CHAR and VARCHAR2?
Both store strings, though varchar2 contains variable length strings

### What is the primary key?
Unique identifier used to identify table rows

### What are constraints?
Specify limits on the data types of the table \
Types: \
NOT NULL	\
UNIQUE		\
CHECK		- satifies specific condition 			\
DEFAULT		- must adhere to a set of default values	\
INDEX		- used to create and retrieve data quickly	\

### How to get the current date?
SELECT GETDATE();

### What is denormalization?
Technique used to access data from higher to lower forms of a database	\
Improves performance

