# SQL Cheat Sheet


## CMDS
### Basic
SELECT		- grab data		\
UPDATE		- update		\
INSERT		- insert a new entry	\
DELETE		- delete a row

### Keywords / Clauses
WHERE				- for conditionals
AND, OR, NOT			- traditional logic
ORDER BY [args] ASC|DESC	- sort columns in ascending or descending order
SET				- used with update statements
LIKE				- used with WHERE to search for specified pattern

### Functions
MIN() | MAX()			- used with SELECT for min/max entries
COUNT()				- counts number of matching entries
AVG()				- average
SUM()				- summation
GETDATE()			- returns current date/time

### Commonly Used
INSERT INTO		- inserts new data into a database	\
CREATE DATABASE		\
ALTER DATABASE		\
DROP TABLE		- delete table	\
CREATE INDEX		- creates an index (search key) \
DROP INDEX		- delete an index

### Wildcard characters
Note: for MySQL	\
%	*- zero, one, or multiple characters* \
_	*- one character*
```sql
-- select all strings that start with "abc"
"abc%" 
-- all strings that contain letter "x"
"%x%"
-- all strings where 2nd letter is "z"
"_z%"
```

## Usage Examples
Select all distinct column entries
```sql
SELECT DISTINCT ColumnName FROM TableName;
```

Update an existing table record
```sql
UPDATE TableName
SET ColumnName = data, ColumnName2 = data, ...
WHERE ... ;
```

Delete a table entry
```sql
DELETE FROM TableName
WHERE ... ;
```

Select a certain number of results out of a query
*Note that this is platform dependent
```sql
--mySQL
SELECT ColumnName
FROM TableName
WHERE ...
LIMIT number;
```

Select the maximum return value for a query
```sql
SELECT MAX(columnName)
FROM tableName
WHERE ...
```

Search for entries that contains the string "sample" in the data
```sql
SELECT *
FROM TableName
WHERE ... LIKE "%sample%";
``` 

## Concept Questions

### Difference between DELETE / TRUNCATE
Truncate is non-reversible operation

### What are joins in SQL?
Four types <Inner Join / Full Join / Left Join / Right Join> \
Used to combine rows from two or more tables

### Diff between CHAR and VARCHAR2?
Both store strings, though varchar2 contains variable length strings

### What is the primary key?
Unique identifier used to identify table rows

### What are constraints?
Specify limits on the data types of the table \
Types: \
IS|NOT NULL	\
UNIQUE		\
CHECK		- satifies specific condition 			\
DEFAULT		- must adhere to a set of default values	\
INDEX		- used to create and retrieve data quickly	\

### What is denormalization?
Technique used to access data from higher to lower forms of a database	\
Improves performance

