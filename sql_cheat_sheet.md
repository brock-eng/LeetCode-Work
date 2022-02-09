# SQL Cheat Sheet


## CMDS
### Basic
SELECT		- grab data		\
UPDATE		- update		\
INSERT		- insert a new entry	\
DELETE		- delete a row

### Keywords / Clauses
WHERE				- for conditionals	\
AND, OR, NOT			- traditional logic	\
ORDER BY [args] ASC|DESC	- sort columns in ascending or descending order	\
SET				- used with update statements	\
LIKE				- used with WHERE to search for specified pattern	\
IN				- used with WHERE to specify multiple search terms	\
BETWEEN				- used with WHERE to specify two limits	\
AS				- used to alias items	\
UNION				- combine two select statements

### Functions
*Not all available functions listed, just the more commonly seen ones*	\
MIN() | MAX()			- used with SELECT for min/max entries	\
COUNT()				- counts number of matching entries	\
AVG()				- average	\
SUM()				- summation	\
GETDATE()			- returns current date/time	\
CONCAT()			- concatenate strings	\

### Commonly Used
INSERT INTO		- inserts new data into a database	\
CREATE DATABASE		\
ALTER DATABASE		\
DROP TABLE		- delete table	\
CREATE INDEX		- creates an index (search key) \
DROP INDEX		- delete an index

### Wildcard characters
Similar to regex notation			\
%	 |zero, one, or multiple characters 	\
_	 |one character				\
[chars]  |select multiple characters		\
!	 |not operator
```sql
-- for mysql
-- select all strings that start with "abc"
"abc%" 
-- all strings that contain letter "x"
"%x%"
-- all strings where 2nd letter is "z"
"_z%"
-- all strings where 1st letter is either "a", "b", or "c"
"[abc]%"
-- all strings where 1st letter is NOT "a", "b", or "c"
"[!abc]%
```

## Usage Examples
**Select** all distinct column entries
```sql
SELECT DISTINCT ColumnName FROM TableName;
```

**Update** an existing table record
```sql
UPDATE TableName
SET ColumnName = data, ColumnName2 = data, ...
WHERE ... ;
```

**Delete** a table entry
```sql
DELETE FROM TableName
WHERE ... ;
```

**Limit** the number of results out of a query	\
*Note that this is platform dependent*
```sql
--mySQL
SELECT ColumnName
FROM TableName
WHERE ...
LIMIT number;
```

Select the **maximum** (or minimum) return value for a query
```sql
SELECT MAX(columnName)
FROM tableName
WHERE ...
```

Search for entries that **contains** the string "sample" in the data
```sql
SELECT *
FROM TableName
WHERE ... LIKE "%sample%";
``` 

Search for entries where the columnValue is **in** another table
```sql
SELECT * FROM tableName
WHERE columnName IN (SELECT columnName FROM alternateTable);
```

Search and **sort** entries where value is **between** two values
```sql
SELECT * FROM tableName
WHERE columnName BETWEEN value1 AND value2
ORDER BY columnName;
```

Search using **aliasing** to trim code footprint
```sql
SELECT t.Column, t2.Column2
FROM tableName AS t, tableName2 AS t2
WHERE ...
```

Sample **JOIN** statement: pair customer data with order data	\
Selects data where customer table and order table both contain customerID	
```sql
SELECT Orders.OrderID, Customers.CustomerName
FROM Orders
INNER JOIN Customer ON Orders.CustomerID = Customers.CustomerID
```

## Concept Questions

### Difference between DELETE / TRUNCATE
Truncate is non-reversible operation

### What are joins in SQL?
Used to combine records from two or more tables w/ matching column	\
Four types 
1. Inner Join - Returns records with matching values in both tables				
2. Left Join - Returns records from the left table, and the matched records from the right	
3. Right Join - Returns records from the right table, and the matched records from the left	
4. Cross Join - Returns all records from both tables

### Diff between CHAR and VARCHAR2?
Both store strings, though varchar2 contains variable length strings

### What is the primary key?
Unique identifier used to identify table rows

### What are constraints?
Specify limits on the data types of the table \
Types: 		\
IS|NOT NULL	\
UNIQUE		\
CHECK		- satifies specific condition 			\
DEFAULT		- must adhere to a set of default values	\
INDEX		- used to create and retrieve data quickly	\

### What is denormalization?
Technique used to access data from higher to lower forms of a database	\
Improves performance

### What are Information Schema views?
Used to obtain metadata from SQL tables, i.e. all column headers	\
Sample query:
```sql
SELECT *
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'tableName';
```
Different metadata fields that can be accessed:
```sql
INFORMATION_SCHEMA.(...	)
    CHECK_CONSTRAINTS
    COLUMN_DOMAIN_USAGE
    COLUMN_PRIVILEGES
    COLUMNS
    CONSTRAINT_COLUMN_USAGE
    CONSTRAINT_TABLE_USAGE
    DOMAIN_CONSTRAINTS
    DOMAINS
    KEY_COLUMN_USAGE
    PARAMETERS
    REFERENTIAL_CONSTRAINTS
    ROUTINES
    ROUTINE_COLUMNS
    SCHEMATA
    TABLE_CONSTRAINTS
    TABLE_PRIVILEGES
    TABLES
    VIEW_COLUMN_USAGE
    VIEW_TABLE_USAGE
    VIEWS
```
