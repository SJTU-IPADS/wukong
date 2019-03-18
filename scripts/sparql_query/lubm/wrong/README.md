# Wrong queries and their corresponding replies.

## Attention
* **EVERY** assert method has turned from abort to throw exception.
* Add `assert_error_code()` to let porxy know what happen in the engine.

## Example
|     |  command  | error | detail |
| :----| :------| :--------   |:-------- |
|q1v1 | sparql -f sparql_query/lubm/wrong/q1 -p sparql_query/lubm/wrong/manual_plan/q1v1.fmt | Unsupported triple pattern. | Unsupported triple pattern \[UNKNOWN\|KNOWN\|??\] |
|q1v2 | sparql -f sparql_query/lubm/wrong/q1 -p sparql_query/lubm/wrong/manual_plan/q1v2.fmt | Unsupported triple pattern. | Unsupported triple pattern \[CONST\|UNKNOWN\|KNOWN\].|
|q2v1 | sparql -f sparql_query/lubm/wrong/q2 | Unsupported triple pattern. | `query.hpp: get_start()` failed |
|q2v2 | sparql -f sparql_query/lubm/wrong/q2 -p sparql_query/lubm/wrong/manual_plan/q2.fmt | Const_X_X or index_X_X must be the first pattern. | `res.get_col_num() == 0` failed |
|q3 | sparql -f sparql_query/lubm/wrong/q3 -p sparql_query/lubm/wrong/manual_plan/q3.fmt | object should not be index | `pattern_group.patterns[0].predicate == PREDICATE_ID \|\| pattern_group.patterns[0].predicate == TYPE_ID` failed|
|q4 | sparql -f sparql_query/lubm/wrong/q4 -p sparql_query/lubm/wrong/manual_plan/q4.fmt | Unsupported filter type | regex error
|syntax | sparql -f sparql_query/lubm/wrong/syntax | Something wrong in the query syntax, fail to parse! | `?Y2` instead of `Y2`|

