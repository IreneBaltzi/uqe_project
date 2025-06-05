import ply.lex as lex
import ply.yacc as yacc

# Lexer 
reserved = {
    'select': 'SELECT',
    'from': 'FROM',
    'where': 'WHERE',
    'as': 'AS',
    'limit': 'LIMIT',
    'group': 'GROUP',
    'order': 'ORDER',
    'by': 'BY',
    'to': 'TO',
    'and': 'AND',
    'or': 'OR',
    'count': 'COUNT',
    'avg': 'AVG',
    'sum': 'SUM',
    'desc': 'DESC',
}

tokens = [
    'SEPARATOR', 'ALL', 'NL_LITERAL', 'VAR_NAME',
    'TABLE_URL', 'COMPARE_OPERATOR', 'INTEGER', 'FLOAT',
    'LEFT_PARENTHESIS', 'RIGHT_PARENTHESIS',
] + list(reserved.values())

t_SEPARATOR = r','
t_ALL = r'\*'
t_COMPARE_OPERATOR = r'(<>|>=|<=|!=|>|<|=)'
t_INTEGER = r'[-+]?\d+'
t_FLOAT = r'[-+]?[0-9]*\.[0-9]+'
t_LEFT_PARENTHESIS = r'\('
t_RIGHT_PARENTHESIS = r'\)'
t_NL_LITERAL = r'"((?:\\.|[^"\\])*)"'   # captures text inside double quotes
t_ignore = ' \t'

def t_VAR_NAME(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*(\.[a-zA-Z_][a-zA-Z_0-9]*)*'
    t.type = reserved.get(t.value.lower(), 'VAR_NAME')
    return t

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

# Parser

start = 'uql_query'  # entry point for the parser

def p_uql_query(p):
    '''
    uql_query : select_clause from_clause
              | select_clause from_clause optional_clause_combo
    '''

    where_clause   = None
    groupby_clause = None
    orderby_clause = None
    limit_clause   = None

    select_clause = p[1]
    from_clause   = p[2]

    if len(p) == 3:
        p[0] = (
            'QUERY',
            select_clause,
            from_clause,
            where_clause,
            groupby_clause,
            orderby_clause,
            limit_clause
        )
    else:
        # p[3] is a list of one or more optional_clause items
        optional_list = p[3]
        # Iterate through each clause in that list and assign to the correct slot
        for clause in optional_list:
            ctype = clause[0]
            if ctype == 'WHERE':
                where_clause = clause
            elif ctype in ('GROUP_BY', 'GROUP_BY_AS'):
                # GROUP_BY or GROUP_BY_AS
                groupby_clause = clause
            elif ctype in ('ORDER_BY', 'ORDER_BY_DESC'):
                orderby_clause = clause
            elif ctype == 'LIMIT':
                limit_clause = clause
            # We ignore TO or other clauses, since IMDB-only doesn't need them.
            # If you want to support TO or additional clauses, handle them here.
        p[0] = (
            'QUERY',
            select_clause,
            from_clause,
            where_clause,
            groupby_clause,
            orderby_clause,
            limit_clause
        )

def p_select_clause(p):
    'select_clause : SELECT select_expression'
    # p[2] is a list of select literals, e.g. [ ('AGG_AS','COUNT','*','cnt'), ('NL','"foo"') ]
    p[0] = ('SELECT', p[2])

def p_select_expression(p):
    '''
    select_expression : select_expression SEPARATOR select_literal
                      | select_literal
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_select_literal(p):
    '''
    select_literal : ALL
                   | variable_literal
                   | nl_literal
                   | aggregation
                   | INTEGER
    '''
    # Could be '*' or column name (VAR, VAR_AS), NL literal, or an aggregate
    p[0] = p[1]

def p_aggregation(p):
    '''
    aggregation : agg_op LEFT_PARENTHESIS VAR_NAME RIGHT_PARENTHESIS
                | agg_op LEFT_PARENTHESIS ALL RIGHT_PARENTHESIS
                | agg_op LEFT_PARENTHESIS VAR_NAME RIGHT_PARENTHESIS AS VAR_NAME
                | agg_op LEFT_PARENTHESIS ALL RIGHT_PARENTHESIS AS VAR_NAME
    '''
    # Four cases: COUNT(x), COUNT(*), COUNT(x) AS alias, COUNT(*) AS alias
    if len(p) == 5:
        # ('AGG', 'COUNT', 'column_or_*')
        p[0] = ('AGG', p[1], p[3])
    else:
        # ('AGG_AS', 'COUNT', 'column_or_*', 'alias')
        p[0] = ('AGG_AS', p[1], p[3], p[6])

def p_agg_op(p):
    '''
    agg_op : AVG
           | COUNT
           | SUM
    '''
    p[0] = p[1]

def p_variable_literal(p):
    '''
    variable_literal : VAR_NAME
                     | VAR_NAME AS VAR_NAME
    '''
    if len(p) == 2:
        p[0] = ('VAR', p[1])
    else:
        p[0] = ('VAR_AS', p[1], p[3])

def p_nl_literal(p):
    '''
    nl_literal : NL_LITERAL
               | NL_LITERAL AS VAR_NAME
    '''
    # NL_LITERAL already includes the quotes; p[1] is something like '"some text"'
    if len(p) == 2:
        p[0] = ('NL', p[1])
    else:
        p[0] = ('NL_AS', p[1], p[3])

def p_from_clause(p):
    'from_clause : FROM VAR_NAME'
    p[0] = ('FROM', p[2])

### EDIT: We keep optional clauses together, but will split them out in p_uql_query.

def p_optional_clause_combo(p):
    '''
    optional_clause_combo : optional_clause_combo optional_clause
                          | optional_clause
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_optional_clause(p):
    '''
    optional_clause : limit_clause
                    | to_clause
                    | where_clause
                    | group_by_clause
                    | order_by_clause
    '''
    p[0] = p[1]

def p_limit_clause(p):
    'limit_clause : LIMIT INTEGER'
    p[0] = ('LIMIT', int(p[2]))

def p_to_clause(p):
    'to_clause : TO VAR_NAME'
    p[0] = ('TO', p[2])

def p_where_clause(p):
    'where_clause : WHERE where_expression'
    # p[2] can be a nested predicate tree: ('PRED', text) or ('PRED_OP', col, op, literal)
    p[0] = ('WHERE', p[2])

def p_where_expression(p):
    '''
    where_expression : where_expression AND predicate
                     | where_expression OR predicate
                     | predicate
    '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        # p[2] is 'AND' or 'OR'; build a binary tree: ( 'AND', left_expr, right_expr )
        p[0] = (p[2], p[1], p[3])

### EDIT: Rewrite group_by_clause to emit either ('GROUP_BY', [list]) or
### a more specific ('GROUP_BY_AS', nl_text, alias) when the literal is NL_AS.

def p_group_by_clause(p):
    'group_by_clause : GROUP BY group_by_expression'
    # p[3] is a list of group_by_literal items, but for IMDB we typically only have a single NL literal.
    literals = p[3]
    if len(literals) != 1:
        # For simplicity, only allow exactly one grouping expression in IMDB
        raise SyntaxError("Only a single GROUP BY expression is supported in IMDB.")
    lit = literals[0]
    if lit[0] == 'NL_AS':
        # ('NL_AS', '"some text"', alias)
        nl_text, alias = lit[1], lit[2]
        p[0] = ('GROUP_BY_AS', nl_text, alias)
    elif lit[0] == 'NL':
        # ('NL', '"some text"')
        nl_text = lit[1]
        # If no alias is provided, we can auto-generate one or keep None.
        # Here we choose to auto-generate: alias = nl_text.strip('"').replace(' ', '_')
        alias = nl_text.strip('"').replace(' ', '_')
        p[0] = ('GROUP_BY_AS', nl_text, alias)
    elif lit[0] in ('VAR', 'VAR_AS'):
        # group by a column name (not likely for IMDB, but handle anyway)
        if lit[0] == 'VAR_AS':
            _, colname, alias = lit
            p[0] = ('GROUP_BY_AS', colname, alias)
        else:
            _, colname = lit
            p[0] = ('GROUP_BY_AS', colname, colname)
    else:
        raise SyntaxError(f"Unsupported GROUP BY literal: {lit}")

def p_group_by_expression(p):
    '''
    group_by_expression : group_by_expression SEPARATOR group_by_literal
                        | group_by_literal
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_group_by_literal(p):
    '''
    group_by_literal : variable_literal
                    | nl_literal
    '''
    p[0] = p[1]

def p_order_by_clause(p):
    '''
    order_by_clause : ORDER BY order_by_expression
                    | ORDER BY order_by_expression DESC
    '''
    # We capture both ORDER_BY and ORDER_BY_DESC
    if len(p) == 4:
        p[0] = ('ORDER_BY', p[3])
    else:
        p[0] = ('ORDER_BY_DESC', p[3])

def p_order_by_expression(p):
    '''
    order_by_expression : order_by_expression SEPARATOR order_by_literal
                        | order_by_literal
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_order_by_literal(p):
    '''
    order_by_literal : VAR_NAME
                     | NL_LITERAL
                     | INTEGER
    '''
    p[0] = p[1]

def p_predicate(p):
    '''
    predicate : NL_LITERAL
              | VAR_NAME COMPARE_OPERATOR NL_LITERAL
              | VAR_NAME COMPARE_OPERATOR INTEGER
    '''
    if len(p) == 2:
        # Pure NL_LITERAL predicate, e.g. WHERE "some text"
        p[0] = ('PRED', p[1])
    else:
        # Structured predicate, e.g. WHERE rating >= "7"
        p[0] = ('PRED_OP', p[1], p[2], p[3])

def p_error(p):
    if p:
        print(f"Syntax error at token '{p.value}' (type: {p.type})")
    else:
        print("Syntax error at end of input")

parser = yacc.yacc()

def parse_query(query: str):
    """
    Returns a 7-tuple:
      ('QUERY',
       select_clause,
       from_clause,
       where_clause_or_None,
       groupby_clause_or_None,
       orderby_clause_or_None,
       limit_clause_or_None)
    """
    return parser.parse(query)

if __name__ == '__main__':
    test_queries = [
        'SELECT derived_attribute, COUNT(*) FROM reviews GROUP BY "extract attribute" AS derived_attribute LIMIT 10',
        'SELECT COUNT(*) AS cnt FROM movie_reviews WHERE "the review is positive"',
        'SELECT * FROM movie_reviews WHERE "the review is positive" LIMIT 50'
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = parse_query(q)
        print("Parse result:")
        print(result)
