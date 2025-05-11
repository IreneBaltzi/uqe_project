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
t_NL_LITERAL = r'"((?:\\.|[^"\\])*)"'
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

start = 'uql_query' # for the parser to start

# Rules
def p_uql_query(p):
    '''uql_query : select_clause from_clause
                 | select_clause from_clause optional_clause_combo'''
    if len(p) == 3:
        p[0] = ('QUERY', p[1], p[2])
    else:
        p[0] = ('QUERY', p[1], p[2], p[3])

def p_select_clause(p):
    'select_clause : SELECT select_expression'
    p[0] = ('SELECT', p[2])

def p_select_expression(p):
    '''select_expression : select_expression SEPARATOR select_literal
                          | select_literal'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_select_literal(p):
    '''select_literal : ALL
                      | variable_literal
                      | nl_literal
                      | aggregation
                      | INTEGER'''
    p[0] = p[1]

def p_aggregation(p):
    '''aggregation : agg_op LEFT_PARENTHESIS VAR_NAME RIGHT_PARENTHESIS
                   | agg_op LEFT_PARENTHESIS ALL RIGHT_PARENTHESIS
                   | agg_op LEFT_PARENTHESIS VAR_NAME RIGHT_PARENTHESIS AS VAR_NAME
                   | agg_op LEFT_PARENTHESIS ALL RIGHT_PARENTHESIS AS VAR_NAME'''
    if len(p) == 5:
        p[0] = ('AGG', p[1], p[3])
    else:
        p[0] = ('AGG_AS', p[1], p[3], p[6])

def p_agg_op(p):
    '''agg_op : AVG
              | COUNT
              | SUM'''
    p[0] = p[1]

def p_variable_literal(p):
    '''variable_literal : VAR_NAME
                        | VAR_NAME AS VAR_NAME'''
    p[0] = ('VAR', p[1]) if len(p) == 2 else ('VAR_AS', p[1], p[3])

def p_nl_literal(p):
    '''nl_literal : NL_LITERAL
                  | NL_LITERAL AS VAR_NAME'''
    p[0] = ('NL', p[1]) if len(p) == 2 else ('NL_AS', p[1], p[3])

def p_from_clause(p):
    'from_clause : FROM VAR_NAME'
    p[0] = ('FROM', p[2])

def p_optional_clause_combo(p):
    '''optional_clause_combo : optional_clause_combo optional_clause
                              | optional_clause'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_optional_clause(p):
    '''optional_clause : limit_clause
                        | to_clause
                        | where_clause
                        | group_by_clause
                        | order_by_clause'''
    p[0] = p[1]

def p_limit_clause(p):
    'limit_clause : LIMIT INTEGER'
    p[0] = ('LIMIT', int(p[2]))

def p_to_clause(p):
    'to_clause : TO VAR_NAME'
    p[0] = ('TO', p[2])

def p_where_clause(p):
    'where_clause : WHERE where_expression'
    p[0] = ('WHERE', p[2])

def p_where_expression(p):
    '''where_expression : where_expression AND predicate
                        | where_expression OR predicate
                        | predicate'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = (p[2], p[1], p[3])

def p_group_by_clause(p):
    'group_by_clause : GROUP BY group_by_expression'
    p[0] = ('GROUP_BY', p[3])

def p_group_by_expression(p):
    '''group_by_expression : group_by_expression SEPARATOR group_by_literal
                            | group_by_literal'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_group_by_literal(p):
    '''group_by_literal : variable_literal
                        | nl_literal'''
    p[0] = p[1]

def p_order_by_clause(p):
    '''order_by_clause : ORDER BY order_by_expression
                       | ORDER BY order_by_expression DESC'''
    if len(p) == 4:
        p[0] = ('ORDER_BY', p[3])
    else:
        p[0] = ('ORDER_BY_DESC', p[3])

def p_order_by_expression(p):
    '''order_by_expression : order_by_expression SEPARATOR order_by_literal
                            | order_by_literal'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_order_by_literal(p):
    '''order_by_literal : VAR_NAME
                        | NL_LITERAL
                        | INTEGER'''
    p[0] = p[1]

def p_predicate(p):
    '''predicate : NL_LITERAL
                 | VAR_NAME COMPARE_OPERATOR NL_LITERAL
                 | VAR_NAME COMPARE_OPERATOR INTEGER'''
    if len(p) == 2:
        p[0] = ('PRED', p[1])
    else:
        p[0] = ('PRED_OP', p[1], p[2], p[3])

def p_error(p):
    if p:
        print(f"Syntax error at token '{p.value}' (type: {p.type})")
    else:
        print("Syntax error at end of input")

parser = yacc.yacc()

def parse_query(query):
    return parser.parse(query)

if __name__ == '__main__':
    test_query = 'SELECT derived_attribute, COUNT(*) FROM reviews GROUP BY "extract attribute" AS derived_attribute LIMIT 10'

    print("Tokens:")
    lexer.input(test_query)
    while True:
        tok = lexer.token()
        if not tok:
            break
        print(f"type={tok.type:16} value={tok.value}")

    result = parse_query(test_query)
    print("\nParse result:")
    print(result)