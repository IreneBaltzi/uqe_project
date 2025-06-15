
class Kernel:
    def __init__(self, kind, config, fuse_with=None):
        self.kind = kind  
        self.config = config  
        self.fuse_with = fuse_with  

    def __repr__(self):
        return f"Kernel({self.kind}, fuse_with={self.fuse_with}, config={self.config})"

def plan_query(parsed_query):
    _, select_clause, _, where_clause, groupby_clause, orderby_clause, limit_clause = parsed_query
    plan = []

    if groupby_clause:
       
        if select_clause:
            plan.append(Kernel("GROUPBY", {
                "attribute": groupby_clause[1],
                "condition": where_clause[1] if where_clause else None
            }, fuse_with="SELECT"))
        else:
            plan.append(Kernel("GROUPBY", {
                "attribute": groupby_clause[1],
                "condition": where_clause[1] if where_clause else None
            }))
    
    elif limit_clause:
        
        if where_clause:
            plan.append(Kernel("LIMIT", {
                "condition": where_clause[1],
                "limit": int(limit_clause[1])
            }))
        else:
            raise ValueError("LIMIT without WHERE is not supported in UQE")

    elif where_clause:
        
        plan.append(Kernel("WHERE", {
            "condition": where_clause[1]
        }))

    elif select_clause:
        
        plan.append(Kernel("SELECT", {
            "expression": select_clause
        }))

    if orderby_clause:
        plan.append(Kernel("ORDERBY", {
            "attribute": orderby_clause[1]
        }))

    return plan
